# https://github.com/teddylee777/langchain-kr/blob/main/19-Streamlit/01-MyProject/pages/01_PDF.py

import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
# from langchain_teddynote.prompts import load_prompt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from langchain_teddynote import logging
from dotenv import load_dotenv
import os


# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("PDF 기반 QA💬")

# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화기록을 저장하기 위한 용도로 생성한다.
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    # 아무런 파일을 업로드 하지 않을 경우
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메시지를 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장합니다.
    file_content = file.read()
    file_path = f".cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # 단계 1: 문서 로드(Load Documents)
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성 : 이 부분도 바꿈
    # from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device": "cpu"} # "cuda"
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local(".cache/embeddings/vectorstore.pkl") # 내가 추가한 부분 # 로컬에 벡터db저장

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever

@st.cache_resource(show_spinner="기존의 벡터DB를 로드 중입니다...") # 추가함 start
def load_existing_vector_db():
    # 기존 임베딩이 저장된 경로를 로드
    vectorstore_path = ".cache/embeddings/vectorstore.pkl"

    # 기존 벡터스토어 로드
    if os.path.exists(vectorstore_path):
        vectorstore = FAISS.load_local(vectorstore_path) 
        retriever = vectorstore.as_retriever()
        return retriever
    else:
        return None
### 이까지 추가 end

# 체인 생성
def create_chain(retriever, model_name="gemma2"): # 모델 바꿈 gemma2로
    # 단계 6: 프롬프트 생성(Create Prompt)
    # 프롬프트를 생성합니다.
    from langchain import hub # 프롬프트도 일단 간단하게 설정 : 나중에 바꿔야 함 ㅇㅇ
    prompt = hub.pull("rlm/rag-prompt")

    # 단계 7: 언어모델(LLM) 생성
    # 모델(LLM) 을 생성합니다.
    from langchain_ollama import ChatOllama

    llm = ChatOllama(
        model=model_name,
        temperature=0,
        # other params...
    )

    # 단계 8: 체인(Chain) 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# 벡터 DB 로드 or 생성 : 내가 추가 start
retriever = None
if uploaded_file:
    # 파일 업로드 시, 벡터 DB생성
    retriever = embed_file(uploaded_file)
else:
    # 기존 벡터DB 로드
    retriever = load_existing_vector_db()

if retriever is not None:
    chain = create_chain(retriever, model_name="gemma2")
    st.session_state["chain"] = create_chain
else:
    st.write("벡터 DB를 생성하려면 먼저 PDF파일을 업로드하세요.") 

# # 파일이 업로드 되었을 때 : 테디꺼지만 일단 보류
# if uploaded_file:
#     # 파일 업로드 후 retriever 생성 (작업시간이 오래 걸릴 예정...)
#     retriever = embed_file(uploaded_file)
#     chain = create_chain(retriever, model_name="gemma2")
#     st.session_state["chain"] = chain

# 초기화 버튼이 눌리면...
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화 기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 경고 메시지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약에 사용자 입력이 들어오면...
if user_input:
    # chain 을 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장한다.
        add_message("user", user_input)
        add_message("assistant", ai_answer)
    else:
        # 파일을 업로드 하라는 경고 메시지 출력
        warning_msg.error("파일을 업로드 해주세요.")