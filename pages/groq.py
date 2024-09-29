import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
import time
import os
from dotenv import load_dotenv

from retriever import create_retriever

# API KEY 로드
load_dotenv()

# 캐시 디렉토리 생성
    # 앞에 .이 있으면 숨김폴더로 운용
if not os.path.exists(".cache"):
    os.mkdir(".cache")
# 파일 업로드 전용 폴더(임시로)
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("GROQ") # 제일 큰 글씨

# 스트림릿은 실행할 때 마다 페이지가 새로고침되는 방식
# 그래서 대화기록들도 다 날아감 -> 유일하게 이런걸 저장하는 기능이 session_state -> dict와 비슷하게 key-value로 저장

# 사이드바 
with st.sidebar: # with구문 : 하나의 컨테이너안에 컨텐츠들 입력
    # 초기화 버튼
    clear_button = st.button("대화 초기화")
    
    model = st.selectbox(
        "사용할 모델을 선택해 주세요.",
        ("llama-3.1-70b-versatile","llama-3.2-90b-text-preview","llama3-70b-8192","gemma-7b-it","gemma2-9b-it"),
        index=0
    )

    st.warning('답변이 제한되면, 다른 모델로 바꿔주세요.',icon="⚠️")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])


    # 토큰량 확인하는 것도 추가하고 싶다.

# 처음 1번만 실행하면 됨
# if "messages" not in st.session_state:
#     # 대화기록을 저장하기 위함
#     st.session_state["messages"] = []

# ChatGroq 전용 세션 상태 초기화
if "groq_messages" not in st.session_state:
    st.session_state["groq_messages"] = []
if "groq_chain" not in st.session_state:
    st.session_state["groq_chain"] = None  # ChatGroq 모델 초기화


# 대화기록 저장 함수(세션에 새로운 메시지 추가)
def add_message(role, message):
    st.session_state["groq_messages"].append(ChatMessage(role=role, content=message))

# 파일이 업로드 되었을 때
@st.cache_resource(show_spinner="업로드한 파일을 처리 중입니다...") # 캐싱된 파일을 사용하는 데코레이터
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # step 1
    # # 단계 1: 문서 로드(Load Documents)
    # loader = PyPDFLoader(file_path)
    # docs = loader.load()

    # # 단계 2: 문서 분할(Split Documents)
    # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    # split_documents = text_splitter.split_documents(docs)

    # # 단계 3: 임베딩(Embedding) 생성
    # model_name = "BAAI/bge-m3"
    # model_kwargs = {"device":"cpu"}
    # encode_kwargs = {"normalize_embeddings":True}

    # embeddings = HuggingFaceBgeEmbeddings(
    #     model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

    # # 단계 4: DB 생성(Create DB) 및 저장
    # # 벡터스토어를 생성합니다.
    #     # 이게 데모용으로 빠르게 확인하기 좋아서 쓰는거지, 다른걸로 꼭 봐꿔줘야 함
    # vectorstore = Chroma.from_documents(documents=split_documents, embedding=embeddings)

    # # 단계 5: 검색기(Retriever) 생성
    # # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    # retriever = vectorstore.as_retriever()

    # 함수화 시킴
    return  create_retriever(file_path)


# 체인 생성
def create_chain(retriever):
    # prompt | llm | output_parser
    
    # step 1
    # prompt = ChatPromptTemplate.from_messages(
    #     [
    #         ("assistant", "너는 친근한 어시스턴트야"),
    #         ("user", "#Question:\n{question}"),
    #     ]
    # )

    prompt = PromptTemplate.from_template(
        """
        Answer the question half based on the following context:
        Use what you already know!
        {context}
    
        Question: {question}
    
        If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
        Answer:
        """
    )

    llm = ChatGroq(model=model, temperature=0.7)

    output_parser = StrOutputParser() # 커스텀 파서 만들고 싶음

    # chain
    # chain = prompt | llm | output_parser # 너무 구식 chain, 정말 간단한 데모용이지 이렇게 쓸 일은 없음
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return chain


# # 이전 대화 기록 출력
# step 1
# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)

# 이전 대화 기록 출력 함수
def print_messages():
    for chat_message in st.session_state["groq_messages"]:
        # st.write(f"{chat_message.role}: {chat_message.content}") # 이건 그.. 블럭아이콘 형태로 안나오고 그냥 텍스트로만 나오게 됨.
        st.chat_message(chat_message.role).write(chat_message.content)

# 파일이 업로드 완료되면
if uploaded_file:
    # 파일 업로드 후 retriever 생성(오래걸림)
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever)
    st.session_state["groq_chain"] = chain



# 초기화 버튼이 눌리면... 대화기록 초기화 -> 그래서? 대화기록 출력하기 전에 위치시킴
if clear_button:
        st.session_state["groq_messages"] = [] # 빈리스트로 초기화

print_messages() # 호출 : 대화기록 출력


# 사용자 입력
user_input = st.chat_input("메시지 창이 생기고 그 안에 들어갈 상태메시지 입력")

# 경고 메시지를 띄우기 위한 빈 영엉
warning_message = st.empty()

if user_input: # 사용자 입력(user_input)이 들어오면
    # st.write(user_input) # 그대로 입력값을 받아서 메시지로 출력
    st.chat_message("user").write(user_input) # chat_message("~") : 틀을 만들어줌 -> 유저의 입력
    # chain 생성 for ai의 답변을 위해
    # chain = create_chain()
  
    # 매번 가져오는 게 아닌, 세션에 저장된 체인 가져옴
    chain = st.session_state["groq_chain"]

    if chain is not None:
        # st.chat_message("user").write(user_input) # 사용자입력을 여기에 붙인다면, 파일이 들어올 때 까지 입력 메시지가 안뜸 -> 경고메시지만
            # 이거 응용해서 쓰면 좋을듯, 특정 조건 만족할 때 까지 안뜨게 하는

        # answer = chain.invoke({"question":user_input})
        # #ai의 답변
        # st.chat_message("assistant").write(answer) # 이거는 답벼나올 때 까지 너무 답답하므로 streaming형식으로 받아낼거임


        # 스트리밍 답변 : 그록사용시엔 토큰 사용량을 봐야해서 보류하기로 함, 그록은 빠르니까 ㄱㅊ
        # response = chain.stream({"question":user_input}) # RunnablePassthrough 쓰면 이렇게 dict아닌, 값만 넣어줘야 함 <ERROR-CODE with RunnablePassthrough>
            # 이 형식 그대로 사용한다면 prompt에서 수정이 필요한데, "question": itemgetter("question") 이렇게 하면 됨
        response = chain.stream(user_input) # RunnablePassthrough 쓰면 이렇게 dict아닌, 값만 넣어줘야 함
        with st.chat_message("assistant"):
            # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력
            container = st.empty()
            answer = ""
            for token in response:
                answer += token
                container.markdown(answer)
    
        # # 대화기록들 세션에 저장 : 롤(user와 ai)과 내용으로 -> st.session_state["messages"] = [("user",:"안녕하세요!"), ("assistant","안녕하세요!")] 이렇게 tuple형식으로 저장되어있음
        # step 1
        # st.session_state["messages"].append(("user",user_input))
        # st.session_state["messages"].append(("assistant",user_input))
        # step 2
        # # ChatMessage로 바꾸기 -> 더 편하고 직관적이게
        # ChatMessage(role="user", content=user_input)
        # ChatMessage(role="assistant", content=user_input)

        # 여기서 더 해서 함수화 ㄱㄱ : 최종본
        add_message("user",user_input)
        add_message("assistant",answer)
    
    else:
        warning_message.error("파일을 업로드 해주세요.")


##################################
    # 토큰 사용량 띄우고싶은데, 쉽지 않다