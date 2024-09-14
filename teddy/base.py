# 이게 성공한것, 앞으로 이걸로 ㄱㄱ

import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, ChatMessage
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_huggingface import HuggingFaceEmbeddings : 구식버전
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
import os

########################################## 변수 지정 ##################################################
PDF_PATH = "/home/jun/my_project/jun/teddy/.cache/files/(IS-168) 인공지능 기술에 대한 중소기업의 인식 및 수요 조사분석.pdf"
VECTOR_STORE_PATH = ".cache/embeddings/vectorstore.pkl"
CACHE_DIR = ".cache"
EMBEDDINGS_DIR = ".cache/embeddings"

# Sidebar
with st.sidebar:
    clear_btn = st.button("대화 초기화")
    selected_model = st.selectbox("LLM 선택", ["gemma2:2b", "gemma2"], index=0)
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")

# 캐시 & 임베딩 디렉토리 생성
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

################################### 세션 초기화 ######################################
# session_state : 웹 앱의 상태를 저장하고 관리하는 방법 = 사용자 세션 동안 데이터를 유지
    # dict와 유사하게 작동, 키-값 쌍 저장
    # streamlit은 기본적으로 실행할 때 마다, 전체 페이지 다시 실행하는데 세션을 통해 데이터 유지가능

# messages : 사용자와 챗봇 간의 대화 기록
if "messages" not in st.session_state:
    st.session_state.messages = []
# RAG chain -> 이후 create_rag_chain() 함수를 통해 실제 chain객체로 초기화
if "chain" not in st.session_state:
    st.session_state.chain = None
# store : 여러 세션의 대화 기록을 관리하는 dict
    # 각 세션 ID를 키로 사용하여 해당 세션의 ChatMessageHistory객체 저장 -> 여러 사용자(대화) 세션 관리
if "store" not in st.session_state:
    st.session_state.store = {}


# 각 세션에 대한 대화기록 관리
def get_session_history(session_id):
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

################################# 임베딩 정의 ###########################################
# 여기서 항상 오류가 많이 나옴
# 제일 많이 쓰는 임베딩함수 사용
@st.cache_resource
def get_embeddings():
    return HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"}, # cuda or cpu
        encode_kwargs={"normalize_embeddings": True}
    )

# 임베딩이 다양한 입력 형식을 처리하도록 함 : 임베딩이 항상 문자열 입력을 받도록 보장
def safe_embed_query(embedding_function, text):
    if isinstance(text, dict) and "question" in text:
        text = text["question"]
    if not isinstance(text, str):
        text = str(text)
    return embedding_function.embed_query(text)

############################### 로드 ~ 벡터스토어 생성 ############################################

# 기존것이 있으면 그대로 사용, 아니면 새로 생성
@st.cache_resource(show_spinner="벡터스토어 로드... or 생성...중입니다.")
def load_or_create_vector_store():
    embeddings = get_embeddings()
    
    if os.path.exists(VECTOR_STORE_PATH):
        # st.warning("답변 생성중입니다...@")
        vectorstore = FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        loader = PDFPlumberLoader(PDF_PATH) # 단일 파일이 아닌 폴더째로 가져오도록 수정 필요
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        split_documents = text_splitter.split_documents(docs)
        
        vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
        vectorstore.save_local(VECTOR_STORE_PATH)
    
    vectorstore._embed_query = lambda text: safe_embed_query(embeddings, text)
    
    return vectorstore

################################################### 체인생성 ####################################################

def create_rag_chain():
    vectorstore = load_or_create_vector_store()
    retriever = vectorstore.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "당신은 Question-Answering 챗봇입니다. 주어진 질문에 대한 답변을 제공해주세요."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", """
주어진 정보를 바탕으로 사용자의 질문에 답변해주세요. 다음 지침을 따라주세요:

1. 한국어로 답변하세요.
2. 간결하고 명확하게 답변하세요.
3. 확실하지 않은 정보는 추측하지 말고 모른다고 하세요.
4. 답변은 3-4문장을 넘지 않도록 해주세요.

컨텍스트: {context}

질문: {question}

답변:
""")
    ])
    
    llm = ChatOllama(model=selected_model, temperature=0)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        RunnablePassthrough.assign(
            context=retriever | format_docs
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    # RunnableWithMessageHistory : 기존 chain에 대화기록관리 기능 추가
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    
    return chain_with_history

############################################ 스트림릿 ##############################################

# Streamlit UI
st.title("대화내용을 기억하는 PDF 기반 Q&A 챗봇 💬")



# chain 객체 생성
if st.session_state.chain is None:
    st.session_state.chain = create_rag_chain()

# 대화기록 초기화 및 새로운 세션 할당
if clear_btn:
    st.session_state.messages = []
    st.session_state.store[session_id] = ChatMessageHistory()

# 대화기록 ui 생성
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input : 사용자 입력 받기
if user_input := st.chat_input("궁금한 내용을 물어보세요!"):
    st.session_state.messages.append({"role": "user", "content": user_input}) # 대화기록 추가
    with st.chat_message("user"): # streamlit에 사용자(user) 메세지 표시
        st.markdown(user_input)

    with st.chat_message("assistant"): # streamlit에 챗봇(assistant) 메세지 표시
        message_placeholder = st.empty() # 응답이 생성되는 동안 실시간으로 업데이트 될 placeholder생성
        full_response = ""
        
        # Stream the response
        try:
            for chunk in st.session_state.chain.stream(
                {"question": user_input},
                config={"configurable": {"session_id": session_id}}
            ):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            full_response = "죄송합니다. 답변 생성 중 오류가 발생했습니다."
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response}) # 대화기록 추가

if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.session_state.store[session_id] = ChatMessageHistory()
    st.rerun()