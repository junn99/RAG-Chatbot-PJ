# 1의 오류해결 버전
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain import hub
import os

# Constants
PDF_PATH = "/home/jun/my_project/jun/teddy/.cache/files/(IS-168) 인공지능 기술에 대한 중소기업의 인식 및 수요 조사분석.pdf"  # Update this with your PDF file path
VECTOR_STORE_PATH = ".cache/embeddings/vectorstore.pkl"
CACHE_DIR = ".cache"
EMBEDDINGS_DIR = ".cache/embeddings"

# Create cache directories
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain" not in st.session_state:
    st.session_state.chain = None

# Function to get embeddings
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

# Function to load or create vector store
@st.cache_resource(show_spinner="벡터스토어 로드... or 생성...중입니다.")
def load_or_create_vector_store():
    embeddings = get_embeddings()
    
    if os.path.exists(VECTOR_STORE_PATH):
        st.warning("기존의 벡터스토어를 로드중입니다. 신뢰할 수 있는 파일인지 확인하세요.")
        return FAISS.load_local(VECTOR_STORE_PATH, embeddings, allow_dangerous_deserialization=True)
    
    loader = PDFPlumberLoader(PDF_PATH)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)
    
    vectorstore = FAISS.from_documents(documents=split_documents, embedding=embeddings)
    vectorstore.save_local(VECTOR_STORE_PATH)
    return vectorstore

# Function to create RAG chain
@st.cache_resource(show_spinner="RAG체인 생성 중입니다..")
def create_rag_chain():
    vectorstore = load_or_create_vector_store()
    retriever = vectorstore.as_retriever()
    
    # prompt = hub.pull("rlm/rag-prompt")
    prompt = ChatPromptTemplate.from_template("""
주어진 정보를 바탕으로 사용자의 질문에 답변해주세요. 다음 지침을 따라주세요:

1. 한국어로 답변하세요.
2. 간결하고 명확하게 답변하세요.
3. 확실하지 않은 정보는 추측하지 말고 모른다고 하세요.
4. 답변은 3-4문장을 넘지 않도록 해주세요.

컨텍스트: {context}

질문: {question}

답변:
""")
    
    llm = ChatOllama(model="gemma2", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

# Streamlit UI
st.title("PDF-based Q&A 💬")

# Initialize RAG chain
if st.session_state.chain is None:
    st.session_state.chain = create_rag_chain()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("드루와"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Stream the response
        for chunk in st.session_state.chain.stream(prompt):
            full_response += chunk
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Add a button to clear the chat history
if st.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()