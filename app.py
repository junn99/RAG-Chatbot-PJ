import os
import json

import streamlit as st
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.document_loaders import UnstructuredFileLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

working_dir = os.path.dirname(os.path.abspath(__file__))
GROQ_API_KEY = ""

def setup_vectorstore():
    persist_directory = f"{working_dir}/vector_db_dir"

    model_name = "BAAI/bge-m3"
    model_kwargs = {"device":"cpu"}
    encode_kwargs = {"normalize_embeddings":True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs
)
    if os.path.exists(persist_directory):
        # Use existing ChromaDB
        return Chroma(persist_directory=persist_directory, embedding_function=embeddings,collection_name="chroma_db")
    else:
        # Create new ChromaDB
        loader = DirectoryLoader(path="/home/jun/my_project/langchain_tutorial/data", glob="./*.pdf", loader_cls=UnstructuredFileLoader)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        text_chunks = text_splitter.split_documents(documents)
        
        vectordb = Chroma.from_documents(
            documents=text_chunks,
            embedding=embeddings,
            collection_name="chroma_db",
            persist_directory=persist_directory
        )
        print("Documents Vectorized and stored in ChromaDB")
        return vectordb

def setup_chain(vectorstore):
    llm = ChatGroq(model="llama-3.1-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever()
    
    template = """Answer the question based on the following context:
    {context}
    
    Question: {question}
    
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain, memory

st.set_page_config(
    page_title="Multi Doc Chat",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Multi Documents Chatbot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = setup_vectorstore()

if "lcel_chain" not in st.session_state or "memory" not in st.session_state:
    st.session_state.lcel_chain, st.session_state.memory = setup_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask AI...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.lcel_chain.invoke(user_input)
        assistant_response = response.content
        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        
        # Update memory
        st.session_state.memory.chat_memory.add_user_message(user_input)
        st.session_state.memory.chat_memory.add_ai_message(assistant_response)
