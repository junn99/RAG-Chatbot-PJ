import streamlit as st

import os
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 여기서 로드하는 거에 따라서 if로 나눠놓고 load하고 vectordb에서 merge하는 식으로 가야하나


model_name = "BAAI/bge-m3"
model_kwargs = {"device": "cpu"} # "cpu" or "cuda"
encode_kwargs = {"normalize_embeddings": True}

@st.cache_resource
def load_or_create_vector_store():
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    faiss_index_path = 'faiss_index'

    if os.path.exists(faiss_index_path):
        # 인덱스가 존재하면 로드
        st.info("기존 FAISS 인덱스를 로드합니다.")
        vector_store = FAISS.load_local(faiss_index_path, embeddings)
    else:
        # 인덱스가 없으면 새로 생성
        st.info("FAISS 인덱스를 새로 생성합니다.")
        # PDF 파일 경로 (실제 경로로 변경 필요)
        pdf_path = "/home/jun/my_project/jun/data/AI_REPORT_2024_1_2024년_AI_이슈를_용어와_함께_쉽게_이해하기_최종수정_게시용_20240725.pdf"
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        vector_store = FAISS.from_documents(texts, embeddings)
        # 생성한 인덱스 저장
        vector_store.save_local(faiss_index_path)

    return vector_store