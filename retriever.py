import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma


def create_retriever(file_path):

    # 단계 1: 문서 로드(Load Documents)
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 단계 2: 문서 분할(Split Documents)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    split_documents = text_splitter.split_documents(docs)

    # 단계 3: 임베딩(Embedding) 생성
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device":"cpu"}
    encode_kwargs = {"normalize_embeddings":True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs,encode_kwargs=encode_kwargs)

    # 단계 4: DB 생성(Create DB) 및 저장
    # 벡터스토어를 생성합니다.
        # 이게 데모용으로 빠르게 확인하기 좋아서 쓰는거지, 다른걸로 꼭 봐꿔줘야 함
    vectorstore = Chroma.from_documents(documents=split_documents, embedding=embeddings)

    # 단계 5: 검색기(Retriever) 생성
    # 문서에 포함되어 있는 정보를 검색하고 생성합니다.
    retriever = vectorstore.as_retriever()
    return retriever