import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import time
from langchain_community.document_loaders import DirectoryLoader, PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tempfile
import shutil
from langchain_pinecone import PineconeVectorStore


# .env 파일의 환경 변수 로드
load_dotenv()

# 환경 변수에서 키 값 가져오기
groq_api_key_env = os.getenv("GROQ_API_KEY", "")
pinecone_api_key_env = os.getenv("PINECONE_API_KEY", "")
langchain_api_key_env = os.getenv("LANGCHAIN_API_KEY","")
# pinecone_env_env = os.getenv("PINECONE_ENV", "")
index_name_env = os.getenv("INDEX_NAME", "")

# Streamlit 앱 제목
st.title("Pinecone Data Manager")
# 사이드바에 API 키 입력 필드 추가
st.sidebar.header("API 설정")
groq_api_key_env = st.sidebar.text_input("GROQ API Key", type="password", value=groq_api_key_env)
pinecone_api_key = st.sidebar.text_input("Pinecone API Key", type="password",value=pinecone_api_key_env)
langchain_api_key_env = st.sidebar.text_input("Langchain API Key", type="password",value=langchain_api_key_env)
# 이건 어캐해야할 지 고민이다
    # 파인콘에서 최대 한 프로젝트에서 인덱스 5개까지라고 했으니까 "클릭" 선택으로 정할까?
    # 프로젝트 개수는 잘 모르겠다 : 프로젝트당 인덱스라 같이 묶어서 해야하는데 헷갈리네
    # 일단 보류!
# pinecone_env = st.sidebar.text_input("Pinecone Environment")
index_name = st.sidebar.text_input("Pinecone Index Name",value="pdf-rag-with-hybridsearch")

# API 키 설정 및 인덱스 초기화
if groq_api_key_env and pinecone_api_key and index_name:
    os.environ["GROQ API Key"] = groq_api_key_env
    
    # Pinecone 초기화
    pc = Pinecone()

    # HuggingFace 임베딩 초기화
    model_name = "BAAI/bge-m3"
    model_kwargs = {"device":"cpu"}
    encode_kwargs = {"normalize_embeddings":True}

    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
)
    
    # 인덱스 존재 여부 확인 및 생성
    existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]

    if index_name not in existing_indexes:
        st.info(f"인덱스 '{index_name}'가 존재하지 않습니다. 새로 생성합니다...")
        pc.create_index(
            name=index_name,
            dimension=1024,  # BAAI/bge-m3의 임베딩 차원과 맞춰야 함
            # len(embeddings.embed_query("Hello!")) # 이렇게 하는건 어떨까?
            metric="cosine", # 유사도 측정 방법을 지정 (dotproduct, euclidean, cosine)
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            # pods=1,
            # pod_type="p1.x1" # 이것들은 뭐지?
        )
        
        # 인덱스가 준비될 때까지 대기 : 이거 없어도 되나? 일단 고민
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        st.success(f"인덱스 '{index_name}'가 성공적으로 생성되었습니다.")
    
    #Pinecone 인덱스 & pineconedb 가져오기
    index = pc.Index(index_name)
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    st.sidebar.success("API 및 인덱스가 성공적으로 설정되었습니다.")

    # 파일 목록 보기 기능 (사이드바에 추가)
    st.sidebar.header("파일 목록")
    if st.sidebar.button("파일 목록 새로고침"):
        try:
            # Pinecone에서 고유한 'source' 메타데이터 가져오기
            query_response = index.query(
                vector=[0] * 1024,  # 임의의 벡터 (차원은 인덱스와 일치해야 함)
                top_k=10000,  # 충분히 큰 수를 지정
                include_metadata=True
            )
            
            # 고유한 파일 이름 추출 및 정제
            file_names = set()
            for match in query_response['matches']:
                if 'source' in match['metadata']:
                    # 파일 경로에서 순수한 파일명만 추출
                    full_path = match['metadata']['source']
                    file_name = os.path.basename(full_path)
                    file_names.add(file_name)
            
            if file_names:
                st.sidebar.write("처리된 파일 목록:")
                for file_name in sorted(file_names):
                    st.sidebar.write(file_name)
            else:
                st.sidebar.info("처리된 파일이 없습니다.")
        except Exception as e:
            st.sidebar.error(f"파일 목록을 가져오는 중 오류가 발생했습니다: {str(e)}")
else:
    st.warning("모든 API 설정을 입력해주세요.")
    st.stop()


# 파일 업로드 및 처리 기능
st.header("PDF 파일 업로드 및 처리")

# 임시 다운로드 폴더 생성
temp_download_dir = tempfile.mkdtemp()
st.info(f"임시 다운로드 폴더가 생성되었습니다: {temp_download_dir}")

uploaded_files = st.file_uploader("PDF 파일들을 선택하세요", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    if st.button("파일 업로드 및 처리"):
        for uploaded_file in uploaded_files:
            # 임시 폴더에 파일 저장
            with open(os.path.join(temp_download_dir, uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())

        # DirectoryLoader를 사용하여 PDF 파일 로드
        loader = DirectoryLoader(temp_download_dir, glob="**/*.pdf", loader_cls=PDFPlumberLoader)
        documents = loader.load()

        # 텍스트 스플리터 설정
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)

        # Pinecone에 데이터 업로드
        vector_store.from_documents(texts, embeddings, index_name=index_name)

        # 처리된 파일을 원래 위치로 이동 (예: 'processed' 폴더)
        processed_dir = os.path.join(os.path.dirname(temp_download_dir), "processed")
        os.makedirs(processed_dir, exist_ok=True)
        for filename in os.listdir(temp_download_dir):
            shutil.move(os.path.join(temp_download_dir, filename), os.path.join(processed_dir, filename))

        st.success(f"모든 PDF 파일이 성공적으로 처리되고 Pinecone에 업로드되었습니다.")
        st.info(f"처리된 파일은 다음 위치로 이동되었습니다: {processed_dir}")

        # 임시 다운로드 폴더 삭제
        shutil.rmtree(temp_download_dir)
        