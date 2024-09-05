import streamlit as st
from utils import print_messages, StreamHandler
# from vectordb import load_or_create_vector_store

from langchain_core.messages import ChatMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from langchain_ollama import ChatOllama

# history chat
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

st.set_page_config(page_title="AI Trend Bot",page_icon="💬")
st.title("💬AI Trend Bot")

# 입력한 메세지들을 기록할 필요가 있음 -> session state = 캐싱기능
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 채팅 대회기록을 저장하는 store 세션 상태 변수
if "store" not in st.session_state:
    st.session_state["store"] = dict()

with st.sidebar:
    session_id = st.text_input("Session ID", value="abc123")
    # 세션 id? -> 카톡당 id -> 새로운 채팅으로 생각하면 될듯

    clear_button = st.button("대화기록 초기화")
    if clear_button:
        st.session_state["messages"] = [] # 그냥 웹상에서 정리
        # st.session_state["store"] = dict() # 이거 활성화하면 아예 기록까지 초기화
        st.experimental_rerun()

# 이전 대화기록을 출력해주는 코드
# print_messages()

# 데이터 로드 및 벡터DB 저장
# db = load_or_create_vector_store()
# retriever
# retriever = db.as_retriever(search_kwargs={"k": 2})

# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids: str)-> BaseChatMessageHistory:
    if session_ids not in st.session_state["store"]: # 세션 ID가 store에 없는 경우
        # 새로운 ChatNessageHistory 객체를 생성하여 store에 저장
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids] # 해당 세션 ID에 대한 세션 기록 반환

# 채팅기능
# "assistant" & "user" : 채팅 아이콘
user_input = st.chat_input("메세지를 입력해주세요.") # 지시문
if user_input:
    # 사용자가 입력한 내용
    st.chat_message("user").write(user_input)
    st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

    # LLM을 사용하여 AI의 답변 생성
    # 프람프트 -> 모델 -> 체인 + 아웃풋파서 -> 인보크 = msg로 들어가면 됨

    # 1. 모델 생성 wiht ollama
    llm = ChatOllama(
    model="gemma2",
    temperature=0.5
    # other params...
)
    
    # 2. 프롬프트 생성
    # 직접 만들어야 겟네 -> hub에서 pull하는 건 안되겠다.
    # 여기다가 한국어로 답변 + 페르소나까지 해서 추가해야 할 듯.
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "질문에 짧고 간결하게 답변해 주세요."
            ),
            # 대화 기록을 변수로 사용, history가 MessageHistory의 key가 됨.
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"), # 사용자 질문을 입력 
        ]
    )

    chain = prompt | llm # chain거는게 좀 빡세네 prompt에 맞춰서 먼가 해야할 것 같은데 와우

    chain_with_memory = (
    RunnableWithMessageHistory( # RunnableWithMessageHistory객체 생성
        chain, # 실행할 Runnable 객체
        get_session_history, # 세션 기록을 가져오는 함수
        input_messages_key="question", # 사용자 질문의 키
        history_messages_key="history", # 기록 메세지의 키
    )
)

    # response = chain.invoke({"question": user_input})
    response = chain_with_memory.invoke(
        {"question":user_input},
        # 세션 ID설정
        config = {"configurable":{"session_id":session_id}}
    )
    

    msg = response.content

    # AI의 답변
    with st.chat_message("assistant"):
        stream_hander = StreamHandler(st.empty())
        st.write(msg)
        st.session_state["messages"].append(ChatMessage(role="assistant", content=msg))