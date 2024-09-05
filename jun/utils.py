import streamlit as st

# 이전 대화기록을 출력해주는 코드
def print_messages():
    if "messages" in st.session_state and len(st.session_state['messages']) > 0:
        for chat_message in st.session_state["messages"]:
            st.chat_message(chat_message.role).write(chat_message.content)
            # ChatMessage객체이므로 .으로 각각 옵션들 표현



# 스트리밍으로 출력하는 코드 -> 그냥 쓰면 됨 ㅇㅇ
    # 올라마는 그 기능 없는 것 같은데
from langchain_core.callbacks.base import BaseCallbackHandler

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text
    
    def on_llm_new_token(self, token:str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)