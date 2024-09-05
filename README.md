# RAG-Chatbot-PJ

## 패키지 설치(첫 세팅할때만)
1. 가상환경 먼저 만들고 깔끔하게 설치하자! 충돌남 : conda create --name myenv python=3.10
2. conda activate myenv
3. pip install -r requirements.txt
4. ollama 설치 : curl -fsSL https://ollama.com/install.sh | sh ->  ollama pull gemma2
5. streamlit run app.py

### 환경 세팅 다했다면..
1. ollama server
2. ollama list : 다운받은 모델 확인!
3. streamlit run app.py
* ollama model 삭제 : ollama rm gemma2

2024/9/5
- 스트림릿 베이스라인 구현 : ollama 모델 pull해서 답변받기 + 메모리 기억
- but? : ISUEE = local ubuntu환경에서 진행하는데, 스트림릿에서 모델 용량이 환경 메모리보다 작아서 실행히 안됨
- 코드는 둘 다 해야함! 함수 import한 것들 있어서
