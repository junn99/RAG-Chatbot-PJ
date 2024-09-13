항상 하기 전에 가상환경 세팅부터 하고 가자!
- 이번엔 파이썬 버전 3.11부터임 ㅇㅇ

# 환경 설정
1. 가상환경 만들기 : conda create --name teddy python=3.11
    - 삭제 : conda remove --name teddy --all + 다른 가상환경 활성화한다음 해야함
2. 가상환경 활성화 : conda activate teddy
3. 작업할 폴더로 이동 : cd /home/jun/my_project/jun/teddy
4. ls & pwd로 파일 확인
5. 패키지 설치 : pip install -r requirements.txt  -> 아니 이거 어떻게 알지?

# 실행
그 일단 ollama로 gemma2랑 gemma2:2b 로컬에 다운받기 
-> 일단? 2개의 성능차이가 크게 안느껴짐 -> 로컬에서 할 땐 2b짜리가 훨씬 빨라서 이걸로 사용
streamlit run app.py 실행 ㄱㄱ = 이거는 메모리 기능 없는 거
add memory = 메모리만 구현한 거
streamlit run app2.py 실행 = 메모리 기능 추가 -> 이걸로 하면 됨!
