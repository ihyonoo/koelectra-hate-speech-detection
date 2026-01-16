# BLEP 데이터 활용 경진대회  
- 프로그램명: 딥러닝 기반 혐오 발언 자동 모자이크 시스템
- 작성자: 최현우  
- 작성일: 2025.10.28
- 학번: 20214056


## 프로젝트 개요  
본 프로그램은 한국어 **혐오 발언을 자동 탐지하여 모자이크 처리하는 시스템**입니다.  
KoELECTRA-base-v3-discriminator 모델을 파인튜닝하여 8가지 혐오 유형을 분류합니다.  


**분류 가능한 유형**
["출신차별", "외모차별", "정치성향차별", "혐오욕설", "연령차별", "성차별", "인종차별","종교차별", "차별X"]


## 디렉토리 구조
├─ README.md
├─ app.py               # Streamlit 웹 데모
├─ train.py             # 모델 학습 스크립트
├─ requirements.txt     # 필요 라이브러리
├─ dataset/             # 데이터셋
│ ├─ train/kmhas_train.txt
│ └─ val/kmhas_valid.txt
│ └─ test/kmhas_test.txt
└─ models/              # 학습 후 저장되는 모델 파일


# 프로그램 실행 순서
## 1. 필요한 라이브러리 설치
프로젝트 폴더로 이동 후 아래 명령을 실행합니다.
```bash
pip install -r requirements.txt
```

## 2. Streamlit 웹 데모 실행
```bash
streamlit run app.py
```


## 모델 학습 방법
```bash
python train.py
```