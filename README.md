# KoELECTRA Hate Speech Detection

한국어 댓글/문장에서 혐오 표현을 탐지하고, 웹 데모에서 해당 문장을 모자이크 형태로 보여주는 프로젝트입니다.  
`monologg/koelectra-base-v3-discriminator` 모델을 기반으로 멀티라벨 분류를 수행합니다.

## Overview

- 한국어 혐오 표현 8개 유형을 멀티라벨로 분류합니다.
- Streamlit 기반 웹 데모를 제공합니다.
- CLI 예측 스크립트와 학습 스크립트를 함께 포함합니다.

## Labels

이 프로젝트는 아래 8개 혐오 유형을 예측합니다.

- 출신차별
- 외모차별
- 정치성향차별
- 혐오욕설
- 연령차별
- 성차별
- 인종차별
- 종교차별

`차별X`는 별도 출력 클래스로 학습하지 않습니다.
즉, 위 8개 라벨의 확률이 모두 임계값 미만일 때 비혐오 문장으로 간주합니다.

## Demo

웹 데모에서는 사용자가 입력한 문장을 모델이 예측한 뒤:

- 혐오 표현이 감지되면 모자이크 처리된 상태로 표시
- 클릭 시 원문 확인 가능
- 감지된 라벨과 확률을 함께 출력

## Tech Stack

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Streamlit

## Dataset

학습 및 검증에는 K-MHaS(Korean Multilabel Hate Speech) 데이터셋을 기반으로 사용했습니다.

- Dataset repository: https://github.com/adlnlp/K-MHaS

주의:

- 이 저장소에는 `dataset/` 폴더가 포함되어 있지 않습니다.
- 직접 데이터셋을 준비한 뒤 아래 구조에 맞게 배치해야 합니다.

## Project Structure

```text
koelectra-hate-speech-detection/
├── README.md
├── app.py
├── predict.py
├── train.py
├── requirements.txt
├── dataset/
│   ├── train/
│   │   └── kmhas_train.txt
│   ├── val/
│   │   └── kmhas_valid.txt
│   └── test/
│       └── kmhas_test.txt
└── models/
```

파일 설명:

- `app.py`: Streamlit 웹 데모
- `predict.py`: 터미널 기반 예측 스크립트
- `train.py`: 모델 학습 스크립트
- `models/`: 학습 완료 후 저장되는 모델 디렉터리
- `dataset/`: 학습/검증/테스트 데이터셋 디렉터리

## Installation

프로젝트 폴더에서 아래 명령을 실행합니다.

```bash
pip install -r requirements.txt
```

## Data Preparation

`train.py`는 아래 경로를 기준으로 데이터를 읽습니다.

```text
dataset/train/kmhas_train.txt
dataset/val/kmhas_valid.txt
```

각 파일은 TSV 형식이며, 코드 기준으로 다음 구조를 기대합니다.

```text
document<TAB>label
```

- `document`: 입력 문장
- `label`: 쉼표로 구분된 라벨 인덱스 목록

예시:

```text
document	label
예시 문장입니다.	1,3
```

## Train

학습을 시작하려면:

```bash
python train.py
```

기본 설정:

- backbone: `monologg/koelectra-base-v3-discriminator`
- max length: `128`
- batch size: `64`
- epochs: `5`
- learning rate: `3e-5`

학습이 완료되면 최적 모델이 `models/` 폴더에 저장됩니다.

## Run Web Demo

학습이 끝나 `models/` 폴더가 준비된 뒤 실행합니다.

```bash
streamlit run app.py
```

## Run CLI Prediction

터미널에서 간단히 예측하려면:

```bash
python predict.py
```

`q`를 입력하면 종료됩니다.

## Notes

- 이 저장소에는 학습된 모델 파일이 포함되어 있지 않습니다.
- 웹 데모와 CLI 예측은 모두 `models/` 폴더가 준비되어 있어야 실행됩니다.
- 현재 코드 기준으로 테스트셋 평가 스크립트는 별도로 포함되어 있지 않습니다.