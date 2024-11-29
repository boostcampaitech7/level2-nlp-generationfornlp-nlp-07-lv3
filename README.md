# CSAT-Solver Project (수능형 문제 풀이 모델)

<div align="center">
  <a href="#korean">🇰🇷 한국어</a> | <a href="#english">🇺🇸 English</a>
</div>

<h2 id="korean">한국어</h2>

`CSAT-Solver`는 고급 언어 모델을 사용하여 한국의 대학수학능력시험 문제를 해결하기 위해 설계된 프로젝트입니다. 이 프로젝트는 대규모 언어 모델에 미세 조정 기법을 적용하여 CSAT의 전형적인 객관식 문제에 대한 성능을 향상시킵니다.

## 프로젝트 기간

11월 11일 (월) 10:00 ~ 11월 28일 (목) 19:00

## 팀원

<h3 align="center">NLP-7조 NOTY</h3>

<table align="center">
  <tr height="100px">
    <td align="center" width="150px">
      <a href="https://github.com/Uvamba"><img src="https://avatars.githubusercontent.com/u/116945517?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/doraemon500"><img src="https://avatars.githubusercontent.com/u/64678476?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/simigami"><img src="https://avatars.githubusercontent.com/u/46891822?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/DDUKDAE"><img src="https://avatars.githubusercontent.com/u/179460223?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/mrsuit0114"><img src="https://avatars.githubusercontent.com/u/95519378?v=4"/></a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/hskhyl"><img src="https://avatars.githubusercontent.com/u/155405525?v=4"/></a>
    </td>
  </tr>
  <tr height="10px">
    <td align="center" width="150px">
      <a href="https://github.com/simigami">강신욱</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/doraemon500">박규태</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/simigami">이정민</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/ksj1368">장요한</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/ksj1368">한동훈</a>
    </td>
    <td align="center" width="150px">
      <a href="https://github.com/ksj1368">홍성균</a>
    </td>
  </tr>
</table>
&nbsp;

## 설정 및 사용법

#### 1. `requirements.txt`를 통해 필요한 라이브러리 설치

```bash
pip install -r requirements.txt
```

#### 2. 훈련 및 테스트 데이터셋을 `data` 디렉토리에 배치

#### 3. `arguments.py`에서 학습 진행할 모델 이름, max sequence length, chat template 등 여러 인자들 설정

```python
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='beomi/gemma-ko-2b', # 사용할 모델
    )
    train_test_split: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "test_size"
        },
    )
    .
    .
    .

```

#### 3. `run.py` 스크립트를 실행하여 훈련 및 추론 프로세스를 시작

```bash
python run.py
```

## 주요 기능

- 사전 훈련된 언어 모델(예: Gemma-ko-2b)의 미세 조정
- 4비트 및 8비트 양자화 지원
- CSAT 스타일 문제에 대한 맞춤형 데이터 처리
- 훈련 및 평가 파이프라인
- 테스트 데이터셋에 대한 추론
- Weights & Biases를 통한 실험 추적 통합

## 프로젝트 구조

```plaintext
CSAT-Solver/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── models/                 # LoRA 학습된 adapter 저장 디렉토리
│
├── output/                 # test.csv로 inference 결과 저장 디렉토리
│
├── src/
│   ├── arguments.py        # 학습에 필요한 여러 인자
│   ├── utils.py            # 시드 고정 및 데이터 셋 chat message 형태로 변환
│   ├── streamlit_app.py    # EDA
│   └── main.py             # 모델 학습 및 추론
│
├── requirements.txt
├── README.md
└── run.py                  # 실행 파일
```

## streamlit_app.py 설명

Streamlit 기반의 EDA 제공

- 패턴 기반 ID 조회
- 전체 데이터셋 표시
- 길이 분포 시각화
- 토큰화된 길이 분포 시각화
- 채팅 템플릿 적용 후 토큰화 길이 분포
- 답변 분포 표시

---

<h2 id="english">English</h2>

`CSAT-Solver` is a project designed to solve questions from the Korean College Scholastic Ability Test (CSAT) using advanced language models. This project applies fine-tuning techniques to large language models to improve performance on typical multiple-choice questions in the CSAT.

## Project Duration

November 11 (Monday) 10:00 AM ~ November 28 (Thursday) 7:00 PM

## Setup and Usage

#### 1. Install required libraries using requirements.txt

```bash
pip install -r requirements.txt
```

#### 2. Place training and test datasets in the `data` directory

#### 3. Set various arguments in `arguments.py` such as model name, max sequence length, chat template, etc.

```py
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        default='beomi/gemma-ko-2b', # Model to use
    )
    train_test_split: Optional[float] = field(
        default=0.1,
        metadata={
            "help": "test_size"
        },
    )
    .
    .
    .
```

#### 4. Run the `run.py` script to start the training and inference process

```bash
python run.py
```

## Key Features

- Fine-tuning of pre-trained language models (e.g., Gemma-ko-2b)
- Support for 4-bit and 8-bit quantization
- Custom data processing for CSAT-style questions
- Training and evaluation pipeline
- Inference on test datasets
- Integration with Weights & Biases for experiment tracking

## Project Structure

```plaintext
CSAT-Solver/
│
├── data/
│ ├── train.csv
│ └── test.csv
│
├── models/                 # Directory for storing LoRA trained adapters
│
├── output/                 # Directory for storing inference results on test.csv
│
├── src/
│   ├── arguments.py        # Various arguments needed for training
│   ├── utils.py            # Seed fixing and dataset conversion to chat message format
│   ├── streamlit_app.py    # EDA
│   └── main.py             # Model training and inference
│
├── requirements.txt
├── README.md
└── run.py                  # Execution file
```

## streamlit_app.py explanation

Provides Streamlit-based Exploratory Data Analysis (EDA)

- Pattern-based ID lookup
- Display of the entire dataset
- Visualization of length distribution
- Visualization of tokenized length distribution
- Distribution of tokenized lengths after applying chat template
- Display of answer distribution
