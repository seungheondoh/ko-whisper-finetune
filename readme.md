# Whisper finetuner for Korean

한국어 ASR 모델 학습을 위한 레파지토리입니다. 간단한 Seq2Seq Model (whipser) finetunning 모델을 지원하기 위해 만들어 졌습니다.

### Install
1. Install python and PyTorch: 
    - python==3.10 
    - torch==2.1.2 (Please install it according to your [CUDA version](https://pytorch.org/get-started/previous-versions/).) 

2. Other requirements: - pip install -e .

### Finetunning with Whisper (Seq2Seq)

```
python train.py
```

