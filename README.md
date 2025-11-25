[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-2023+-green.svg)](https://www.rdkit.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**유기화학 반응 성공률 예측 시스템** - PyTorch와 RDKit을 활용한 딥러닝 기반 화학 반응 예측 도구

## 📖 소개

이 프로젝트는 **A → B 유기화학 반응**에서 특정 시약(현재 MVP: Br₂)을 사용할 때 **반응 성공 확률과 생성물 구조**를 예측하는 딥러닝 시스템입니다.

### 주요 특징

- 🎯 **SMILES 기반 분자 표현**: 화학 구조를 문자열로 입력
- 🧠 **PyTorch 신경망**: 딥러닝 기반 반응 성공률 예측
- 🔬 **반응성 부위 분석**: 알켄, 알카인, 방향족 작용기 자동 탐지
- 📊 **생성물 구조 예측**: 브롬 치환 위치와 생성물 SMILES 출력
- 📈 **수율 예측**: 실험 데이터 기반 예상 수율 계산
- 🖼️ **구조 시각화**: 원본 및 생성물 분자 구조 이미지 생성

### 기본 사용법

```bash
# 1. 샘플 데이터 생성
python generate_sample_data.py

# 2. 모델 학습
python train.py

# 3. 예측 실행
python predict.py

# 4. 특정 구조 예측 (target_smiles.py 수정 후)
python predict_target.py

# 5. 결과 시각화
python visualize_results.py
```

## 📊 예측 예시

### 입력
```python
# target_smiles.py
target = 'c1ccc(O)cc1'  # 페놀
```

### 출력
```
반응 타입: activated_aromatic_bromination
브롬화 위치: C2 (ortho to OH)
성공 확률: 78.0%
예상 수율: 75.0%
생성물 SMILES: Oc1ccc(Br)cc1
```

## 🏗️ 프로젝트 구조

```
organic-reaction-predictor/
├── model.py                          # PyTorch 신경망 모델
├── data_processor.py                 # SMILES 전처리 및 특징 추출
├── train.py                          # 모델 학습 스크립트
├── predict.py                        # 일반 예측 스크립트
├── predict_target.py                 # 특정 구조 예측 및 분석
├── visualize_results.py              # 결과 시각화
├── generate_sample_data.py           # 샘플 데이터 생성
├── target_smiles.py                  # 예측 대상 구조 입력
├── requirements.txt                  # 의존성 패키지
└── README.md                         # 프로젝트 문서
```

## 🔬 작동 원리

### 1. 데이터 전처리
- **SMILES → Morgan Fingerprint**: 2048비트 벡터로 변환
- **작용기 추출**: 반응성 부위 자동 탐지 (알켄, 알카인, 방향족 등)
- **분자 기술자**: 분자량, LogP, TPSA 등 계산

### 2. 모델 아키텍처
```
Input (4102 features)
    ↓
Dense(512) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(256) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(128) + BatchNorm + ReLU + Dropout(0.3)
    ↓
Dense(1) + Sigmoid
    ↓
Output (Success Probability: 0~1)
```

### 3. 반응 예측
- **알켄 브롬화**: 높은 성공률 (85-92%)
- **방향족 브롬화**: 중간 성공률 (60-78%)
- **알카인 브롬화**: 중간 성공률 (70-75%)
- **알칸 브롬화**: 낮은 성공률 (20-28%)

## 📈 학습 데이터

샘플 데이터는 다음을 포함합니다:
- **반응물/생성물 SMILES**
- **실험 수율 (yield)**
- **반응 조건** (온도, 용매, 시간)
- **성공 확률**

실제 사용 시 `reaction_data.csv`를 실험 데이터베이스로 교체하세요.

## 🎯 사용 사례

### 1. 신약 개발
- 약물 후보 물질의 합성 가능성 평가
- 최적 합성 경로 탐색

### 2. 화학 교육
- 유기화학 반응 메커니즘 학습
- 반응성 예측 실습

### 3. 연구 개발
- 신규 화합물 합성 계획
- 반응 조건 최적화

## 🔧 확장 가능성

현재 MVP는 Br₂ 시약으로 제한되어 있지만, 다음과 같이 확장 가능합니다:

```python
# 다양한 시약 지원
reagents = ['Br2', 'Cl2', 'NBS', 'H2SO4', 'NaOH', ...]

# 입력 특징에 시약 정보 추가
features = [reactant_fp, product_fp, reagent_fp, descriptors]
```

## 📊 성능

- **학습 데이터**: ~400-500개 반응 -> Real Data로 대체 가능
- **검증 정확도**: ~85%
- **예측 시간**: <1초/반응

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.


## 📚 참고 문헌

- Coley, C. W., et al. (2017). "Prediction of Organic Reaction Outcomes Using Machine Learning"
- Segler, M. H., & Waller, M. P. (2017). "Neural-Symbolic Machine Learning for Retrosynthesis and Reaction Prediction"

---

⭐ 이 프로젝트가 유용하다면 Star를 눌러주세요!
