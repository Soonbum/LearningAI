# AI 학습 노트

## 목차

[1단계: AI의 큰 그림과 작동 원리 파악]

1. AI, 머신러닝, 딥러닝, 강화학습의 개념과 차이점

2. AI 모델의 종류와 기본 작동 원리

[2단계: 프레임워크 실습과 블랙박스 해소]

3. PyTorch 기초 입문

4. PyTorch를 이용해 간단한 딥러닝 모델 구축해보기

[3단계: 모델의 확장과 실무 적용]

5. 기존 모델에 나만의 지식 확장시키기 1: 파인튜닝 (Fine-Tuning) (LoRA 등 가벼운 파인튜닝 기법으로 텍스트/이미지 모델의 가중치 업데이트해 보기)

6. 기존 모델에 나만의 지식 확장시키기 2: RAG (검색 증강 생성) (파인튜닝 없이 외부 데이터를 AI에게 참조시키는 방법 학습)

[4단계: 최신 AI 엔지니어링 및 애플리케이션 개발]

7. 에이전트(Agent)란 무엇인가? (LLM이 스스로 계획하고 도구를 사용하는 원리)

8. MCP(Model Context Protocol)란 무엇인가? (다양한 데이터 소스와 도구를 모델에 표준화된 방식으로 연결하는 구조 이해)

9. MCP, 에이전트, RAG를 융합하여 나만의 AI 비서 만들어보기 (최종 프로젝트)


## AI, 머신러닝, 딥러닝, 강화학습의 개념과 차이점

* 인공지능 (Artificial Intelligence): 인간의 지능(학습, 추론, 지극 등)을 모방하는 모든 컴퓨터 시스템을 통칭하는 가장 넓은 개념
  - 전통적인 규칙 기반 시스템도 넓은 의미에서는 AI에 해당함
 
* 머신 러닝 (Machine Learning): 데이터를 통해 컴퓨터가 스스로 학습하고 패턴을 찾아내는 알고리즘
  - 전통적인 프로그래밍: 규칙을 작성하고 데이터를 입력하면 결과를 출력함
  - 머신 러닝: 데이터와 결과(정답)을 입력 받으면 "규칙"을 작성하고 출력함
  - 개발자가 모든 예외 상황을 찾지 못해도 수많은 데이터와 결과를 입력함으로써 수학적 최저고하 알고리즘을 통해 최적의 파라미터를 찾아냄

* 딥 러닝 (Deep Learning): 머신 러닝의 한 분야로, 인간의 뇌 신경망을 모방한 인공 신경망(Artificial Neural Network)를 여러 겹(Deep)으로 쌓아 학습하는 기술
  - 쉽게 말해서 "거대한 행렬 곱셈과 미분 연산의 반복"
  - 데이터(Input)와 정답(Output)을 주고, 그 사이의 로직을 구성하는 수많은 변수(가중치)들을 수학적으로 찾아내는 과정
    * 인공 뉴런(Perceptron)이라는 기본 단위가 있는데 이것은 각 입력에 가중치를 곱하고 편향 값을 더함
    * 각 뉴런의 선형 결합 결과를 비선형 함수에 통과시키는 것을 활성화 함수라고 함
    * 비선형성을 부여한 뉴런들을 여러 층(Hidden Layers)으로 깊게(Deep) 쌓아 올린 것이 딥러닝 모델(DNN: Deep Neural Network)라고 함
  - 딥러닝 모델의 학습 과정 예시 (지도 학습의 원리)
    * 순전파 (Forward Propagation): 이미지를 입력하고 현재의 랜덤한 가중치 W로 결과를 계산함 (예: "고양이입니다.")
    * 손실 함수 (Loss Function): 정답("강아지")과 예측 사이의 오차를 수학적으로 계산함
    * 역전파 (Backpropagation): 미분의 연쇄 법칙을 사용하여, 출력층에서부터 거꾸로 입력층 방향으로 각 가중치들이 이 오차에 얼마나 기여했는지 기울기(Gradient)를 계산함
    * 경사 하강법 (Gradient Descent): 오차를 줄이는 방향으로 모든 가중치 W의 값을 아주 조금씩 업데이트
    * 오차를 0으로 수렴시키는 방향으로 위 사이클을 계속 반복하여 모델의 학습을 완료시킴
  - 일단 학습을 마친 모델은 수정이 불가능하나, 오탐지 데이터를 피드백으로 주어서 가중치를 미세 조정(Fine-Tuning)할 수 있음 (Active Learning이라고 함)
  - 인간의 눈과 귀 역할처럼 패턴 인식 및 맵핑을 통한 문제 해결에 주로 사용됨 (이미지 속 객체 판별, 음성-텍스트 변환(STT), 영어 문장을 한국어로 번역, 악성 코드 판별 등)
  - 현대의 거의 모든 대형 언어 모델(LLM)은 DNN의 발전된 형태인 "트랜스포머" 아키텍처를 사용함
  - 트랜스포머 아키텍처는 단순한 DNN과 달리 문장 속 단어들의 연관성을 수치로 계산하는 어텐션(Attention) 메커니즘을 도입하였음
  - ChatGPT, Claude 같은 언어 모델이나 Midjourney 같은 이미지 생성 모델 모두, 형태와 연산 방식이 데이터(텍스트, 이미지)에 맞게 특화되었을 뿐 본질적으로는 거대한 수학적 연산을 수행하는 '초거대 DNN'인 셈이다.

* 강화 학습 (Reinforcement Learning): 에이전트(Agent)가 주어진 환경(Environment)과 상호작용하며, 보상(Reward)을 최대화하는 방향으로 행동 방침(Policy)을 학습하는 방법
  - 지도 학습과 달리 정답이 없고, 피드백(가점/감점)을 기반으로 최적의 선택 드리를 탐색해가는 과정
  - 예: 알파고, 로봇 보행 학습, 자율 주행 자동차의 실시간 조향 및 속도 조절 등

* AI 모델의 구축 과정
  - ChatGPT나 Claude 같은 최신 AI 모델들은 DNN(딥러닝)의 뼈대 위에 강화학습(RL)의 영혼을 불어넣어 완성됨
  - 이 두 가지를 결합한 핵심 기술을 RLHF(Reinforcement Learning from Human Feedback, 인간 피드백 기반 강화학습)라고 부름
  - 구축 단계
    * 1단계: 사전 학습 (Pre-training) - "지식은 많지만 눈치 없는 천재"
      - 사용 기술: 딥러닝 (비지도학습/자기지도학습)
      - 과정: 인터넷에 있는 수조 개의 텍스트를 DNN(트랜스포머)에 쏟아붓고, 오직 '다음 단어 예측하기'만 무한히 반복시킵니다.
      - 결과: 문법과 세상의 지식은 엄청나게 많이 알게 되지만, 단순히 통계적으로 가장 그럴싸한 말을 이어 붙이는 '거대한 자동 완성기'에 불과합니다. 질문을 하면 대답을 하는 대신, 비슷한 질문을 이어서 만들어내기도 합니다.
    * 2단계: 미세 조정 (Supervised Fine-Tuning) - "대화의 형식을 배우다"
      - 사용 기술: 딥러닝 (지도학습)
      - 과정: 고품질의 [질문-답변] 데이터셋을 수만 개 준비하여 모델에게 "질문이 들어오면 이렇게 대답하는 거야"라고 패턴을 훈련시킵니다.
      - 결과: 이제 묻는 말에 대답할 줄 아는 '챗봇'의 형태를 갖춥니다. 하지만 여전히 심각한 문제가 있습니다. 인종차별적인 발언, 위험한 폭탄 제조법, 혹은 완전히 거짓말(할루시네이션)을 사실처럼 뻔뻔하게 말할 수 있습니다. 딥러닝 입장에서는 그저 오차(Loss)가 적은, 확률이 높은 단어의 조합일 뿐이기 때문입니다.
    * 3단계: RLHF (인간 피드백 기반 강화학습) - "사회화와 예절 교육"
      - 사용 기술: 강화학습 (RL)
      - 과정: 여기서 드디어 강화학습이 등판합니다.
        * AI에게 똑같은 질문에 대해 여러 개의 답변을 생성하게 합니다.
        * 사람이 직접 그 답변들을 읽고 "어떤 답변이 더 도움이 되고(Helpful), 진실되며(Honest), 무해한가(Harmless)?"를 기준으로 순위를 매겨 점수(보상, Reward)를 줍니다.
        * AI(에이전트)는 이 '보상'을 최대한 많이 받기 위해, 자신의 내부 가중치(행동 방침)를 스스로 업데이트합니다.
      - 결과: 비로소 위험한 질문에는 정중히 거절하고, 사용자에게 유용한 형태로 답변을 구조화할 줄 아는 진정한 AI 비서가 탄생합니다.


## AI 모델의 종류와 기본 작동 원리

* 지도학습 (Supervised Learning)
  - 입력 데이터(문제)와 그에 대한 레이블(정답)이 쌍으로 주어지는 학습 방식입니다. 모델은 입력과 정답 사이의 관계를 매핑하는 함수를 학습합니다.
    * 분류 (Classification): 이산형(Discrete) 값을 예측합니다.
      - 목적: 입력 데이터가 어느 클래스(카테고리)에 속하는지 판별합니다.
      - 예시: 입력된 패킷이 정상인지 악성(DDoS, 봇넷 등)인지 이진 판별하기, 이미지 속 동물이 개인지 고양이인지 다중 판별하기.
    * 회귀 (Regression): 연속형(Continuous) 수치를 예측합니다.
      - 목적: 두 변수 사이의 관계를 모델링하여 특정 값을 정확히 추정합니다.
      - 예시: 내일의 CPU 온도 예측, 주택 가격 예측.

* 비지도학습 (Unsupervised Learning)
  - 레이블(정답)이 없는 데이터만 주어지며, 데이터 내부에 숨겨진 구조나 패턴을 스스로 발견하는 학습 방식입니다.
    * 군집화 (Clustering)
      - 목적: 유사한 특성을 가진 데이터들을 묶어 여러 그룹으로 분류합니다.
      - 예시: 접속 로그 데이터에서 비슷한 행동 패턴을 보이는 사용자 그룹 묶기. (K-Means 알고리즘 등)
    * 차원 축소 (Dimensionality Reduction)
      - 목적: 데이터의 주요 특성을 유지하면서 변수의 개수(차원)를 줄여 컴퓨팅 비용을 아끼고 시각화를 용이하게 합니다.
      - CS 관점의 이해: 고해상도 이미지를 처리하기 전, 혹은 수백 개의 특징을 가진 센서 데이터 배열을 분석하기 전에 주성분 분석(PCA)을 통해 핵심 데이터만 남기는 손실 압축 과정과 유사합니다.


## PyTorch 기초 입문

* PyTorch: AI 연구와 실무에서 가장 많이 사용되는 프레임워크
  - 백엔드는 고속 행렬 연산을 위해 C++과 CUDA(NVIDIA의 GPU C 확장)로 최적화되어 있음

* 핵심 개념 3가지
  - 텐서 (Tensor): 딥러닝의 기초 데이터 구조이며 스칼라(0차원), 벡터(1차원), 행렬(2차원)을 넘어서는 다차원 배열을 통칭 (C/C++의 다차원 배열과 동일한 구조)
    * 비디오 메모리(VRAM)에 할당하여 GPU로 행렬 곱셈을 병렬로 수행
      ```python
      import torch

      # 1. 텐서 생성 (C++의 2차원 배열 초기화와 유사)
      x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

      # 2. GPU 메모리로 데이터 이동 (이 한 줄이 PyTorch의 핵심 중 하나입니다)
      if torch.cuda.is_available():
          x = x.to('cuda') 

      # 3. 텐서 연산 (내부적으로 C++ 백엔드를 호출하여 초고속으로 계산)
      y = x * 2  # 모든 원소에 2를 곱함
      z = torch.matmul(x, x)  # 행렬의 내적 연산
      ```
    - 자동 미분 (Autograd): 오차(Loss)를 줄이기 위해 역전파(Backpropagation)를 수행하며 가중치를 업데이트하기 위해 미분(Gradient) 연산을 수행함
      * 과거에는 이 미분 공식을 수학적으로 일일이 계산해서 코딩해야 했습니다.
      * 하지만 PyTorch의 Autograd 엔진은 텐서에 수행되는 모든 연산을 추적하여 메모리에 '연산 그래프(Computational Graph)'라는 방향성 비순환 그래프(DAG)를 동적으로 그립니다. 그리고 명령어 하나만 호출하면 연쇄 법칙(Chain Rule)을 적용해 역방향으로 모든 변수의 편미분 값을 자동으로 계산해 줍니다.
      * `backward()` 함수 한 번으로 수백만 개의 가중치가 가진 기울기가 순식간에 계산되며, 이는 모델 학습(가중치 업데이트)의 핵심 재료가 됩니다.
        ```python
        # requires_grad=True : "PyTorch야, 이 변수가 연산되는 모든 과정을 메모리에 기록해줘!"
        w = torch.tensor(2.0, requires_grad=True)

        # 순전파 (Forward) 수식 정의: y = 3 * w^2
        y = 3 * (w ** 2)

        # 역전파 (Backward) 실행: y를 w로 미분 (dy/dw)
        y.backward()

        # 결과 확인: y = 3w^2 이므로, 미분하면 6w. w=2이므로 기울기는 12.0
        print(w.grad) # 출력: tensor(12.)
        ```
    - 모델 구성 방법 (torch.nn.Module 상속)
      * PyTorch에서 신경망 모델을 만들 때는 객체지향의 상속(Inheritance)을 적극 활용합니다. 모든 신경망 모델은 torch.nn.Module이라는 기본 클래스를 상속받아 만들어집니다.
        ```python
        import torch.nn as nn
        import torch.nn.functional as F

        # nn.Module 상속
        class MySimpleNetwork(nn.Module):
            def __init__(self):
                super(MySimpleNetwork, self).__init__()
                # 입력층(노드 2개) -> 은닉층(노드 3개)로 가는 가중치 행렬 W1 생성
                self.layer1 = nn.Linear(in_features=2, out_features=3)
                # 은닉층(노드 3개) -> 출력층(노드 1개)로 가는 가중치 행렬 W2 생성
                self.layer2 = nn.Linear(in_features=3, out_features=1)

            # 데이터의 흐름 (OpenCV의 처리 파이프라인과 유사)
            def forward(self, x):
                # 1. 첫 번째 층 통과 (행렬 곱 Wx + b)
                x = self.layer1(x)
                # 2. 비선형 활성화 함수(ReLU) 적용 (음수 컷오프)
                x = F.relu(x)
                # 3. 출력층 통과
                x = self.layer2(x)
                return x

        # 모델 객체 생성 (메모리에 네트워크 구조 할당)
        model = MySimpleNetwork()
        ```
      * 모델을 구성할 때는 딱 두 가지 메서드만 오버라이딩(Overriding)하면 됩니다.
        - `__init__(self)`: 생성자입니다. 모델에서 사용할 부품(가중치를 가진 층, 예: 선형 변환 층)들을 메모리에 할당하고 초기화합니다.
        - `forward(self, x)`: 순전파 함수입니다. 입력 데이터 x가 들어왔을 때, __init__에서 만든 부품들을 조립하여 어떤 순서로 연산할지 데이터의 파이프라인(흐름)을 정의합니다.
      * 개발자는 이 `forward()` 함수만 잘 짜두면 됩니다. 앞서 설명한 `Autograd` 마법 덕분에, 역전파를 위한 `backward()` 함수는 PyTorch가 알아서 생성해 주기 때문입니다.


## PyTorch를 이용해 간단한 딥러닝 모델 구축해보기

* 동물 사진 분류 파이프라인
  - 1단계: 폴더 구조만으로 정답(Label) 자동 지정하기
    ```
    # 내가 준비한 하드디스크의 폴더 구조
    dataset/
    ├── dog/
    │   ├── dog1.jpg
    │   └── dog2.jpg
    ├── cat/
    │   ├── cat1.jpg
    │   └── cat2.jpg
    └── bird/
        ├── bird1.jpg
        └── bird2.jpg
    ```
    ```python
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    # 1. 이미지 전처리: 모든 동물의 사진 크기가 다르므로 224x224로 통일하고 텐서로 변환
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 색상 정규화
    ])

    # 2. 데이터 로드: 폴더 경로만 주면 알아서 dog=0, cat=1, bird=2로 라벨링 완료!
    animal_dataset = datasets.ImageFolder(root='./dataset', transform=transform)

    # 3. GPU로 올릴 준비 (한 번에 32장씩 묶어서 처리)
    animal_loader = DataLoader(animal_dataset, batch_size=32, shuffle=True)
    ```
  - 2단계: 동물 분류용 CNN 아키텍처 설계
    * 동물의 털 질감, 귀의 모양, 눈의 위치 같은 복잡한 특징을 잡아내기 위해 여러 겹의 합성곱(Conv) 층을 쌓습니다.
      ```python
      import torch.nn as nn
      import torch.nn.functional as F

      class AnimalClassifier(nn.Module):
          def __init__(self):
              super().__init__()
              # 특징 추출기 (Conv Layers)
              self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1) # 3채널(RGB) 입력
              self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
              self.pool = nn.MaxPool2d(2, 2)
        
              # 분류기 (Fully Connected Layers)
              # 224x224 이미지가 두 번의 풀링(반투막)을 거치면 56x56이 됨
              self.fc1 = nn.Linear(64 * 56 * 56, 512)
              self.fc2 = nn.Linear(512, 3) # 최종 출력: 개, 고양이, 새 (3가지)

          def forward(self, x):
              # 1. 특징 추출: 이미지에서 선 -> 형태 -> 귀/눈 등 구체적 패턴을 찾음
              x = self.pool(F.relu(self.conv1(x)))
              x = self.pool(F.relu(self.conv2(x)))
        
              # 2. 1차원 배열로 평탄화
              x = torch.flatten(x, 1)
        
              # 3. 최종 분류
              x = F.relu(self.fc1(x))
              x = self.fc2(x)
              return x

      model = AnimalClassifier()
      ```
  - 3단계: 결과 확인 (Softmax 함수)
    * 모델에 동물 사진을 넣으면 [12.5, -3.2, 1.1] 같은 해석하기 힘든 숫자들이 나옵니다.
    * 이를 우리가 이해할 수 있는 확률(%)로 바꿔주는 것이 바로 소프트맥스(Softmax)라는 마법의 함수입니다.
      ```python
      # 가상의 고양이 사진 1장을 모델에 입력했다고 가정
      image = torch.randn(1, 3, 224, 224) 
      raw_output = model(image) # 예: tensor([[-2.1, 5.3, 0.4]]) 

      # Softmax 함수 적용: 모든 값의 합이 1(100%)이 되도록 변환
      probabilities = F.softmax(raw_output, dim=1)

      print(probabilities) 
      # 출력 예시: tensor([[0.01, 0.95, 0.04]]) -> [개 1%, 고양이 95%, 새 4%]
      ```

<img width="761" height="613" alt="image" src="https://github.com/user-attachments/assets/fd6fdd52-bd0a-479f-8cd2-218beb5176d9" />

* 이 코드를 직접 짜서 학습시키면 훌륭한 분류기가 탄생하지만, 수만 장의 동물 사진을 수집하고 GPU를 며칠씩 돌려야 하는 한계가 있습니다.
  - 그래서 실무에서는 바닥부터 학습시키지 않고, 구글이나 메타가 이미 수백만 장의 사진으로 학습시켜 둔 똑똑한 모델(ResNet, VGG 등)을 가져와서 '내 데이터'만 살짝 추가로 얹어서 학습시킵니다.


## 기존 모델에 나만의 지식 확장시키기 1: 파인튜닝 (Fine-Tuning)

* 일례로 애니메이션 일러스트 생성 모델(예: Stable Diffusion)을 내 입맛대로 개조할 수 있음
  - LoRA (Low-Rank Adaptation) 기술을 이용하여 거대한 모델을 일반적인 데스크톱 환경에서 학습시킬 수 있게 되었습니다.
  - LoRA가 필요한 이유: 기존의 전통적인 파인튜닝(Full Fine-Tuning)은 새로운 개념을 가르치기 위해 이 수십억 개의 파라미터에 대해 모두 역전파(Backpropagation)를 수행하고 미분값을 업데이트해야 했습니다. 이 과정은 막대한 GPU VRAM(보통 24GB 이상)을 요구합니다.
  - 작동 원리: 거대한 원본 모델을 'Read-Only Memory (읽기 전용)'로 동결(Freeze)시켜 버리는 것입니다. 그리고 그 옆에 아주 작은 크기의 '패치(Patch) 메모리'만 새로 할당하여, 런타임에 원본의 결과값에 패치값을 더해주는(Bypass) 방식으로 결과를 가로채어 수정합니다.

* 실전 애니메이션 LoRA 제작 파이프라인
  - Step 1: 데이터셋 아카이빙 (수집 및 정제)
    * gallery-dl이나 크롤링 스크립트를 이용해 목표로 하는 화풍이나 캐릭터 디자인과 유사한 애니메이션 일러스트를 20~50장 정도 로컬에 수집합니다.
    * 이미지 크기를 모델에 맞게 (예: 512x512 또는 1024x1024) 크롭합니다.
  - Step 2: 캡셔닝 (정답지 만들기)
    * 각 이미지에 대해 AI가 이해할 수 있는 텍스트 태그(Booru tags)를 달아줍니다.
    * 예: 1girl, solo, green hair, bird motif, yellow eyes, white background
  - Step 3: PyTorch에서의 LoRA 학습 (개념적 코드)
    * 앞서 3단계 커리큘럼에서 보셨던 모델 구조에 LoRA 행렬을 주입하는 과정입니다.
      ```python
      import torch
      import torch.nn as nn

      # 1. 거대한 원본 모델 (예: Stable Diffusion의 특정 레이어)
      class OriginalLayer(nn.Module):
          def __init__(self):
              super().__init__()
              self.weight = nn.Linear(4096, 4096)
        
          def forward(self, x):
              return self.weight(x)

      # 2. LoRA를 덧붙인 레이어
      class LoRALayer(nn.Module):
          def __init__(self, original_layer, rank=4):
              super().__init__()
              self.original = original_layer
        
              # 원본은 더 이상 미분(학습)하지 않도록 동결(Freeze)
              for param in self.original.parameters():
                  param.requires_grad = False
            
              # 작은 패치 행렬 A와 B 생성 (이 녀석들만 학습됩니다)
              self.lora_A = nn.Linear(4096, rank, bias=False)
              self.lora_B = nn.Linear(rank, 4096, bias=False)
        
          def forward(self, x):
              # 원본 결과값 + (데이터가 A를 거치고 B를 거친 패치 결과값)
              return self.original(x) + self.lora_B(self.lora_A(x))
      ```
  - Step 4: 인퍼런스 (합체 및 사용)
    * 학습이 끝나면 수십 MB 크기의 아주 가벼운 *.safetensors 파일(LoRA 가중치) 하나만 쏙 떨어집니다.
    * 나중에 그림을 그릴 때, 수 GB짜리 원본 애니메이션 모델을 램에 올리고 그 위에 이 수십 MB짜리 패치 파일을 '덮어쓰기(Merge)'해서 이미지를 생성하게 됩니다.
    * 이렇게 핵심적인 수학 트릭 하나로 일반 유저들도 집에서 모델을 튜닝할 수 있게 된 것입니다.

* 실제로 애니메이션 LoRA를 만들 때, 이 PyTorch 코드를 바닥부터 다 짜지는 않고 보통 `Kohya_ss` 같은 최적화된 오픈소스 GUI 툴을 많이 사용합니다.


## 기존 모델에 나만의 지식 확장시키기 2: RAG (검색 증강 생성)

* RAG(Retrieval-Augmented Generation): 현재 기업용 AI 비서나 챗봇을 만들 때 사실상 업계 표준(De facto standard)으로 쓰이는 아키텍처입니다.
  - 파인튜닝(LoRA)이 모델의 '뇌(가중치)'를 직접 수술해서 지식을 주입하는 방식이라면, RAG는 똑똑한 AI에게 '최신 매뉴얼(오픈북)'을 쥐여주고 시험을 보게 하는 방식입니다.

* RAG의 4단계 파이프라인
  - Step 1. 문서 청킹 (Chunking) 및 임베딩 (Embedding)
    * 수백 페이지짜리 방어 시스템 매뉴얼이나 보안 로그를 문단 단위(Chunk)로 잘게 쪼갭니다.
    * 이 텍스트 조각들을 '임베딩 모델'이라는 가벼운 신경망에 통과시켜 수백 차원의 실수 배열(벡터)로 변환합니다. (예: `[0.12, -0.55, 0.89, ...]`)
    * 의미가 비슷한 문장일수록 벡터 공간에서 서로 가까운 좌표를 갖게 됩니다.
  - Step 2. 벡터 데이터베이스 (Vector DB) 저장
    * 변환된 수많은 벡터 배열을 메모리나 디스크(ChromaDB, Pinecone 등)에 인덱싱하여 저장합니다.
  - Step 3. 검색 (Retrieval)
    * 사용자가 "야간 모드에서 열화상 센서가 오작동하면 어떻게 해?"라고 질문합니다.
    * 시스템은 이 질문 역시 벡터로 변환한 뒤, Vector DB에서 이와 가장 거리가 가까운(유사도가 높은) 매뉴얼 조각 상위 3개를 초고속 행렬 내적 연산으로 찾아냅니다.
  - Step 4. 프롬프트 조합 및 생성 (Augmented Generation)
    * 찾아낸 매뉴얼 조각(Context)과 사용자의 원래 질문을 합쳐서 하나의 거대한 문자열(Prompt)을 조립합니다.
    * "너는 보안 플랫폼 AI 비서야. 아래 제공된 [매뉴얼]만 참고해서 사용자의 [질문]에 답변해. [매뉴얼]: 1. 야간 열화상 센서 오류 시... [질문]: 야간 모드에서 열화상 센서가 오작동하면 어떻게 해?"
    * 이 조립된 텍스트를 LLM(ChatGPT 등)의 API에 던져서 최종 답변을 받아냅니다.

* 왜 RAG가 엔지니어링의 영역인가?
  - 위 파이프라인을 보시면 아시겠지만, GPU를 이용한 무거운 학습 루프(`loss.backward()`)가 아예 존재하지 않습니다.
  - 성능 좋은 외부 LLM API를 호출하기만 하면 되므로 로컬 VRAM이 필요 없고, 데이터가 추가되거나 수정되면 그저 Vector DB의 해당 행(Row)만 갱신해 주면 끝납니다. 어떤 크기로 문서를 자를 것인지, 검색 속도를 어떻게 최적화할 것인지, 프롬프트 문자열을 어떻게 디자인할 것인지 등 전형적인 시스템 설계 문제로 귀결됩니다.


## 에이전트(Agent)란 무엇인가? (LLM이 스스로 계획하고 도구를 사용하는 원리)


## MCP(Model Context Protocol)란 무엇인가? (다양한 데이터 소스와 도구를 모델에 표준화된 방식으로 연결하는 구조 이해)


## MCP, 에이전트, RAG를 융합하여 나만의 AI 비서 만들어보기 (최종 프로젝트)
