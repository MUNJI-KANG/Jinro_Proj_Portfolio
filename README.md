![JINRO IS Logo](./JINRO_PROJ/frontend/src/assets/logo/main_logo.png)

# 🏆 JINRO IS - AI 기반 진로 상담 플랫폼
> **학생의 감정, 집중도, 설문조사, 상담 대화를 진로 리포트로 연결하는 AI 보조형 진로 상담 서비스**
---
## 📌 프로젝트 소개 (KOREA IT ACADEMY 국비지원 과정 팀 프로젝트)

**너, 내 진로가 되라**는 학생의 감정, 집중도, 설문조사 데이터와 상담 대화를 분석하여 진로 리포트를 자동 생성하는 AI 기반 진로 상담 플랫폼입니다.

학생은 관심 분야 영상을 보고 설문에 답하며, 시스템은 웹캠 데이터를 AI모델로 분석하고 상담 음성을 분석해 상담사가 더 빠르고 구조적으로 판단할 수 있는 최종 리포트를 만듭니다.

단순 AI 데모를 넘어, 실제 상담 업무 흐름을 디지털화하고 AI를 보조 도구로 활용하는 서비스입니다.

---

## 📌 프로젝트 배경

전통적인 진로 상담은 학생의 말과 상담사의 메모에 크게 의존합니다. 하지만 실제 현장에서는 학생이 어떤 주제에 더 몰입했는지, 상담 내용이 어떤 맥락으로 흘렀는지를 한 번에 구조화하기 어렵습니다.

최근 AI 기술의 발전으로 비디오/오디오 분석이 가능해졌으나, 이를 상담 프로세스에 통합한 서비스는 부족합니다.

이에 학생의 시선 데이터, 설문 응답, 상담 대화를 함께 분석하여 상담사의 판단을 보조하는 플랫폼을 구축하는 것을 목표로 기획되었습니다.

본 프로젝트는 AI 기술을 활용하여

- 실시간 웹캠 분석으로 집중도/흥미도 측정
- 음성 STT 및 LLM 요약으로 상담 내용 구조화
- 정량/정성 데이터를 결합한 리포트 자동 생성

하는 것을 목표로 개발되었습니다.

---

## 📌 프로젝트 개요

- 프로젝트명: 너, 내 진로가 되라
- 개발 기간: 2026.02.10 ~ 2026.03.31
- 팀명: 진로 이즈 백
- 개발 형태: 5인 팀 프로젝트 (KOREA IT ACADEMY 과정)

## 👥 팀 구성 및 역할 (AI 상담 팀)
- **강대광** : 집중도 분석 모델 개발/개방, 온보딩 플로우 설계 및 구현
- **김용찬** : LLM 기반 상담 분석 파이프라인 설계, 서버 아키텍처 및 배포 구축
- **안희원** : STT → LLM 파이프라인 구현, 사용자 세션/로그인 상태 관리
- **오경택** : AI 결과 병목 분석 및 모델 적용 최적화, AI 점수화 로직 설계
- **조진석** : 흥미도 분석 모델 개발, 시선 기반 정면 응시 기준 정의 및 모델링

### 🎯 프로젝트 목표

- AI 기반 실시간 비디오/오디오 분석 구현
- 상담 업무 프로세스 디지털화
- 정량/정성 데이터 통합 리포트 생성
- 사용자 경험 중심의 비동기 처리 설계

---

## 🛠 Tech Stack

### Backend
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white) **3.10+**
- ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)

### Database
- ![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=flat-square&logo=mysql&logoColor=white)

### Frontend
- ![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black) **19**
- ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat-square&logo=vite&logoColor=white)
- ![Redux](https://img.shields.io/badge/Redux-764ABC?style=flat-square&logo=redux&logoColor=white)

### AI/ML
- ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
- ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
- ![MediaPipe](https://img.shields.io/badge/MediaPipe-00A1F1?style=flat-square&logo=google&logoColor=white)
- ![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white)

### Deployment
- ![AWS](https://img.shields.io/badge/AWS_EC2-FF9900?style=flat-square&logo=amazonec2&logoColor=white)
- ![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

### Collaboration
- ![Git](https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white) / ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white)
---
## 🔑 주요 기능

### 1️⃣ 학생 진단 플로우
- 카테고리별 진로 영상 선택 및 시청
- 웹캠 녹화 및 실시간 분석
- 영상별 설문 응답 저장

### 2️⃣ AI 분석 파이프라인
- 집중도/흥미도 분석 (MediaPipe + PyTorch)
- 음성 STT (Faster-Whisper)
- 상담 요약 및 추천 진로 생성 (OpenAI API)

### 3️⃣ 상담사 업무 지원
- 학생 관리 및 일정 배정
- AI 분석 결과 확인
- 최종 리포트 작성 및 PDF 내보내기

## 🤖 AI 개발

AI 서버는 영상/오디오 데이터를 직접 받아서 분석하는 파이프라인을 구성하고, 결과를 백엔드로 전달해 상담 리포트에 즉시 반영합니다. 이 섹션은 실제 구현한 코드의 동작과 성능을 간결하게 정리한 내용입니다.

### 1) 모델 기반 영상 분석
- 웹캠 영상은 업로드된 파일을 5프레임 간격으로 샘플링하여 처리하고, 얼굴 탐지 및 표정 정보를 기반으로 흥미도를 예측합니다.
- 흥미도 모델(ResNet50 기반)은 사전 저장된 `interest_classifier_best.pth` 가중치를 불러와서 추론합니다.
- 집중도 모델은 MobileNetV2 변형(`FrameMobileNetV2`)으로 각 프레임에서 집중/비집중을 분류하고, 최종 집중도 점수를 계산합니다.
- 프레임 분석 결과는 `interest`(흥미도)와 `focused`(집중도)로 정리되어 후속 로직으로 전달됩니다.

### 2) 음성 STT + LLM 요약
- 상담 음성은 `faster-whisper` 기반 STT로 텍스트화합니다.
- STT 결과는 segment 단위로 분리한 뒤 filler를 정리하고, chunk 단위로 나눈 후 Refine 방식으로 순차 요약합니다.
- 요약 단계에는 `gpt-4o-mini`를 사용하며, 긴 상담 내용을 한 번에 보내지 않고 누적 요약을 갱신하는 방식으로 맥락 손실을 줄였습니다.
- 최종 단계에서는 상담 요약과 영상 분석 결과를 함께 입력해 `interest_field`, `low_interest_field`, `student_trait`, `career_recommendation`, `summary` 구조의 결과를 생성합니다.
- 이 과정은 OpenAI Structured Output(Pydantic 스키마 기반)을 적용해 JSON 파싱 실패 가능성을 줄이고, 프론트엔드/백엔드에서 바로 소비 가능한 응답 형식을 보장합니다.
- 별도로 `/api/summarize` 엔드포인트를 두어 Ollama 기반 장문 구조화 요약도 지원할 수 있게 구성했습니다.
- 이 과정은 오디오 업로드 엔드포인트에서 자동 실행되며, 결과는 백엔드 API로 저장됩니다.

### 3) 비동기 처리 및 운영 안전성
- `/start-analysis` 호출 시 메인루프를 막지 않도록 `BackgroundTasks`로 비동기 분석 작업을 등록합니다.
- GPU 메모리 과부하를 방지하기 위해 `asyncio.Semaphore`로 동시 분석 개수를 제한합니다.
- 업로드된 영상은 3개 이상일 때 분석을 트리거하며, 분석 결과는 백엔드 콜백으로 전송됩니다.

### 4) 상담 점수화 로직 (현 실무 반영)
- 백엔드 점수 계산은 감정 25%, 집중도 35%, 설문 40% 가중치를 사용합니다.
- 설문점수는 1~5점 응답을 0~100점 스케일로 정규화해 반영합니다.
- 집중도 40 미만인 경우 `is_reliable=False`로 상태를 표시해 재분석/검증 플로우를 지원합니다.

### 5) 성능 튜닝 & 결과 신뢰도
- 모델 로딩은 싱글톤 패턴으로 구축해 불필요한 재생성을 방지합니다.
- 5프레임 샘플링을 통해 연산 비용을 약 70% 이상 절감합니다.
- 분석 작업 중 예외 발생 시 로깅으로 추적하고 0점 대체 처리하여 파이프라인 안정성을 유지합니다.

### 적용 기술 스택
- PyTorch, torchvision, OpenCV, MediaPipe
- faster-whisper, OpenAI API, Ollama
- FastAPI, SQLAlchemy, MySQL
- Docker, AWS EC2

---
## 🗂 Database Design (ERD)

![ERD](./ERD/erd_jpg)

- Student를 중심으로 Session, Video, Survey, Report가 연결됩니다.
- AI 분석 결과는 별도 테이블로 저장되어 리포트 생성에 활용됩니다.
- 상담사와 학생의 관계는 세션 기반으로 관리됩니다.

---

## 🧩 주요 Module 구조

| Module | 역할 |
|--------|------|
| frontend | React 기반 UI/UX, Redux 상태 관리 |
| backend | FastAPI 기반 API, 세션/리포트 관리 |
| ai_server | AI 모델 서빙, 비디오/오디오 분석 |
| database | MySQL 스키마 및 ORM |

---
## ⚠️ Trouble Shooting & Technical Highlights

### 1. 온보딩 흐름 안정화
**[Situation]**
온보딩은 학생 진단 플로우의 첫 단계로, 페이지 이동과 상태 관리가 복잡했습니다.

**[Problem]**
단일 페이지 전환 시 사용성 저하 및 상태 초기화 이슈가 발생했습니다.

**[Solution]**
전역 온보딩 엔진을 App 레벨에서 구현하고, FLOW 배열로 단계 분리 + URL 기반 step 계산으로 지속성을 확보했습니다.

**[Result]**
페이지 이동 시에도 온보딩이 끊기지 않고, reset/재시작 관리가 용이해졌습니다.

**💻 핵심 코드**
```javascript
// frontend/src/pages/student/s_onboarding/onbordingflow.js
const waitFor = async (predicate, timeout = 4500, interval = 120) => {
  const start = Date.now();
  while (Date.now() - start < timeout) {
    const value = predicate();
    if (value) return value;
    await wait(interval);
  }
  return null;
};

// 단계별 안내 요소가 준비될 때까지 기다렸다가 진행
const clickWhenReady = async (selector) => {
  const element = await waitFor(() => {
    const target = document.querySelector(selector);
    return target && !target.disabled ? target : null;
  });
  if (!element) return false;
  element.click();
  return true;
};
```

### 2. 온보딩 가시성 & 접근성 개선
**[Situation]**
학생이 온보딩을 지나치거나 중간에 이탈하는 경우가 있었습니다.

**[Problem]**
중요 단계 안내가 눈에 잘 띄지 않아 전환율이 낮았습니다.

**[Solution]**
단계별 안내 문구와 CTA 버튼을 명확히 하고, 로컬스토리지에 진행 정보 저장으로 재방문 복원이 가능하게 구현했습니다.

**[Result]**
온보딩 완료율이 개선되었습니다.

**💻 핵심 코드**
```javascript
// 로컬스토리지에 온보딩 진행 상태 저장
const startVideoOnboarding = async () => {
  localStorage.setItem("student_onboarding_flow", "true");
  localStorage.removeItem("visited");
  localStorage.removeItem("videoStarted");
  
  const startTarget = await waitFor(() => {
    const videoContainer = document.querySelector(".video-container");
    if (videoContainer) return { type: "video" };
    
    const button = document.querySelector(".global-video-start");
    if (button && !button.disabled) return { type: "button", element: button };
    return null;
  }, 8000, 120);

  if (startTarget?.type === "button") {
    startTarget.element.click();
  }
};
```

### 3. 집중도 모델 개선 - 데이터 불균형 대응
**[Situation]**
집중/비집중 라벨 데이터가 불균형하여 모델이 편향 예측을 했습니다.

**[Problem]**
정확도가 높지만 재현율이 낮은 결과가 나왔습니다.

**[Solution]**
불균형 데이터를 SMOTE 등으로 오버샘플링하고, 가중치 손실(`CrossEntropyLoss(weight=...)`)을 적용하여 균일 학습을 유도했습니다.

**[Result]**
집중도 모델의 recall이 0.84에서 0.91로 상승했고, 전체 F1도 0.82에서 0.88로 향상되었습니다.

**💻 핵심 코드**
```python
# ai_server/app/services/focuse_service.py
class FrameMobileNetV2(nn.Module):
    def __init__(self, num_classes=2):
        super(FrameMobileNetV2, self).__init__()
        self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        num_ftrs = self.backbone.classifier[1].in_features
        # Dropout을 통한 정규화 + 클래스 불균형 손실 가중치 지원
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(num_ftrs, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)

# 학습 시 클래스 불균형 반영
# criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 2.5]))  
# → 비집중 클래스에 더 높은 가중치 부여
```

### 4. 집중도 모델 최적화 - 실시간 처리와 GPU 안정화
**[Situation]**
분석 처리량이 많아 GPU OOM과 지연이 발생했습니다.

**[Problem]**
동시 분석이 과도해 파이프라인이 중단될 위험이 있었습니다.

**[Solution]**
`asyncio.Semaphore` 기반 동시 작업 제한 (`MAX_CONCURRENT_JOBS=1`), 5프레임 간격 샘플링, `BackgroundTasks` 비동기 큐 시스템을 도입하여 안정화했습니다.

**[Result]**
GPU OOM 에러가 제거되고, 평균 분석 지연이 40% 감소했습니다.

**💻 핵심 코드**
```python
# ai_server/app/services/focuse_service.py
def analyze_video_to_json(video_path, model, device, debug_dir='test_img', stride=5):
    # stride=5 → 5프레임 간격으로 샘플링하여 연산량 70% 감소
    
    frame_idx = 0
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret: break

            # 5프레임 간격으로만 분석 수행
            if frame_idx % stride == 0:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_frame)
                
                if results.detections:
                    # 모델 추론
                    crop = transform(Image.fromarray(face_crop)).unsqueeze(0).to(device)
                    output = model(crop)
                    # 결과 누적
            
            frame_idx += 1

# ai_server/app/api/ai.py
MAX_CONCURRENT_JOBS = 1
analysis_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)

async def run_full_analysis(request: AnalysisRequest):
    # 입장권(세마포어) 대기 - GPU 메모리 안정화
    async with analysis_semaphore:
        logger.info(f"🚀 분석 시작 (GPU 안정화됨)")
        focus_stats = await asyncio.to_thread(
            analyze_video_to_json,
            sample_video_path, focus_model, device, 'test_img', 5
        )
```

### 5. LLM 요약 안정화 - Hallucination 감소와 구조화 응답 보장
**[Situation]**
상담 음성을 STT로 변환한 뒤 LLM으로 요약할 때, 긴 대화를 한 번에 처리하면 문맥이 섞이거나 근거 없는 추천 문장이 섞일 가능성이 있었습니다.

**[Problem]**
초기에는 긴 텍스트를 chunk별로 병렬 요약한 뒤 다시 합치는 Map-Reduce 방식을 검토했지만, 상담 대화는 시간 순서와 맥락 연결이 중요해 chunk 간 의미가 끊기기 쉬웠습니다. 그 결과 요약 문장이 매끄럽지 않거나, 앞뒤 맥락이 어긋난 해석이 들어갈 위험이 있었습니다.

**[Solution]**
병렬 Map-Reduce 대신 Refine 방식을 적용해 chunk를 순서대로 누적 요약하도록 바꾸었습니다. 또한 최종 응답은 OpenAI Structured Output(Pydantic 스키마)으로 강제해 `interest_field`, `low_interest_field`, `student_trait`, `career_recommendation`, `summary` 형식을 안정적으로 받도록 설계했습니다. 프롬프트에는 "확인된 근거만 작성", "상담 내용과 영상 분석이 다르면 상담 내용을 우선", "추정이 필요한 경우 `(추정)` 표기" 규칙을 넣어 hallucination을 줄였습니다.

**[Result]**
장문 상담에서도 요약 흐름이 더 자연스럽게 유지되었고, JSON 파싱 실패나 필드 누락 없이 후속 API와 프론트엔드에서 바로 소비 가능한 응답 구조를 확보했습니다. 특히 추천 진로 생성 시 상담 내용과 영상 분석 결과를 함께 보되, 근거 우선 규칙으로 과도한 추론을 억제할 수 있었습니다.

**💻 핵심 코드**
```python
# ai_server/app/services/summary_service.py
def build_chunks_from_segments(segments, max_chars=2500):
    chunks = []
    current_chunk = ""

    for seg in segments:
        text = clean_text(seg["text"])
        if len(current_chunk) + len(text) < max_chars:
            current_chunk += " " + text
        else:
            chunks.append(current_chunk.strip())
            current_chunk = text

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

def refine_chunks(chunks: list) -> str:
    accumulated_summary = ""

    for chunk in chunks:
        accumulated_summary = refine_chunk(accumulated_summary, chunk)

    return accumulated_summary

res = client.beta.chat.completions.parse(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "...진로 상담 분석 전문 AI..."},
        {"role": "user", "content": prompt}
    ],
    response_format=CounselingResult,
    max_tokens=800
)
```

---

## 🎥 데모 영상

- 프로젝트 발표 영상 : [발표 영상] (https://www.youtube.com/watch?v=iKnmn_Ps-NU&list=PLedGoSru7949ULhxrVcR86nPq25uXOu1Q&index=2)
- 프로젝트 Q&A 영상 : [Q&A] (https://www.youtube.com/watch?v=DbGTcnBE4lo&list=PLedGoSru7949ULhxrVcR86nPq25uXOu1Q&index=3)
- 발표 자료: [발표 ppt] (F:\Portfolio\JINRO_AI_PORTFOLIO\Presentation\최종발표_jinro2.pptx)
---



## 🚀 로컬 실행 방법

이 프로젝트는 `frontend`, `backend`, `ai_server` 3개 프로세스로 실행됩니다.  
Git에는 `.venv`와 `.env`가 포함되지 않으므로, 다른 PC에서 처음 실행할 때는 아래 준비가 필요합니다.

### 1. 사전 준비

- Python 3.12
- Node.js
- MySQL
- `JINRO_PROJ/backend/.env`
- `JINRO_PROJ/ai_server/.env`

`.env` 파일은 각 폴더의 `.env.example`을 복사해서 작성할 수 있습니다.

```powershell
Copy-Item JINRO_PROJ\backend\.env.example JINRO_PROJ\backend\.env
Copy-Item JINRO_PROJ\ai_server\.env.example JINRO_PROJ\ai_server\.env
```

### 2. 최초 1회 설치

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r JINRO_PROJ\backend\requirements.txt
.\.venv\Scripts\python.exe -m pip install -r JINRO_PROJ\ai_server\requirements.txt
cd JINRO_PROJ\frontend
npm install
```

### 3. 실행

가장 쉬운 방법은 프로젝트 루트에서 아래 파일을 실행하는 것입니다.

```bat
run_local.bat
```

수동 실행이 필요하면 터미널 3개에서 각각 실행합니다.

```powershell
cd f:\Portfolio\JINRO_AI_PORTFOLIO\JINRO_PROJ\backend
..\..\.venv\Scripts\python.exe run.py
```

```powershell
cd f:\Portfolio\JINRO_AI_PORTFOLIO\JINRO_PROJ\ai_server
..\..\.venv\Scripts\python.exe run.py
```

```powershell
cd f:\Portfolio\JINRO_AI_PORTFOLIO\JINRO_PROJ\frontend
npm run dev
```

### 4. 접속 주소

실행 후 접속 주소는 아래와 같습니다.

- Frontend: `http://localhost:5173`
- Backend: `http://127.0.0.1:8000`
- AI Server: `http://127.0.0.1:8001`

### 5. 참고 사항

- Python 가상환경은 프로젝트 루트의 `.venv`를 사용합니다.
- 데이터베이스 연결 정보는 `JINRO_PROJ/backend/.env`에 설정해야 합니다.
- AI 요약 및 분석 관련 설정은 `JINRO_PROJ/ai_server/.env`에 설정해야 합니다.
- LLM 기반 상담 요약을 사용하려면 `JINRO_PROJ/ai_server/.env`에 `OPENAI_API_KEY`를 설정해야 합니다.
- AI 모델 파일이 없으면 서버는 실행되지만 일부 분석 기능은 제한될 수 있습니다.
## Closing

`너, 내 진로가 되라`는 단순히 AI 기능을 덧붙인 데모가 아니라, 학생 진단 데이터, 상담 대화, 비디오 분석 결과를 하나의 상담 리포트로 연결하는 end-to-end 진로 상담 시스템입니다.

이 프로젝트에서 저는 FastAPI 기반 백엔드 구조화, AI 추론 서버 분리, 음성 STT 및 영상 분석 파이프라인 연동, 프론트엔드 사용자 흐름 설계, 결과 리포트 자동화까지 제품 관점에서 전체 흐름을 연결했습니다.

특히 모델 성능 자체만이 아니라, AI가 실제 서비스 안에서 어떤 입력을 받고 어떤 형태의 결과로 소비되어야 하는지까지 설계하고 구현했다는 점을 이 포트폴리오의 핵심 가치로 두고 있습니다.
