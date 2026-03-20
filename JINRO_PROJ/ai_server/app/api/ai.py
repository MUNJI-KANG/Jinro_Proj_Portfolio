import asyncio
import logging
import os
import shutil
from pathlib import Path

import httpx
import mediapipe as mp
import requests
import torch
import torch.nn as nn
from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from torchvision import models, transforms

from app.schemas.ai import AnalysisRequest, SummaryRequest
from app.services.focuse_service import FrameMobileNetV2, analyze_video_to_json
from app.services.interest_analyze import analyze_video_with_face_crop
from app.services.stt_service import speech_to_text
from app.services.summary_service import summarize_text


logger = logging.getLogger("ai_server.router")
router = APIRouter(prefix="/ai", tags=["Client"])

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

API_DIR = Path(__file__).resolve().parent
SERVER_DIR = API_DIR.parent.parent
UPLOAD_DIR = SERVER_DIR / "audio_uploads"
UPLOAD_VIDEO = SERVER_DIR / "videos"
MODEL_DIR = SERVER_DIR / "model"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_VIDEO.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class_names = ["interested", "not_interested"]
test_transforms = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

analysis_semaphore = asyncio.Semaphore(1)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


def _resolve_model_path(*candidates: str):
    for candidate in candidates:
        path = MODEL_DIR / candidate
        if path.exists():
            return path
    return None


def _load_interest_model():
    model = models.resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model_path = _resolve_model_path("interest_classifier_best.pth")
    if model_path is None:
        logger.warning("Interest model file was not found in %s", MODEL_DIR)
        return None

    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Loaded interest model from %s", model_path)
    return model


def _load_focus_model():
    model_path = _resolve_model_path(
        "best_focus_model_frame.pth",
        "focus_best.pt",
        "daisee_best_focus_model.pth",
        "daisee_best_focus_model_val.pth",
    )
    if model_path is None:
        logger.warning("Focus model file was not found in %s", MODEL_DIR)
        return None

    model = FrameMobileNetV2(num_classes=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info("Loaded focus model from %s", model_path)
    return model


interest_model = _load_interest_model()
focus_model = _load_focus_model()


@router.get("/")
def get_client_list():
    return {"message": "AI server is running."}


@router.post("/audio/stt")
def audio_stt(data: dict):
    audio_path = data["audio_path"]
    text = speech_to_text(audio_path)
    return {"success": True, "text": text}


@router.post("/audio/analyze")
def audio_analyze(data: dict):
    audio_path = data["audio_path"]
    text = speech_to_text(audio_path)
    return {"success": True, "stt_text": text}


@router.post("/audio/upload/{counseling_id}")
def upload_audio(
    counseling_id: int,
    file: UploadFile = File(...),
    ai_report: str = Form(...),
):
    counseling_dir = UPLOAD_DIR / str(counseling_id)
    counseling_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename or "audio.webm").suffix or ".webm"
    filename = f"counseling_{counseling_id}{ext}"
    file_path = counseling_dir / filename

    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stt_result = speech_to_text(file_path)
    stt_text = stt_result["text"]
    summary = summarize_text(stt_result, ai_report)

    try:
        res = requests.post(
            f"{BACKEND_URL}/counselor/report/con/{counseling_id}/stt-result",
            json={
                "stt_text": stt_text,
                "summary": summary.get("summary", ""),
                "analysis": summary,
                "career_recommendation": summary.get("career_recommendation", []),
            },
            timeout=30,
        )
        res.raise_for_status()
    except Exception as exc:
        logger.error("Failed to send STT result to backend: %s", exc)

    return {"success": True, "stt_text": stt_text}


@router.post("/api/summarize")
async def summarize_api(summary_request: SummaryRequest):
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": summary_request.model,
                    "messages": [
                        {"role": "system", "content": summary_request.system_prompt},
                        {"role": "user", "content": summary_request.text},
                    ],
                    "stream": False,
                },
            )
            response.raise_for_status()
            payload = response.json()

        return {
            "success": True,
            "model": summary_request.model,
            "summary": payload.get("message", {}).get("content", ""),
        }
    except Exception as exc:
        logger.error("Ollama summarization failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/audio/load/{counseling_id}")
async def audio_load(counseling_id: int):
    counseling_dir = UPLOAD_DIR / str(counseling_id)
    candidates = sorted(counseling_dir.glob(f"counseling_{counseling_id}.*"))
    if not candidates:
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=candidates[0], media_type="audio/mpeg")


def run_ai_analysis(counseling_id: int, client_id: int, report_id: int):
    logger.info(
        "Queued legacy video analysis trigger: counseling=%s client=%s report=%s",
        counseling_id,
        client_id,
        report_id,
    )


@router.post("/upload-video")
async def ai_upload_video(
    background_tasks: BackgroundTasks,
    counseling_id: int = Form(...),
    client_id: int = Form(...),
    report_id: int = Form(...),
    c_id: str = Form(...),
    file: UploadFile = File(...),
):
    try:
        counseling_folder = UPLOAD_VIDEO / str(counseling_id)
        counseling_folder.mkdir(parents=True, exist_ok=True)

        numbers = []
        for existing_file in counseling_folder.glob(f"{c_id}_*.webm"):
            try:
                numbers.append(int(existing_file.stem.split("_")[1]))
            except Exception:
                continue

        next_number = max(numbers, default=0) + 1
        filename = f"{c_id}_{next_number}.webm"
        file_path = counseling_folder / filename

        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if next_number >= 3:
            background_tasks.add_task(run_ai_analysis, counseling_id, client_id, report_id)

        return {
            "success": True,
            "message": "Video upload succeeded.",
            "filename": filename,
            "next_number": next_number,
        }
    except Exception as exc:
        logger.error("Video upload failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


async def _analyze_single_video(sample_video_path: Path, survey_score: float):
    if interest_model is None or focus_model is None:
        return {"interest": 0.0, "focused": 0.0, "survey_score": survey_score}

    interest_score = 0.0
    focus_score = 0.0

    df_results, interest_stats = await asyncio.to_thread(
        analyze_video_with_face_crop,
        str(sample_video_path),
        interest_model,
        test_transforms,
        class_names,
        device,
        face_detector,
        5,
        0.3,
    )
    if df_results is not None and interest_stats:
        interest_score = interest_stats.get("Interested_Percentage", 0.0)

    focus_stats = await asyncio.to_thread(
        analyze_video_to_json,
        str(sample_video_path),
        focus_model,
        device,
        "test_img",
        5,
    )
    if isinstance(focus_stats, dict):
        focus_score = focus_stats.get("focus_rate", 0.0)

    return {
        "interest": interest_score,
        "focused": focus_score,
        "survey_score": survey_score,
    }


async def run_full_analysis(request: AnalysisRequest):
    results = []
    max_retries = 24

    for task in request.videos:
        sample_video_path = UPLOAD_VIDEO / str(request.counseling_id) / f"{request.c_id}_{task.idx}.webm"

        file_ready = False
        for _ in range(max_retries):
            if sample_video_path.exists():
                file_ready = True
                break
            await asyncio.sleep(5)

        if not file_ready:
            results.append(
                {
                    "ai_v_erp_id": task.ai_v_erp_id,
                    "survey_score": task.survey_score,
                    "interest": 0.0,
                    "focused": 0.0,
                }
            )
            continue

        async with analysis_semaphore:
            try:
                scores = await _analyze_single_video(sample_video_path, task.survey_score)
                results.append(
                    {
                        "ai_v_erp_id": task.ai_v_erp_id,
                        "survey_score": scores["survey_score"],
                        "interest": scores["interest"],
                        "focused": scores["focused"],
                    }
                )
            except Exception as exc:
                logger.error("Video analysis failed for %s: %s", sample_video_path, exc, exc_info=True)
                results.append(
                    {
                        "ai_v_erp_id": task.ai_v_erp_id,
                        "survey_score": task.survey_score,
                        "interest": 0.0,
                        "focused": 0.0,
                    }
                )

    callback_payload = {"status": "success", "results": results}
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{BACKEND_URL}/client/analysis/callback",
                json=callback_payload,
                timeout=10.0,
            )
    except Exception as exc:
        logger.error("Failed to send analysis callback: %s", exc)


@router.post("/start-analysis")
async def start_analysis_endpoint(request: AnalysisRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_full_analysis, request)
    return {"message": "Analysis task registered."}
