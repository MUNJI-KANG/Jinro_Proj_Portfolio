import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def clean_text(text: str):
    text = re.sub(r"\b(음|어|그|저)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


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


def summarize_chunk(chunk: str):
    client = get_client()
    if client is None:
        return chunk[:300]

    prompt = f"""
다음은 학생과 상담사가 진행한 진로 상담 대화입니다.

핵심 내용만 간단히 3~4문장으로 요약해 주세요.

상담 대화:
{chunk}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 진로 상담 내용을 정리하는 AI입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=200,
    )

    return res.choices[0].message.content.strip()


def summarize_chunks(chunks):
    summaries = []

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(summarize_chunk, chunks)

    for result in results:
        summaries.append(result)

    return summaries


def summarize_final(text, video_analyze):
    client = get_client()
    if client is None:
        return {
            "interest_field": "",
            "low_interest_field": "",
            "student_trait": "",
            "career_recommendation": [],
            "summary": text,
        }

    prompt = f"""
당신은 고등학교 진로 상담 내용을 종합 정리하는 AI입니다.
상담 요약과 영상 분석 결과를 함께 보고 아래 JSON 형식으로만 답변하세요.

입력 상담 요약:
{text}

입력 영상 분석:
{video_analyze}

반드시 아래 키를 모두 포함하세요.
{{
  "interest_field": "",
  "low_interest_field": "",
  "student_trait": "",
  "career_recommendation": [],
  "summary": ""
}}
"""

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "진로 상담 분석 AI입니다."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )

    content = res.choices[0].message.content.strip()
    content = re.sub(r"^```json", "", content)
    content = re.sub(r"^```", "", content)
    content = re.sub(r"```$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except Exception:
        return {
            "interest_field": "",
            "low_interest_field": "",
            "student_trait": "",
            "career_recommendation": [],
            "summary": content,
        }


def summarize_text(stt_result, ai_report):
    if not stt_result or "segments" not in stt_result or not stt_result["segments"]:
        return {
            "interest_field": "",
            "low_interest_field": "",
            "student_trait": "",
            "career_recommendation": [],
            "summary": "음성 인식 결과가 없습니다.",
        }

    segments = stt_result["segments"]
    chunks = build_chunks_from_segments(segments)
    chunk_summaries = summarize_chunks(chunks)
    merged = "\n".join(chunk_summaries)
    return summarize_final(merged, ai_report)
