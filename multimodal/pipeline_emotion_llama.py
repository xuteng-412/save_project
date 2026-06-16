from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

import httpx

from config.settings import settings
from multimodal.audio_emotion import get_audio_emotion_recognizer
from multimodal.emotion_fusion import build_signal, fuse_emotions

logger = logging.getLogger(__name__)

PipelineMode = Literal["sequential", "parallel"]
LlamaTask = Literal["emotion", "reason"]


@dataclass
class AVEmotionPipelineResult:
    transcript: str
    language: str = ""
    duration: float = 0.0
    audio_emotion: Optional[Dict[str, Any]] = None
    visual_emotion: Optional[Dict[str, Any]] = None
    fused_emotion: Optional[Dict[str, Any]] = None
    emotion_llama_text: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


def _which_ffmpeg() -> Optional[str]:
    return shutil.which("ffmpeg")


def _extract_wav_from_video(video_path: str) -> str:
    out = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot extract audio from video")
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        video_path,
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(out):
            os.unlink(out)
        raise RuntimeError(proc.stderr[:500] if proc.stderr else "ffmpeg failed while extracting wav")
    return out


def _audio_file_to_black_video_mp4(audio_path: str) -> str:
    ffmpeg = _which_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg not found; cannot create placeholder video for audio")
    out = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=black:s=320x240:r=1",
        "-i",
        audio_path,
        "-shortest",
        "-c:v",
        "libx264",
        "-tune",
        "stillimage",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        out,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        if os.path.exists(out):
            os.unlink(out)
        raise RuntimeError(proc.stderr[:500] if proc.stderr else "ffmpeg failed while creating placeholder video")
    return out


def _sample_frame_jpeg(video_path: str) -> Optional[bytes]:
    try:
        import cv2
    except Exception:
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames // 4))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    ok, buf = cv2.imencode(".jpg", frame)
    return buf.tobytes() if ok else None


def _build_prompt(
    transcript: str,
    task: LlamaTask,
    *,
    audio_hint: Optional[Dict[str, Any]] = None,
    visual_hint: Optional[Dict[str, Any]] = None,
    fused_hint: Optional[Dict[str, Any]] = None,
) -> str:
    pieces: List[str] = []
    pieces.append(f"Spoken content: {transcript.strip() or 'empty / unavailable'}. ")
    if audio_hint:
        pieces.append(
            f"Audio emotion estimate: {audio_hint.get('primary_emotion')} ({float(audio_hint.get('confidence') or 0):.2f}). "
        )
    if visual_hint:
        pieces.append(
            f"Visual facial estimate: {visual_hint.get('primary_emotion')} ({float(visual_hint.get('confidence') or 0):.2f}). "
        )
    if fused_hint:
        pieces.append(f"Fused emotion summary: {fused_hint.get('summary', '')}. ")
    if task == "reason":
        pieces.append(
            "[reason] Explain the speaker's facial expression, vocal tone, likely emotion, and intended meaning."
        )
    else:
        pieces.append("[emotion] What emotion is expressed in this clip?")
    return "".join(pieces)


def _transcribe_path_sync(audio_path: str, language: Optional[str]) -> Any:
    from multimodal.asr import get_speech_recognizer

    recognizer = get_speech_recognizer()
    return recognizer.transcribe(audio_path, language=language)


def _audio_emotion_sync(audio_path: str, transcript: str) -> Dict[str, Any]:
    result = get_audio_emotion_recognizer().recognize(audio_path=audio_path, transcript=transcript)
    return {
        "primary_emotion": result.primary_emotion,
        "confidence": result.confidence,
        "all_emotions": result.all_emotions,
        "model_name": result.model_name,
        "backend": result.backend,
        "warnings": result.warnings,
    }


def _visual_emotion_sync(jpeg_bytes: bytes) -> Dict[str, Any]:
    from multimodal.emotion import get_emotion_recognizer

    result = get_emotion_recognizer().recognize_from_bytes(jpeg_bytes)
    return {
        "primary_emotion": result.primary_emotion,
        "confidence": result.confidence,
        "all_emotions": result.all_emotions,
        "face_detected": result.face_detected,
        "model_name": settings.VISUAL_EMOTION_BACKEND,
    }


async def _call_gradio_predict(base_url: str, video_abs_path: str, prompt: str, timeout: float) -> str:
    url = base_url.rstrip("/") + "/api/predict/"
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, json={"data": [video_abs_path, prompt]})
        response.raise_for_status()
        payload = response.json()
    if isinstance(payload, dict) and payload.get("data"):
        first = payload["data"][0]
        return first if isinstance(first, str) else str(first)
    return str(payload)


async def run_av_emotion_pipeline(
    *,
    media_path: str,
    is_video: bool,
    language: Optional[str] = None,
    mode: PipelineMode = "sequential",
    llama_task: LlamaTask = "emotion",
    call_emotion_llama: bool = True,
) -> AVEmotionPipelineResult:
    warnings: List[str] = []
    temp_paths: List[str] = []
    video_for_llama = os.path.abspath(media_path)

    try:
        if is_video:
            wav_path = await asyncio.to_thread(_extract_wav_from_video, media_path)
            temp_paths.append(wav_path)
        else:
            wav_path = media_path
            if call_emotion_llama and settings.EMOTION_LLAMA_ENABLED:
                try:
                    placeholder = await asyncio.to_thread(_audio_file_to_black_video_mp4, media_path)
                    temp_paths.append(placeholder)
                    video_for_llama = placeholder
                except RuntimeError as exc:
                    warnings.append(str(exc))
                    call_emotion_llama = False

        transcription = await asyncio.to_thread(_transcribe_path_sync, wav_path, language)
        transcript = transcription.text
        audio_emotion = await asyncio.to_thread(_audio_emotion_sync, wav_path, transcript)

        visual_emotion: Optional[Dict[str, Any]] = None
        if is_video:
            jpeg = await asyncio.to_thread(_sample_frame_jpeg, media_path)
            if jpeg:
                try:
                    visual_emotion = await asyncio.to_thread(_visual_emotion_sync, jpeg)
                except Exception as exc:
                    warnings.append(f"visual emotion inference failed: {exc}")
            elif mode == "parallel":
                warnings.append("Failed to sample video frame for visual emotion analysis.")

        fused = None
        if settings.ENABLE_MULTIMODAL_EMOTION_FUSION:
            fused_obj = fuse_emotions(
                build_signal("audio", audio_emotion),
                build_signal("visual", visual_emotion),
            )
            fused = fused_obj.to_dict() if fused_obj else None

        llama_text: Optional[str] = None
        if call_emotion_llama and settings.EMOTION_LLAMA_ENABLED:
            prompt = _build_prompt(
                transcript,
                llama_task,
                audio_hint=audio_emotion,
                visual_hint=visual_emotion,
                fused_hint=fused,
            )
            try:
                llama_text = await _call_gradio_predict(
                    settings.EMOTION_LLAMA_GRADIO_URL,
                    video_for_llama,
                    prompt,
                    settings.EMOTION_LLAMA_TIMEOUT,
                )
            except Exception as exc:
                warnings.append(f"Emotion-LLaMA call failed: {exc}")
                logger.exception("Emotion-LLaMA request failed")
        elif call_emotion_llama and not settings.EMOTION_LLAMA_ENABLED:
            warnings.append("Emotion-LLaMA disabled; returning specialist emotion signals only.")

        return AVEmotionPipelineResult(
            transcript=transcript,
            language=transcription.language or "",
            duration=float(transcription.duration),
            audio_emotion=audio_emotion,
            visual_emotion=visual_emotion,
            fused_emotion=fused,
            emotion_llama_text=llama_text,
            warnings=warnings,
        )
    finally:
        for path in temp_paths:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except OSError:
                pass
