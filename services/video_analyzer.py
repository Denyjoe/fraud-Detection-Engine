"""Video analysis: evenly sampled frames through the image checks, plus optional audio.

* Frames: OpenCV samples ``video.frames_to_sample`` frames spread over the first
  ``video.max_seconds_analyzed`` seconds; each frame goes through
  :func:`services.image_analyzer.analyse_image` (OCR on the first, middle and last
  sampled frames only, for speed).
* Audio (optional, ``video.transcribe_audio`` + ``models.speech_to_text.enabled``):
  ffmpeg (bundled by ``imageio-ffmpeg``) extracts 16 kHz mono audio, Whisper
  transcribes it (Swahili and English supported) and the transcript is checked with
  the same lexicon and contact rules as the listing text.
"""

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import numpy as np
from PIL import Image

from config.settings import CONFIG
from services.image_analyzer import ImageAnalysis, analyse_image
from services.model_registry import pipeline_device, register

logger = logging.getLogger(__name__)

VIDEO_CFG: Dict[str, Any] = CONFIG["video"]


def _load_whisper():
    """Whisper speech-to-text pipeline."""
    from transformers import pipeline

    return pipeline("automatic-speech-recognition", model=CONFIG["models"]["speech_to_text"]["name"],
                    device=pipeline_device())


WHISPER_MODEL = register("speech_to_text", _load_whisper, "Whisper speech-to-text (video audio)")


@dataclass
class VideoAnalysis:
    """Results for one video file."""

    source: str
    duration_seconds: float = 0.0
    frames: List[ImageAnalysis] = field(default_factory=list)
    transcript: str = ""
    checks: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serialisable view."""
        return {
            "source": self.source,
            "duration_seconds": round(self.duration_seconds, 2),
            "frames_analysed": len(self.frames),
            "frames": [f.to_dict() for f in self.frames],
            "transcript": self.transcript,
            "checks": self.checks,
        }


def _ffmpeg_exe() -> Optional[str]:
    """Path to an ffmpeg binary (imageio-ffmpeg's bundled one, else PATH)."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        from shutil import which

        return which("ffmpeg")


def _extract_audio(path: str, max_seconds: float) -> Optional[np.ndarray]:
    """Decode the first ``max_seconds`` of audio as 16 kHz mono float32.

    Parameters:
        path (str): Video file path.
        max_seconds (float): Maximum duration to decode.

    Returns:
        Optional[np.ndarray]: Audio samples, or None when there is no audio / no ffmpeg.
    """
    exe = _ffmpeg_exe()
    if exe is None:
        return None
    cmd = [exe, "-v", "error", "-i", path, "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", "16000",
           "-f", "f32le", "pipe:1"]
    proc = subprocess.run(cmd, capture_output=True, timeout=300, check=False)
    if proc.returncode != 0 or not proc.stdout:
        return None
    return np.frombuffer(proc.stdout, dtype=np.float32)


def transcribe(path: str) -> Optional[str]:
    """Transcribe a video's audio track with Whisper (when enabled).

    Parameters:
        path (str): Video file path.

    Returns:
        Optional[str]: Transcript, or None when disabled/unavailable/silent.
    """
    if not VIDEO_CFG.get("transcribe_audio", False):
        return None
    model = WHISPER_MODEL.get()
    if model is None:
        return None
    audio = _extract_audio(path, float(VIDEO_CFG.get("max_seconds_analyzed", 600)))
    if audio is None or audio.size < 16000 // 2:
        return None
    out = model({"raw": audio, "sampling_rate": 16000}, return_timestamps=True)
    return str(out.get("text", "")).strip()


def analyse_video_file(
    path: str,
    source: str,
    claimed_concept: Optional[str] = None,
    compatible_keys: Optional[Set[str]] = None,
) -> VideoAnalysis:
    """Sample frames from a video file and run the image checks (plus optional audio).

    Parameters:
        path (str): Video file path on disk.
        source (str): Original file name for reporting.
        claimed_concept (Optional[str]): Seller's claimed product (English).
        compatible_keys (Optional[Set[str]]): Taxonomy keys matching the claim.

    Returns:
        VideoAnalysis: Per-frame results and transcript.
    """
    import cv2

    result = VideoAnalysis(source=source)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        result.checks["frames"] = "error: could not open video"
        return result
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        result.duration_seconds = total / fps if fps else 0.0
        usable = min(total, int(fps * float(VIDEO_CFG.get("max_seconds_analyzed", 600)))) if total else 0
        n = max(1, int(VIDEO_CFG.get("frames_to_sample", 8)))
        indices = sorted({int(i) for i in np.linspace(0, max(usable - 1, 0), num=n)}) if usable else [0]
        ocr_frames = {indices[0], indices[len(indices) // 2], indices[-1]}
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            label = f"{source}@{index / fps:.1f}s"
            result.frames.append(
                analyse_image(image, label, claimed_concept, compatible_keys, run_ocr=index in ocr_frames)
            )
        result.checks["frames"] = f"ok ({len(result.frames)} sampled)"
    finally:
        cap.release()

    try:
        transcript = transcribe(path)
        if transcript is not None:
            result.transcript = transcript
            result.checks["audio"] = "ok"
        else:
            result.checks["audio"] = "disabled" if not VIDEO_CFG.get("transcribe_audio") else WHISPER_MODEL.status
    except Exception as exc:
        logger.warning("Audio transcription failed for %s: %s", source, exc)
        result.checks["audio"] = "error"
    return result


def analyse_video_bytes(
    data: bytes,
    source: str,
    claimed_concept: Optional[str] = None,
    compatible_keys: Optional[Set[str]] = None,
) -> VideoAnalysis:
    """Analyse an uploaded video given as bytes (written to a temporary file).

    Parameters:
        data (bytes): Video file content.
        source (str): Original file name.
        claimed_concept (Optional[str]): Seller's claimed product (English).
        compatible_keys (Optional[Set[str]]): Taxonomy keys matching the claim.

    Returns:
        VideoAnalysis: Per-frame results and transcript.
    """
    suffix = os.path.splitext(source)[1] or ".mp4"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
        return analyse_video_file(tmp_path, source, claimed_concept, compatible_keys)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            logger.warning("Could not delete temporary video file %s", tmp_path)
