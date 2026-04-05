import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, send_file
from groq import Groq
from werkzeug.utils import secure_filename

load_dotenv()

APP_NAME = "AutoSubtitles"
BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "tmp_storage"
UPLOAD_DIR = STORAGE_DIR / "uploads"
AUDIO_DIR = STORAGE_DIR / "audio"
RENDER_DIR = STORAGE_DIR / "renders"

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "200"))
MAX_VIDEO_DURATION_SEC = int(os.getenv("MAX_VIDEO_DURATION_SEC", "900"))
FILE_TTL_SEC = int(os.getenv("FILE_TTL_SEC", "3600"))
CLEANUP_INTERVAL_SEC = int(os.getenv("CLEANUP_INTERVAL_SEC", "300"))

ALLOWED_VIDEO_EXTENSIONS = {"mp4", "mov", "mkv", "webm", "avi", "m4v"}

for folder in (UPLOAD_DIR, AUDIO_DIR, RENDER_DIR):
    folder.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, template_folder="frontend/templates", static_folder="frontend/static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


def now_ts() -> int:
    return int(time.time())


def safe_ext(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise ValueError(f"Unsupported extension: .{ext}")
    return ext


def ffprobe_duration(video_path: Path) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def extract_audio(video_path: Path, wav_path: Path) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def srt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    ms = int((sec - int(sec)) * 1000)
    total = int(sec)
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02}:{mm:02}:{ss:02},{ms:03}"


def build_srt(cues: list[dict]) -> str:
    chunks = []
    for idx, cue in enumerate(cues, start=1):
        text = str(cue.get("text", "")).replace("\n", " ").strip()
        if not text:
            continue
        start = float(cue.get("start", 0.0))
        end = float(cue.get("end", start + 0.5))
        if end <= start:
            end = start + 0.4
        chunks.append(f"{idx}\n{srt_time(start)} --> {srt_time(end)}\n{text}\n")
    return "\n".join(chunks)


def normalize_response_words(transcript_json: dict) -> tuple[list[dict], list[dict]]:
    words = []
    segments = []
    raw_segments = transcript_json.get("segments") or []

    for seg in raw_segments:
        seg_text = (seg.get("text") or "").strip()
        seg_start = float(seg.get("start") or 0.0)
        seg_end = float(seg.get("end") or max(seg_start + 0.4, 0.4))
        segment_words = []
        for w in seg.get("words") or []:
            w_text = (w.get("word") or "").strip()
            if not w_text:
                continue
            w_start = float(w.get("start") or seg_start)
            w_end = float(w.get("end") or max(w_start + 0.2, w_start + 0.01))
            item = {"text": w_text, "start": w_start, "end": w_end}
            words.append(item)
            segment_words.append(item)

        if not segment_words and seg_text:
            pseudo_words = seg_text.split()
            duration = max(seg_end - seg_start, 0.2)
            step = duration / max(len(pseudo_words), 1)
            for idx, token in enumerate(pseudo_words):
                w_start = seg_start + idx * step
                w_end = min(seg_end, w_start + step)
                item = {"text": token, "start": w_start, "end": w_end}
                words.append(item)
                segment_words.append(item)

        segments.append(
            {
                "text": seg_text,
                "start": seg_start,
                "end": seg_end,
                "words": segment_words,
            }
        )

    words.sort(key=lambda x: x["start"])
    return words, segments


def make_groq_client() -> Groq:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")
    return Groq(api_key=api_key)


def transcribe_with_groq(audio_path: Path, language: str | None = None) -> dict:
    client = make_groq_client()
    with audio_path.open("rb") as audio_file:
        payload = {
            "file": (audio_path.name, audio_file.read()),
            "model": os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3-turbo"),
            "response_format": "verbose_json",
            "timestamp_granularities": ["word", "segment"],
            "temperature": 0,
        }
        if language:
            payload["language"] = language

        response = client.audio.transcriptions.create(**payload)

    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, dict):
        return response
    return json.loads(str(response))


def to_ass_style(style: dict) -> str:
    font = style.get("fontFamily", "Montserrat")
    size = int(style.get("fontSize", 46))
    bold = -1 if style.get("bold", True) else 0
    italic = -1 if style.get("italic", False) else 0
    outline = int(style.get("outlineWidth", 3))
    shadow = int(style.get("shadow", 2))

    text_color = style.get("color", "#FFFFFF")
    box_color = style.get("bgColor", "#000000")
    bg_opacity = float(style.get("bgOpacity", 0.4))

    def hex_to_ass_bgr(hex_color: str, alpha: int = 0) -> str:
        hc = hex_color.lstrip("#")
        if len(hc) != 6:
            hc = "FFFFFF"
        rr, gg, bb = hc[0:2], hc[2:4], hc[4:6]
        return f"&H{alpha:02X}{bb}{gg}{rr}"

    primary = hex_to_ass_bgr(text_color, 0)
    back_alpha = max(0, min(255, int((1 - bg_opacity) * 255)))
    back = hex_to_ass_bgr(box_color, back_alpha)

    return (
        f"Style: Default,{font},{size},{primary},&H000000FF,{back},&H00000000,"
        f"{bold},{italic},0,0,100,100,0,0,1,{outline},{shadow},2,10,10,24,1"
    )


def write_ass_file(cues: list[dict], style: dict, ass_path: Path) -> None:
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes
WrapStyle: 2

[V4+ Styles]
Format: Name,Fontname,Fontsize,PrimaryColour,SecondaryColour,OutlineColour,BackColour,Bold,Italic,Underline,StrikeOut,ScaleX,ScaleY,Spacing,Angle,BorderStyle,Outline,Shadow,Alignment,MarginL,MarginR,MarginV,Encoding
"""
    events = """[Events]
Format: Layer,Start,End,Style,Name,MarginL,MarginR,MarginV,Effect,Text
"""

    rows = []
    for cue in cues:
        start = srt_time(float(cue.get("start", 0.0))).replace(",", ".")
        end = srt_time(float(cue.get("end", 0.5))).replace(",", ".")
        text = str(cue.get("text", "")).replace("\n", "\\N")
        text = re.sub(r"[{}]", "", text)
        rows.append(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}")

    content = header + to_ass_style(style) + "\n\n" + events + "\n".join(rows) + "\n"
    ass_path.write_text(content, encoding="utf-8")


def burn_subtitles(video_path: Path, cues: list[dict], style: dict) -> Path:
    render_id = uuid.uuid4().hex
    out_path = RENDER_DIR / f"{render_id}.mp4"
    subtitle_mode = (style or {}).get("subtitleMode", "srt")

    if subtitle_mode == "ass":
        ass_path = RENDER_DIR / f"{render_id}.ass"
        write_ass_file(cues, style or {}, ass_path)
        sub_filter = f"subtitles='{ass_path.as_posix()}'"
    else:
        srt_path = RENDER_DIR / f"{render_id}.srt"
        srt_path.write_text(build_srt(cues), encoding="utf-8")
        sub_filter = f"subtitles='{srt_path.as_posix()}'"

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        sub_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_path


def metadata_path(file_id: str) -> Path:
    return STORAGE_DIR / f"{file_id}.json"


def save_metadata(file_id: str, payload: dict) -> None:
    payload["updated_at"] = now_ts()
    metadata_path(file_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_metadata(file_id: str) -> dict:
    meta = metadata_path(file_id)
    if not meta.exists():
        raise FileNotFoundError("Unknown file id")
    return json.loads(meta.read_text(encoding="utf-8"))


def cleanup_storage_loop() -> None:
    while True:
        try:
            expire_before = now_ts() - FILE_TTL_SEC
            for meta in STORAGE_DIR.glob("*.json"):
                stat = meta.stat()
                if int(stat.st_mtime) < expire_before:
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    for key in ("video_path", "audio_path", "render_path"):
                        p = data.get(key)
                        if p and Path(p).exists():
                            Path(p).unlink(missing_ok=True)
                    meta.unlink(missing_ok=True)
            for dir_path in (UPLOAD_DIR, AUDIO_DIR, RENDER_DIR):
                for file_path in dir_path.glob("*"):
                    if int(file_path.stat().st_mtime) < expire_before:
                        file_path.unlink(missing_ok=True)
        except Exception:
            pass
        time.sleep(CLEANUP_INTERVAL_SEC)


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": f"File too large. Max {MAX_FILE_SIZE_MB}MB"}), 413


@app.route("/")
def index():
    return render_template("index.html")




@app.get("/api/health")
def api_health():
    return jsonify({"ok": True, "app": APP_NAME})

@app.post("/api/upload")
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    try:
        ext = safe_ext(file.filename)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    file_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    save_path = UPLOAD_DIR / f"{file_id}.{ext}"
    file.save(save_path)

    try:
        duration = ffprobe_duration(save_path)
    except Exception:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": "Failed to probe video (ffprobe missing?)"}), 500

    if duration > MAX_VIDEO_DURATION_SEC:
        save_path.unlink(missing_ok=True)
        return jsonify({"error": f"Video too long. Max {MAX_VIDEO_DURATION_SEC}s"}), 400

    save_metadata(
        file_id,
        {
            "id": file_id,
            "original_name": filename,
            "video_path": str(save_path),
            "duration": duration,
            "created_at": now_ts(),
        },
    )

    return jsonify(
        {
            "ok": True,
            "file_id": file_id,
            "filename": filename,
            "duration": duration,
            "max_file_size_mb": MAX_FILE_SIZE_MB,
            "max_video_duration_sec": MAX_VIDEO_DURATION_SEC,
        }
    )


@app.post("/api/transcribe")
def api_transcribe():
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id")
    language = data.get("language")

    if not file_id:
        return jsonify({"error": "file_id is required"}), 400

    try:
        meta = load_metadata(file_id)
        video_path = Path(meta["video_path"])
        if not video_path.exists():
            return jsonify({"error": "Video file not found"}), 404

        audio_path = AUDIO_DIR / f"{file_id}.wav"
        extract_audio(video_path, audio_path)
        transcript_json = transcribe_with_groq(audio_path, language if language and language != "auto" else None)
        words, segments = normalize_response_words(transcript_json)

        meta["audio_path"] = str(audio_path)
        meta["words_count"] = len(words)
        save_metadata(file_id, meta)

        return jsonify(
            {
                "ok": True,
                "file_id": file_id,
                "language": transcript_json.get("language"),
                "duration": transcript_json.get("duration") or meta.get("duration"),
                "text": transcript_json.get("text", ""),
                "words": words,
                "segments": segments,
            }
        )
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"ffmpeg failed: {exc.stderr[-800:]}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Transcribe failed: {exc}"}), 500


@app.post("/api/render")
def api_render():
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id")
    cues = data.get("cues") or []
    style = data.get("style") or {}

    if not file_id:
        return jsonify({"error": "file_id is required"}), 400
    if not cues:
        return jsonify({"error": "cues[] are required"}), 400

    try:
        meta = load_metadata(file_id)
        video_path = Path(meta["video_path"])
        if not video_path.exists():
            return jsonify({"error": "Video file not found"}), 404

        out_path = burn_subtitles(video_path, cues, style)
        render_id = out_path.stem

        meta["render_path"] = str(out_path)
        meta["render_id"] = render_id
        save_metadata(file_id, meta)

        save_metadata(
            render_id,
            {
                "id": render_id,
                "kind": "render",
                "render_path": str(out_path),
                "created_at": now_ts(),
            },
        )

        return jsonify({"ok": True, "render_id": render_id, "download": f"/api/download/{render_id}"})
    except subprocess.CalledProcessError as exc:
        return jsonify({"error": f"ffmpeg render failed: {exc.stderr[-800:]}"}), 500
    except Exception as exc:
        return jsonify({"error": f"Render failed: {exc}"}), 500


@app.get("/api/download/<file_id>")
def api_download(file_id: str):
    try:
        meta = load_metadata(file_id)
    except FileNotFoundError:
        return jsonify({"error": "Unknown id"}), 404

    render_path = meta.get("render_path")
    if not render_path:
        return jsonify({"error": "No render for this id"}), 404

    path = Path(render_path)
    if not path.exists():
        return jsonify({"error": "Output missing"}), 404

    return send_file(path, as_attachment=True, download_name=f"{file_id}.mp4")


if __name__ == "__main__":
    cleanup_thread = threading.Thread(target=cleanup_storage_loop, daemon=True)
    cleanup_thread.start()

    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=False)
