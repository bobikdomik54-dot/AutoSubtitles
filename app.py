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
from flask import Flask, jsonify, make_response, request, send_file
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

app = Flask(__name__)
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
    html = """
<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width,initial-scale=1" />
<title>AutoSubtitles</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&family=Bebas+Neue&family=Bangers&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box} body{margin:0;font-family:Montserrat,sans-serif;background:#0f1320;color:#fff} .wrap{display:grid;grid-template-columns:1.3fr 1fr;gap:16px;padding:16px;min-height:100vh}
.panel{background:#171d2f;border-radius:14px;padding:14px;box-shadow:0 5px 24px rgba(0,0,0,.35)}
.drop{border:2px dashed #3f4f7a;border-radius:12px;padding:16px;text-align:center;cursor:pointer;margin-bottom:10px}
.drop.drag{background:#1f2740}
video{width:100%;border-radius:10px;max-height:58vh;background:#000}
#overlay{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none;display:flex;align-items:flex-end;justify-content:center;padding-bottom:9%}
#subtitlePreview{max-width:90%;text-align:center;line-height:1.2;white-space:pre-wrap}
.videoBox{position:relative}
.row{display:flex;gap:8px;flex-wrap:wrap}.row>*{flex:1}
label{font-size:12px;display:block;margin:6px 0}
input,select,button,textarea{width:100%;padding:8px;border-radius:8px;border:1px solid #2f3f67;background:#0f1320;color:#fff}
button{cursor:pointer;background:#2f69ff;border:0}.btn2{background:#34405f}.btn3{background:#3b8855}
.small{font-size:12px;color:#a8b5dd}
.cue{border:1px solid #2a355a;border-radius:10px;padding:8px;margin:6px 0;background:#0f1320}
.cueTime{display:flex;gap:6px}
.cueActions{display:flex;gap:6px;margin-top:6px}
.badge{display:inline-block;padding:2px 8px;border-radius:999px;background:#2a355a;font-size:12px}
</style>
</head>
<body>
<div class="wrap">
  <div class="panel">
    <h2>🎬 AutoSubtitles</h2>
    <div id="drop" class="drop">Перетащите видео сюда или нажмите для выбора</div>
    <input id="fileInput" type="file" accept="video/*" style="display:none">
    <div class="small" id="fileInfo">Файл не выбран</div>
    <div class="row" style="margin-top:10px">
      <button id="uploadBtn" class="btn2">1) Загрузить</button>
      <button id="transcribeBtn">2) Транскрибировать</button>
      <button id="styleBtn" class="btn2">3) Применить стиль</button>
      <button id="exportBtn" class="btn3">4) Экспортировать</button>
    </div>
    <div class="row" style="margin-top:8px">
      <div><label>Экспорт</label><select id="exportMode"><option value="backend">A: ffmpeg backend</option><option value="frontend">B: MediaRecorder frontend</option></select></div>
      <div><label>Язык</label><input id="lang" placeholder="ru / en / auto"></div>
      <div><label>Пресет</label><select id="preset"><option>Classic</option><option>Highlight</option><option>Beast-like</option></select></div>
      <div><label>Анимация</label><select id="anim"><option>none</option><option>fade</option><option>pop</option><option>highlight-word</option></select></div>
    </div>

    <div class="videoBox" style="margin-top:12px">
      <video id="video" controls></video>
      <div id="overlay"><div id="subtitlePreview"></div></div>
    </div>
    <div class="small" id="status" style="margin-top:8px">Готово.</div>
  </div>

  <div class="panel">
    <h3>Стили</h3>
    <div class="row">
      <div><label>Шрифт</label><select id="font"><option>Montserrat</option><option>Bebas Neue</option><option>Bangers</option><option>Impact</option><option>Arial Black</option><option>Comic Sans MS</option></select></div>
      <div><label>Размер</label><input id="fontSize" type="number" value="46"></div>
      <div><label>Жирный</label><select id="bold"><option value="1">on</option><option value="0">off</option></select></div>
      <div><label>Курсив</label><select id="italic"><option value="0">off</option><option value="1">on</option></select></div>
    </div>
    <div class="row">
      <div><label>Цвет</label><input id="color" type="color" value="#ffffff"></div>
      <div><label>Обводка</label><input id="outlineWidth" type="number" value="3"></div>
      <div><label>Тень</label><input id="shadow" type="number" value="2"></div>
      <div><label>Межстрочный</label><input id="lineHeight" type="number" step="0.05" value="1.2"></div>
    </div>
    <div class="row">
      <div><label>Фон</label><select id="bgOn"><option value="1">on</option><option value="0">off</option></select></div>
      <div><label>Цвет фона</label><input id="bgColor" type="color" value="#000000"></div>
      <div><label>Opacity</label><input id="bgOpacity" type="number" step="0.05" value="0.4"></div>
      <div><label>Padding</label><input id="padding" type="number" value="8"></div>
      <div><label>Radius</label><input id="radius" type="number" value="8"></div>
    </div>
    <div class="row">
      <div><label>Позиция X%</label><input id="posX" type="number" value="50"></div>
      <div><label>Позиция Y%</label><input id="posY" type="number" value="84"></div>
      <div><label>Align H</label><select id="alignH"><option>center</option><option>left</option><option>right</option></select></div>
      <div><label>Align V</label><select id="alignV"><option>bottom</option><option>middle</option><option>top</option></select></div>
      <div><label>Max слов/cue</label><input id="maxWords" type="number" value="7"></div>
    </div>

    <h3>Субтитры <span id="count" class="badge">0 cues</span></h3>
    <div class="small">Можно редактировать текст/тайминг, split и merge.</div>
    <div id="cueList" style="max-height:42vh;overflow:auto;margin-top:8px"></div>
  </div>
</div>

<script>
const state = {
  file: null,
  fileId: null,
  words: [],
  segments: [],
  cues: [],
  style: {},
  renderedId: null
};

const $ = (id) => document.getElementById(id);
const drop = $('drop');
const fileInput = $('fileInput');
const video = $('video');
const subtitlePreview = $('subtitlePreview');

function log(msg){ $('status').textContent = msg; }
function clamp(n,a,b){ return Math.max(a, Math.min(b, n)); }
function sec(v){ return Number(v || 0); }
function fmt(t){ t=sec(t); const m=Math.floor(t/60); const s=(t%60).toFixed(2).padStart(5,'0'); return `${m}:${s}`; }

function styleFromUI(){
  return {
    fontFamily: $('font').value,
    fontSize: Number($('fontSize').value||46),
    bold: $('bold').value === '1',
    italic: $('italic').value === '1',
    lineHeight: Number($('lineHeight').value||1.2),
    color: $('color').value,
    outlineWidth: Number($('outlineWidth').value||3),
    shadow: Number($('shadow').value||2),
    bgOn: $('bgOn').value === '1',
    bgColor: $('bgColor').value,
    bgOpacity: Number($('bgOpacity').value||0.4),
    padding: Number($('padding').value||8),
    radius: Number($('radius').value||8),
    posX: Number($('posX').value||50),
    posY: Number($('posY').value||84),
    alignH: $('alignH').value,
    alignV: $('alignV').value,
    animation: $('anim').value,
    preset: $('preset').value,
    maxWords: Number($('maxWords').value||7),
    subtitleMode: 'ass'
  };
}

function applyPreset(){
  const p = $('preset').value;
  if (p === 'Classic') {
    $('font').value='Montserrat'; $('fontSize').value=46; $('color').value='#FFFFFF'; $('outlineWidth').value=3; $('shadow').value=2;
  } else if (p === 'Highlight') {
    $('font').value='Bebas Neue'; $('fontSize').value=54; $('color').value='#FFE700'; $('outlineWidth').value=4; $('shadow').value=0;
  } else {
    $('font').value='Bangers'; $('fontSize').value=58; $('color').value='#FFFFFF'; $('outlineWidth').value=5; $('shadow').value=4;
  }
  updatePreviewStyle();
}

function groupWordsToCues(words, maxWords=7, maxDuration=3.2){
  const cues=[];
  let current=[];
  let cueStart=null;

  for (const w of words){
    if (cueStart===null) cueStart = w.start;
    const tooLong = (w.end - cueStart) > maxDuration;
    if (current.length >= maxWords || tooLong){
      cues.push(cueFromWords(current));
      current = [];
      cueStart = w.start;
    }
    current.push(w);
  }
  if (current.length) cues.push(cueFromWords(current));

  return cues.map(c => autoWrapCue(c, 24));
}

function cueFromWords(arr){
  return {
    text: arr.map(x=>x.text).join(' ').trim(),
    start: arr[0]?.start || 0,
    end: arr[arr.length-1]?.end || (arr[0]?.start || 0) + 0.4,
    words: arr
  };
}

function autoWrapCue(cue, maxChars=24){
  const words = cue.text.split(/\s+/).filter(Boolean);
  let lines=[''];
  for (const w of words){
    let next = lines[lines.length-1] ? lines[lines.length-1] + ' ' + w : w;
    if (next.length > maxChars && lines[lines.length-1]) lines.push(w);
    else lines[lines.length-1] = next;
  }
  if (lines.length > 2){
    const half = Math.ceil(words.length/2);
    lines = [words.slice(0,half).join(' '), words.slice(half).join(' ')];
  }
  cue.text = lines.join('\n');
  return cue;
}

function updateCueList(){
  const box = $('cueList');
  box.innerHTML = '';
  state.cues.forEach((c, i) => {
    const el = document.createElement('div');
    el.className='cue';
    el.innerHTML = `
      <div class="cueTime">
        <input data-i="${i}" data-k="start" type="number" step="0.01" value="${c.start.toFixed(2)}">
        <input data-i="${i}" data-k="end" type="number" step="0.01" value="${c.end.toFixed(2)}">
      </div>
      <textarea data-i="${i}" data-k="text" rows="2">${c.text}</textarea>
      <div class="cueActions">
        <button data-act="split" data-i="${i}" class="btn2">Split</button>
        <button data-act="mergePrev" data-i="${i}" class="btn2">Merge ←</button>
        <button data-act="mergeNext" data-i="${i}" class="btn2">Merge →</button>
      </div>
      <div class="small">${fmt(c.start)} - ${fmt(c.end)}</div>
    `;
    box.appendChild(el);
  });
  $('count').textContent = `${state.cues.length} cues`;

  box.querySelectorAll('input[data-k], textarea[data-k]').forEach(inp => {
    inp.onchange = () => {
      const i = Number(inp.dataset.i);
      const k = inp.dataset.k;
      state.cues[i][k] = k==='text' ? inp.value : Number(inp.value);
      if (k !== 'text' && state.cues[i].end <= state.cues[i].start) state.cues[i].end = state.cues[i].start + 0.2;
      updatePreviewByTime(video.currentTime || 0);
    };
  });
  box.querySelectorAll('button[data-act]').forEach(btn => {
    btn.onclick = () => {
      const i = Number(btn.dataset.i); const act = btn.dataset.act;
      if (act==='split') splitCue(i);
      if (act==='mergePrev' && i>0) mergeCue(i-1, i);
      if (act==='mergeNext' && i<state.cues.length-1) mergeCue(i, i+1);
      updateCueList();
    }
  });
}

function splitCue(i){
  const c = state.cues[i];
  const parts = c.text.replace(/\n/g,' ').split(/\s+/).filter(Boolean);
  if (parts.length < 2) return;
  const m = Math.floor(parts.length/2);
  const mid = (c.start + c.end) / 2;
  state.cues.splice(i,1,
    autoWrapCue({text:parts.slice(0,m).join(' '), start:c.start, end:mid, words:[]},24),
    autoWrapCue({text:parts.slice(m).join(' '), start:mid, end:c.end, words:[]},24)
  );
}

function mergeCue(a,b){
  const c1=state.cues[a], c2=state.cues[b];
  state.cues.splice(a,2, autoWrapCue({
    text: (c1.text+' '+c2.text).replace(/\s+/g,' ').trim(),
    start: Math.min(c1.start,c2.start),
    end: Math.max(c1.end,c2.end),
    words: []
  },24));
}

function updatePreviewStyle(){
  state.style = styleFromUI();
  const s = state.style;
  subtitlePreview.style.fontFamily = s.fontFamily;
  subtitlePreview.style.fontSize = s.fontSize + 'px';
  subtitlePreview.style.fontWeight = s.bold ? '700':'400';
  subtitlePreview.style.fontStyle = s.italic ? 'italic':'normal';
  subtitlePreview.style.lineHeight = s.lineHeight;
  subtitlePreview.style.color = s.color;
  subtitlePreview.style.padding = s.bgOn ? `${s.padding}px` : '0';
  subtitlePreview.style.borderRadius = s.radius + 'px';
  subtitlePreview.style.background = s.bgOn ? hexToRgba(s.bgColor, s.bgOpacity) : 'transparent';
  subtitlePreview.style.textShadow = `${s.shadow}px ${s.shadow}px ${Math.max(2,s.shadow*2)}px #000, 0 0 ${s.outlineWidth}px #000`;

  const overlay = $('overlay');
  overlay.style.justifyContent = s.alignH==='left' ? 'flex-start' : s.alignH==='right' ? 'flex-end' : 'center';
  overlay.style.alignItems = s.alignV==='top' ? 'flex-start' : s.alignV==='middle' ? 'center' : 'flex-end';
  overlay.style.paddingLeft = overlay.style.paddingRight = '5%';
  overlay.style.paddingBottom = s.alignV === 'bottom' ? `${100-s.posY}%` : '0';
}

function hexToRgba(hex, op){
  const h = hex.replace('#','');
  const bigint = parseInt(h,16);
  const r=(bigint>>16)&255, g=(bigint>>8)&255, b=bigint&255;
  return `rgba(${r},${g},${b},${clamp(op,0,1)})`;
}

function updatePreviewByTime(t){
  const cue = state.cues.find(c => t >= c.start && t <= c.end);
  if (!cue){ subtitlePreview.textContent=''; subtitlePreview.style.opacity='0'; return; }
  subtitlePreview.style.opacity='1';

  let txt = cue.text;
  if (state.style.animation === 'highlight-word' && cue.words?.length){
    const word = cue.words.find(w => t >= w.start && t <= w.end);
    if (word){
      txt = cue.text.replace(word.text, `【${word.text}】`);
    }
  }
  subtitlePreview.textContent = txt;

  subtitlePreview.style.transform = 'scale(1)';
  subtitlePreview.style.transition = 'all .15s ease';
  if (state.style.animation === 'fade') subtitlePreview.style.opacity = '0.92';
  if (state.style.animation === 'pop') subtitlePreview.style.transform = 'scale(1.05)';
}

video.addEventListener('timeupdate', () => updatePreviewByTime(video.currentTime || 0));

drop.onclick = () => fileInput.click();
['dragover','dragenter'].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.add('drag');}));
['dragleave','drop'].forEach(ev => drop.addEventListener(ev, e=>{e.preventDefault();drop.classList.remove('drag');}));
drop.addEventListener('drop', e => {
  const f = e.dataTransfer.files?.[0];
  if (f) setFile(f);
});
fileInput.onchange = () => { if (fileInput.files?.[0]) setFile(fileInput.files[0]); };

function setFile(file){
  state.file = file;
  $('fileInfo').textContent = `${file.name} (${(file.size/1024/1024).toFixed(1)} MB)`;
  video.src = URL.createObjectURL(file);
  log('Файл выбран. Нажмите "Загрузить".');
}

$('uploadBtn').onclick = async () => {
  if (!state.file) return log('Сначала выберите видео.');
  const fd = new FormData(); fd.append('file', state.file);
  log('Загружаю...');
  const res = await fetch('/api/upload', { method:'POST', body:fd });
  const data = await res.json();
  if (!res.ok) return log('Ошибка upload: '+(data.error||res.status));
  state.fileId = data.file_id;
  log(`Загружено. ID: ${state.fileId}`);
};

$('transcribeBtn').onclick = async () => {
  if (!state.fileId) return log('Сначала загрузите видео.');
  log('Транскрибация...');
  const res = await fetch('/api/transcribe', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ file_id: state.fileId, language: $('lang').value || undefined })
  });
  const data = await res.json();
  if (!res.ok) return log('Ошибка transcribe: '+(data.error||res.status));
  state.words = data.words || [];
  state.segments = data.segments || [];
  state.cues = groupWordsToCues(state.words, Number($('maxWords').value||7), 3.2);
  updateCueList();
  updatePreviewStyle();
  log(`Готово: ${state.words.length} words, ${state.cues.length} cues.`);
};

$('styleBtn').onclick = () => {
  applyPreset();
  state.cues = state.cues.map(c => autoWrapCue(c, 24));
  updateCueList();
  updatePreviewStyle();
  log('Стиль обновлён.');
};

$('exportBtn').onclick = async () => {
  if (!state.fileId || !state.cues.length) return log('Нет данных для экспорта.');
  const mode = $('exportMode').value;
  if (mode === 'frontend') {
    return exportFrontend();
  }

  log('Экспорт backend ffmpeg...');
  const res = await fetch('/api/render', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({ file_id: state.fileId, cues: state.cues, style: styleFromUI() })
  });
  const data = await res.json();
  if (!res.ok) return log('Ошибка render: '+(data.error||res.status));
  state.renderedId = data.render_id;
  const a = document.createElement('a');
  a.href = `/api/download/${state.renderedId}`;
  a.download = 'subtitled.mp4';
  a.click();
  log('Экспорт завершён (backend).');
};

async function exportFrontend(){
  log('Экспорт frontend (MediaRecorder)...');
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  const v = video;
  canvas.width = v.videoWidth || 1280;
  canvas.height = v.videoHeight || 720;

  const stream = canvas.captureStream(30);
  const chunks = [];
  const rec = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
  rec.ondataavailable = e => { if (e.data.size) chunks.push(e.data); };
  rec.start();

  const oldTime = v.currentTime;
  v.currentTime = 0;
  await v.play();

  await new Promise(resolve => {
    function draw(){
      if (v.paused || v.ended){ resolve(); return; }
      ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
      const cue = state.cues.find(c => v.currentTime >= c.start && v.currentTime <= c.end);
      if (cue) {
        const s = styleFromUI();
        ctx.font = `${s.bold?'700':'400'} ${s.fontSize}px ${s.fontFamily}`;
        ctx.textAlign = 'center';
        ctx.fillStyle = s.color;
        ctx.strokeStyle = '#000';
        ctx.lineWidth = s.outlineWidth;
        const lines = cue.text.split('\n');
        const y0 = canvas.height * (s.posY/100);
        lines.forEach((ln, i) => {
          const y = y0 + i * s.fontSize * s.lineHeight;
          ctx.strokeText(ln, canvas.width/2, y);
          ctx.fillText(ln, canvas.width/2, y);
        });
      }
      requestAnimationFrame(draw);
    }
    draw();
  });

  rec.stop();
  await new Promise(r => rec.onstop = r);
  v.pause(); v.currentTime = oldTime;

  const blob = new Blob(chunks, { type:'video/webm' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'subtitled-frontend.webm'; a.click();
  log('Экспорт завершён (frontend).');
}

['font','fontSize','bold','italic','lineHeight','color','outlineWidth','shadow','bgOn','bgColor','bgOpacity','padding','radius','posX','posY','alignH','alignV','anim','preset','maxWords']
  .forEach(id => $(id).addEventListener('change', ()=>{ if (id==='preset') applyPreset(); updatePreviewStyle(); }));

applyPreset();
updatePreviewStyle();
</script>
</body>
</html>
"""
    return make_response(html)


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
