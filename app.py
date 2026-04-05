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
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700;900&family=Bebas+Neue&family=Bangers&family=Pacifico&family=Playfair+Display:wght@400;700&display=swap" rel="stylesheet">
<style>
:root{--bg:#0c0d11;--panel:#1a1b22;--panel2:#20222b;--line:#303343;--muted:#9ba2b7;--accent:#5b67ff}
*{box-sizing:border-box} html,body{margin:0;height:100%;font-family:Montserrat,sans-serif;background:var(--bg);color:#fff}
.app{height:100%;display:flex;flex-direction:column}
.topbar{height:56px;border-bottom:1px solid #232636;background:linear-gradient(180deg,#12131a,#101117);display:flex;align-items:center;justify-content:space-between;padding:0 14px}
.brand{display:flex;gap:10px;align-items:center;font-weight:700}.logo{width:18px;height:18px;border-radius:4px;background:#fff;color:#000;display:grid;place-items:center;font-size:12px}
.top-actions{display:flex;gap:8px;align-items:center}.btn{border:1px solid #303449;background:#222638;color:#fff;border-radius:10px;padding:9px 12px;font-weight:700;cursor:pointer}
.btn.primary{background:var(--accent);border-color:#6e78ff}.btn.ghost{background:#171923}
.workspace{flex:1;display:grid;grid-template-columns:1.45fr 0.95fr;min-height:0}
.left{border-right:1px solid #262938;display:flex;flex-direction:column;min-width:0}
.canvasTop{height:52px;display:flex;gap:8px;align-items:center;padding:8px 10px;border-bottom:1px solid #232636}
.select,input,textarea{background:#151821;border:1px solid #2b3041;color:#fff;border-radius:10px;padding:8px}
.playerWrap{position:relative;flex:1;display:flex;align-items:center;justify-content:center;background:#07080c}
video{max-width:100%;max-height:100%;width:100%;height:100%;object-fit:contain;background:#000}
#overlay{position:absolute;inset:0;display:flex;pointer-events:none;padding:6% 5%}
#subtitlePreview{max-width:100%;white-space:pre-wrap;text-align:center;transition:.15s ease;opacity:0}
.timeline{padding:10px;border-top:1px solid #232636;background:#0f1118}
.row{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
.right{padding:10px;overflow:auto}
.card{background:linear-gradient(180deg,var(--panel2),var(--panel));border:1px solid #2a2d3f;border-radius:14px;padding:12px;margin-bottom:12px}
.card h3{margin:0 0 10px 0;font-size:28px;font-weight:800;letter-spacing:.1px}
.small{font-size:12px;color:var(--muted)}
.chips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.chip{padding:10px;border:1px solid #353a50;border-radius:10px;text-align:center;background:#11131b;cursor:pointer;font-weight:800}
.chip.active{outline:2px solid #fff2}
.grid4{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px}
.grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px}
.grid2{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px}
label{display:block;font-size:12px;color:#c6cbe0;margin:3px 0}
input[type='range']{width:100%}
.toggle{display:flex;align-items:center;justify-content:space-between;padding:9px 0;border-top:1px solid #303447;margin-top:6px}
#cueList{max-height:38vh;overflow:auto}
.cue{border:1px solid #2c3146;border-radius:10px;padding:8px;background:#0f121a;margin-bottom:8px}
.cue textarea{width:100%;margin-top:6px;min-height:58px}
.cueBtns{display:flex;gap:6px;margin-top:6px}.cueBtns button{flex:1}
.dropzone{border:1px dashed #4d5578;border-radius:10px;padding:10px;text-align:center;cursor:pointer;background:#121522}
.dropzone.drag{background:#18203a}
#fileInput{display:none}
@media (max-width:1200px){.workspace{grid-template-columns:1fr}.right{max-height:45vh}}
</style>
</head>
<body>
<div class="app">
  <div class="topbar">
    <div class="brand"><div class="logo">A</div>AutoSubtitles</div>
    <div class="top-actions">
      <button id="newVideoBtn" class="btn primary">New Video</button>
      <button id="exportBtnTop" class="btn">Export Video</button>
      <select id="exportMode" class="select"><option value="backend">Mode A: ffmpeg</option><option value="frontend">Mode B: MediaRecorder</option></select>
    </div>
  </div>

  <div class="workspace">
    <div class="left">
      <div class="canvasTop row">
        <select id="sourceSelect" class="select"><option>Original</option></select>
        <button id="uploadBtn" class="btn ghost">Upload</button>
        <button id="transcribeBtn" class="btn primary">Transcribe</button>
        <button id="applyStyleBtn" class="btn">Apply Style</button>
        <input id="lang" class="select" placeholder="language: ru/en/auto" style="min-width:170px">
      </div>
      <div class="playerWrap">
        <video id="video" controls></video>
        <div id="overlay"><div id="subtitlePreview"></div></div>
      </div>
      <div class="timeline">
        <div class="row small"><span id="status">Ready</span><span id="fileInfo"></span><span id="cueCount"></span></div>
      </div>
    </div>

    <div class="right">
      <div class="card">
        <div id="drop" class="dropzone">Drop video here or click to choose</div>
        <input id="fileInput" type="file" accept="video/*">
      </div>

      <div class="card">
        <h3>Styles</h3>
        <div class="chips" id="presetChips">
          <div class="chip active" data-preset="Classic">Classic</div>
          <div class="chip" data-preset="Highlight">Highlight</div>
          <div class="chip" data-preset="Beast-like">Beast</div>
          <div class="chip" data-preset="Drop In">Drop In</div>
          <div class="chip" data-preset="Rotate">Rotate</div>
          <div class="chip" data-preset="Minimal">Minimal</div>
        </div>
      </div>

      <div class="card">
        <h3 style="font-size:24px">Typography</h3>
        <div class="grid4">
          <div><label>Size</label><input id="fontSize" type="number" value="61"></div>
          <div><label>Bold</label><select id="bold" class="select"><option value="1">On</option><option value="0">Off</option></select></div>
          <div><label>Italic</label><select id="italic" class="select"><option value="0">Off</option><option value="1">On</option></select></div>
          <div><label>Line</label><input id="lineHeight" type="number" step="0.1" value="1.2"></div>
        </div>
        <div><label>Font</label><select id="font" class="select"><option>Montserrat</option><option>Bebas Neue</option><option>Bangers</option><option>Komika Axis</option><option>Pacifico</option><option>Playfair Display</option></select></div>
        <div class="grid3">
          <div><label>Text</label><input id="color" type="color" value="#ffffff"></div>
          <div><label>Outline</label><input id="outlineWidth" type="number" value="3"></div>
          <div><label>Shadow</label><input id="shadow" type="number" value="2"></div>
        </div>
        <div class="toggle"><span>Show punctuation</span><input id="punct" type="checkbox" checked></div>
        <div class="toggle"><span>Auto wrap</span><input id="autoWrap" type="checkbox" checked></div>
      </div>

      <div class="card">
        <h3 style="font-size:24px">Position</h3>
        <label>Position X <span id="xLbl">50%</span></label><input id="posX" type="range" min="0" max="100" value="50">
        <label>Position Y <span id="yLbl">70%</span></label><input id="posY" type="range" min="0" max="100" value="70">
        <div class="grid2">
          <button class="btn" id="centerXBtn">Center X</button>
          <button class="btn" id="centerYBtn">Center Y</button>
        </div>
      </div>

      <div class="card">
        <h3 style="font-size:24px">Background + FX</h3>
        <div class="toggle"><span>Background</span><input id="bgOn" type="checkbox" checked></div>
        <div class="grid3">
          <div><label>Color</label><input id="bgColor" type="color" value="#000000"></div>
          <div><label>Opacity</label><input id="bgOpacity" type="number" step="0.05" value="0.85"></div>
          <div><label>Radius</label><input id="radius" type="number" value="8"></div>
        </div>
        <div class="grid2">
          <div><label>Pad X</label><input id="padX" type="number" value="20"></div>
          <div><label>Pad Y</label><input id="padY" type="number" value="10"></div>
        </div>
        <div><label>Animation</label><select id="anim" class="select"><option>none</option><option>fade</option><option>pop</option><option>highlight-word</option></select></div>
      </div>

      <div class="card">
        <h3 style="font-size:24px">Subtitles</h3>
        <div class="grid2">
          <div><label>Max words/cue</label><input id="maxWords" type="number" value="7"></div>
          <div><label>Max cue sec</label><input id="maxCueDur" type="number" step="0.1" value="3.2"></div>
        </div>
        <div id="cueList"></div>
      </div>
    </div>
  </div>
</div>
<script>
const state = {file:null,fileId:null,words:[],segments:[],cues:[],style:{},preset:'Classic'};
const $=(id)=>document.getElementById(id);

function log(t){$('status').textContent=t}
function clamp(n,a,b){return Math.max(a,Math.min(b,n));}
function sec(v){return Number(v||0)}
function fmt(t){t=sec(t);const m=Math.floor(t/60),s=(t%60).toFixed(2).padStart(5,'0');return `${m}:${s}`}
function hexToRgba(hex,op){const h=hex.replace('#','');const int=parseInt(h,16);const r=(int>>16)&255,g=(int>>8)&255,b=int&255;return `rgba(${r},${g},${b},${clamp(op,0,1)})`}

function styleFromUI(){
  return {
    fontFamily:$('font').value,
    fontSize:Number($('fontSize').value||61),
    bold:$('bold').value==='1',
    italic:$('italic').value==='1',
    lineHeight:Number($('lineHeight').value||1.2),
    color:$('color').value,
    outlineWidth:Number($('outlineWidth').value||3),
    shadow:Number($('shadow').value||2),
    bgOn:$('bgOn').checked,
    bgColor:$('bgColor').value,
    bgOpacity:Number($('bgOpacity').value||0.85),
    radius:Number($('radius').value||8),
    padX:Number($('padX').value||20),
    padY:Number($('padY').value||10),
    posX:Number($('posX').value||50),
    posY:Number($('posY').value||70),
    animation:$('anim').value,
    maxWords:Number($('maxWords').value||7),
    maxCueDur:Number($('maxCueDur').value||3.2),
    subtitleMode:'ass'
  }
}

function setPreset(name){
  state.preset=name;
  const map={
    'Classic':{font:'Montserrat',size:61,color:'#ffffff',outline:3,shadow:2,bg:'#000000',op:.85,anim:'none'},
    'Highlight':{font:'Bebas Neue',size:66,color:'#ffe700',outline:5,shadow:0,bg:'#000000',op:.75,anim:'highlight-word'},
    'Beast-like':{font:'Bangers',size:70,color:'#ffffff',outline:6,shadow:5,bg:'#000000',op:.9,anim:'pop'},
    'Drop In':{font:'Montserrat',size:62,color:'#ffffff',outline:4,shadow:2,bg:'#1c1f2f',op:.82,anim:'fade'},
    'Rotate':{font:'Bangers',size:64,color:'#f8f8ff',outline:4,shadow:4,bg:'#101010',op:.8,anim:'pop'},
    'Minimal':{font:'Playfair Display',size:52,color:'#f5f5f5',outline:1,shadow:0,bg:'#000000',op:.55,anim:'none'}
  };
  const p=map[name]||map['Classic'];
  $('font').value=p.font;$('fontSize').value=p.size;$('color').value=p.color;$('outlineWidth').value=p.outline;$('shadow').value=p.shadow;
  $('bgColor').value=p.bg;$('bgOpacity').value=p.op;$('anim').value=p.anim;
  document.querySelectorAll('.chip').forEach(c=>c.classList.toggle('active',c.dataset.preset===name));
  applyStyle();
}

function groupWordsToCues(words,maxWords,maxDur){
  const cues=[]; let cur=[]; let start=null;
  for(const w of words){
    if(start===null) start=w.start;
    if(cur.length>=maxWords || (w.end-start)>maxDur){ cues.push(cueFromWords(cur)); cur=[]; start=w.start; }
    cur.push(w);
  }
  if(cur.length) cues.push(cueFromWords(cur));
  return cues.map(c=>autoWrapCue(c,28));
}
function cueFromWords(arr){return {text:arr.map(x=>x.text).join(' ').trim(),start:arr[0]?.start||0,end:arr[arr.length-1]?.end||0.4,words:arr}}
function autoWrapCue(c,maxChars){
  if(!$('autoWrap').checked) return c;
  const words=c.text.split(/\s+/).filter(Boolean); let lines=[''];
  for(const w of words){let n=lines[lines.length-1]?lines[lines.length-1]+' '+w:w; if(n.length>maxChars && lines[lines.length-1]) lines.push(w); else lines[lines.length-1]=n;}
  if(lines.length>2){const h=Math.ceil(words.length/2);lines=[words.slice(0,h).join(' '),words.slice(h).join(' ')];}
  c.text=lines.join('\\n'); return c;
}

function renderCueList(){
  const root=$('cueList'); root.innerHTML='';
  state.cues.forEach((c,i)=>{
    const div=document.createElement('div'); div.className='cue';
    div.innerHTML=`<div class='row'><input data-i='${i}' data-k='start' type='number' step='0.01' value='${c.start.toFixed(2)}'><input data-i='${i}' data-k='end' type='number' step='0.01' value='${c.end.toFixed(2)}'></div><textarea data-i='${i}' data-k='text'>${c.text}</textarea><div class='cueBtns'><button class='btn' data-a='split' data-i='${i}'>Split</button><button class='btn' data-a='mergePrev' data-i='${i}'>Merge ←</button><button class='btn' data-a='mergeNext' data-i='${i}'>Merge →</button></div><div class='small'>${fmt(c.start)} - ${fmt(c.end)}</div>`;
    root.appendChild(div);
  });
  $('cueCount').textContent=`${state.cues.length} cues`;
  root.querySelectorAll('input[data-k],textarea[data-k]').forEach(el=>el.onchange=()=>{const i=+el.dataset.i,k=el.dataset.k;state.cues[i][k]=k==='text'?el.value:Number(el.value);if(k!=='text'&&state.cues[i].end<=state.cues[i].start)state.cues[i].end=state.cues[i].start+.2;drawPreview($('video').currentTime||0)});
  root.querySelectorAll('button[data-a]').forEach(b=>b.onclick=()=>{const i=+b.dataset.i,a=b.dataset.a;if(a==='split')splitCue(i);if(a==='mergePrev'&&i>0)mergeCue(i-1,i);if(a==='mergeNext'&&i<state.cues.length-1)mergeCue(i,i+1);renderCueList();});
}
function splitCue(i){const c=state.cues[i];const p=c.text.replace(/\\n/g,' ').split(/\s+/).filter(Boolean);if(p.length<2)return;const m=Math.floor(p.length/2),mid=(c.start+c.end)/2;state.cues.splice(i,1,autoWrapCue({text:p.slice(0,m).join(' '),start:c.start,end:mid,words:[]},28),autoWrapCue({text:p.slice(m).join(' '),start:mid,end:c.end,words:[]},28));}
function mergeCue(a,b){const c1=state.cues[a],c2=state.cues[b];state.cues.splice(a,2,autoWrapCue({text:(c1.text+' '+c2.text).replace(/\s+/g,' ').trim(),start:Math.min(c1.start,c2.start),end:Math.max(c1.end,c2.end),words:[]},28));}

function applyStyle(){
  state.style=styleFromUI(); const s=state.style; const v=$('subtitlePreview'); const ov=$('overlay');
  v.style.fontFamily=s.fontFamily;
  v.style.fontSize=s.fontSize+'px';
  v.style.fontWeight=s.bold?'800':'500';
  v.style.fontStyle=s.italic?'italic':'normal';
  v.style.lineHeight=s.lineHeight;
  v.style.color=s.color;
  v.style.background=s.bgOn?hexToRgba(s.bgColor,s.bgOpacity):'transparent';
  v.style.borderRadius=s.radius+'px';
  v.style.padding=(s.bgOn?`${s.padY}px ${s.padX}px`:'0');
  v.style.textShadow=`${s.shadow}px ${s.shadow}px ${Math.max(2,s.shadow*2)}px #000, 0 0 ${s.outlineWidth}px #000`;
  ov.style.alignItems='flex-start'; ov.style.justifyContent='flex-start';
  v.style.position='absolute'; v.style.left=`${s.posX}%`; v.style.top=`${s.posY}%`; v.style.transform='translate(-50%,-50%)';
  $('xLbl').textContent=`${s.posX}%`; $('yLbl').textContent=`${s.posY}%`;
}

function drawPreview(t){
  const cue=state.cues.find(c=>t>=c.start&&t<=c.end); const v=$('subtitlePreview');
  if(!cue){v.style.opacity='0';v.textContent='';return;}
  let txt=cue.text;
  if(state.style.animation==='highlight-word'&&cue.words?.length){const w=cue.words.find(x=>t>=x.start&&t<=x.end);if(w)txt=cue.text.replace(w.text,`【${w.text}】`)}
  if(!$('punct').checked) txt=txt.replace(/[.,!?;:]/g,'');
  v.textContent=txt; v.style.opacity='1';
  v.style.transition='all .15s ease'; v.style.scale='1';
  if(state.style.animation==='fade') v.style.opacity='0.9';
  if(state.style.animation==='pop') v.style.scale='1.06';
}

function setFile(file){state.file=file;$('fileInfo').textContent=`${file.name} (${(file.size/1024/1024).toFixed(1)}MB)`;$('video').src=URL.createObjectURL(file);log('File selected. Upload to continue.')}

$('drop').onclick=()=>$('fileInput').click();
['dragenter','dragover'].forEach(e=>$('drop').addEventListener(e,(ev)=>{ev.preventDefault();$('drop').classList.add('drag');}));
['dragleave','drop'].forEach(e=>$('drop').addEventListener(e,(ev)=>{ev.preventDefault();$('drop').classList.remove('drag');}));
$('drop').addEventListener('drop',(e)=>{const f=e.dataTransfer.files?.[0]; if(f) setFile(f)});
$('fileInput').onchange=()=>{const f=$('fileInput').files?.[0]; if(f) setFile(f)};
$('newVideoBtn').onclick=()=>$('fileInput').click();

$('uploadBtn').onclick=async()=>{if(!state.file) return log('Choose file first'); const fd=new FormData(); fd.append('file',state.file); log('Uploading...'); const r=await fetch('/api/upload',{method:'POST',body:fd}); const d=await r.json(); if(!r.ok) return log('Upload error: '+(d.error||r.status)); state.fileId=d.file_id; log(`Uploaded ID: ${state.fileId}`)};
$('transcribeBtn').onclick=async()=>{if(!state.fileId) return log('Upload first'); log('Transcribing...'); const r=await fetch('/api/transcribe',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_id:state.fileId,language:$('lang').value||undefined})}); const d=await r.json(); if(!r.ok) return log('Transcribe error: '+(d.error||r.status)); state.words=d.words||[]; state.segments=d.segments||[]; const s=styleFromUI(); state.cues=groupWordsToCues(state.words,s.maxWords,s.maxCueDur); renderCueList(); applyStyle(); log(`Done: ${state.words.length} words / ${state.cues.length} cues`)};
$('applyStyleBtn').onclick=()=>{state.cues=state.cues.map(c=>autoWrapCue(c,28)); applyStyle(); renderCueList(); log('Style applied')};
$('exportBtnTop').onclick=async()=>{if(!state.fileId||!state.cues.length) return log('Nothing to export'); if($('exportMode').value==='frontend') return exportFrontend(); log('Backend render...'); const r=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({file_id:state.fileId,cues:state.cues,style:styleFromUI()})}); const d=await r.json(); if(!r.ok) return log('Render error: '+(d.error||r.status)); const a=document.createElement('a'); a.href='/api/download/'+d.render_id; a.download='subtitled.mp4'; a.click(); log('Export done')};

async function exportFrontend(){
  log('Frontend export...'); const video=$('video'); const canvas=document.createElement('canvas'); const ctx=canvas.getContext('2d'); canvas.width=video.videoWidth||1280; canvas.height=video.videoHeight||720;
  const stream=canvas.captureStream(30); const chunks=[]; const rec=new MediaRecorder(stream,{mimeType:'video/webm;codecs=vp9'}); rec.ondataavailable=e=>e.data.size&&chunks.push(e.data); rec.start();
  const old=video.currentTime; video.currentTime=0; await video.play();
  await new Promise(res=>{const draw=()=>{if(video.paused||video.ended)return res();ctx.drawImage(video,0,0,canvas.width,canvas.height);const cue=state.cues.find(c=>video.currentTime>=c.start&&video.currentTime<=c.end);if(cue){const s=styleFromUI();ctx.font=`${s.bold?'800':'500'} ${s.fontSize}px ${s.fontFamily}`;ctx.textAlign='center';ctx.fillStyle=s.color;ctx.strokeStyle='#000';ctx.lineWidth=s.outlineWidth;cue.text.split('\n').forEach((line,i)=>{const y=(canvas.height*s.posY/100)+i*(s.fontSize*s.lineHeight);ctx.strokeText(line,canvas.width*(s.posX/100),y);ctx.fillText(line,canvas.width*(s.posX/100),y);});}requestAnimationFrame(draw)};draw();});
  rec.stop(); await new Promise(r=>rec.onstop=r); video.pause(); video.currentTime=old; const blob=new Blob(chunks,{type:'video/webm'}); const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='subtitled-frontend.webm';a.click(); log('Frontend export done');
}

$('video').addEventListener('timeupdate',()=>drawPreview($('video').currentTime||0));
['font','fontSize','bold','italic','lineHeight','color','outlineWidth','shadow','bgOn','bgColor','bgOpacity','radius','padX','padY','posX','posY','anim','maxWords','maxCueDur','autoWrap','punct'].forEach(id=>$(id).addEventListener('input',()=>{applyStyle();}));
$('centerXBtn').onclick=()=>{$('posX').value=50;applyStyle()}; $('centerYBtn').onclick=()=>{$('posY').value=70;applyStyle()};
document.querySelectorAll('.chip').forEach(ch=>ch.onclick=()=>setPreset(ch.dataset.preset));
setPreset('Classic'); applyStyle();
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
