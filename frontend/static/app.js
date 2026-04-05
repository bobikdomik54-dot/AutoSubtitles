const state = { file: null, fileId: null, words: [], segments: [], cues: [], style: {}, preset: 'Classic' };
const $ = (id) => document.getElementById(id);

function log(t) { $('status').textContent = t; }
function clamp(n, a, b) { return Math.max(a, Math.min(b, n)); }
function sec(v) { return Number(v || 0); }
function fmt(t) { t = sec(t); const m = Math.floor(t / 60), s = (t % 60).toFixed(2).padStart(5, '0'); return `${m}:${s}`; }
function hexToRgba(hex, op) { const h = hex.replace('#', ''); const i = parseInt(h, 16); const r = (i >> 16) & 255, g = (i >> 8) & 255, b = i & 255; return `rgba(${r},${g},${b},${clamp(op, 0, 1)})`; }

function styleFromUI() {
  return {
    fontFamily: $('font').value,
    fontSize: Number($('fontSize').value || 61),
    bold: $('bold').value === '1',
    italic: $('italic').value === '1',
    lineHeight: Number($('lineHeight').value || 1.2),
    color: $('color').value,
    outlineWidth: Number($('outlineWidth').value || 3),
    shadow: Number($('shadow').value || 2),
    bgOn: $('bgOn').checked,
    bgColor: $('bgColor').value,
    bgOpacity: Number($('bgOpacity').value || 0.85),
    radius: Number($('radius').value || 8),
    padX: Number($('padX').value || 20),
    padY: Number($('padY').value || 10),
    posX: Number($('posX').value || 50),
    posY: Number($('posY').value || 70),
    animation: $('anim').value,
    maxWords: Number($('maxWords').value || 7),
    maxCueDur: Number($('maxCueDur').value || 3.2),
    subtitleMode: 'ass'
  };
}

function setPreset(name) {
  state.preset = name;
  const map = {
    'Classic': { font: 'Montserrat', size: 61, color: '#ffffff', outline: 3, shadow: 2, bg: '#000000', op: .85, anim: 'none' },
    'Highlight': { font: 'Bebas Neue', size: 66, color: '#ffe700', outline: 5, shadow: 0, bg: '#000000', op: .75, anim: 'highlight-word' },
    'Beast-like': { font: 'Bangers', size: 70, color: '#ffffff', outline: 6, shadow: 5, bg: '#000000', op: .9, anim: 'pop' },
    'Drop In': { font: 'Montserrat', size: 62, color: '#ffffff', outline: 4, shadow: 2, bg: '#1c1f2f', op: .82, anim: 'fade' },
    'Rotate': { font: 'Bangers', size: 64, color: '#f8f8ff', outline: 4, shadow: 4, bg: '#101010', op: .8, anim: 'pop' },
    'Minimal': { font: 'Playfair Display', size: 52, color: '#f5f5f5', outline: 1, shadow: 0, bg: '#000000', op: .55, anim: 'none' }
  };
  const p = map[name] || map['Classic'];
  $('font').value = p.font; $('fontSize').value = p.size; $('color').value = p.color; $('outlineWidth').value = p.outline; $('shadow').value = p.shadow;
  $('bgColor').value = p.bg; $('bgOpacity').value = p.op; $('anim').value = p.anim;
  document.querySelectorAll('.chip').forEach(c => c.classList.toggle('active', c.dataset.preset === name));
  applyStyle();
}

function groupWordsToCues(words, maxWords, maxDur) {
  const cues = []; let cur = []; let start = null;
  for (const w of words) {
    if (start === null) start = w.start;
    if (cur.length >= maxWords || (w.end - start) > maxDur) { cues.push(cueFromWords(cur)); cur = []; start = w.start; }
    cur.push(w);
  }
  if (cur.length) cues.push(cueFromWords(cur));
  return cues.map(c => autoWrapCue(c, 28));
}
function cueFromWords(arr) { return { text: arr.map(x => x.text).join(' ').trim(), start: arr[0]?.start || 0, end: arr[arr.length - 1]?.end || 0.4, words: arr }; }
function autoWrapCue(c, maxChars) {
  if (!$('autoWrap').checked) return c;
  const words = c.text.split(/\s+/).filter(Boolean); let lines = [''];
  for (const w of words) { let n = lines[lines.length - 1] ? lines[lines.length - 1] + ' ' + w : w; if (n.length > maxChars && lines[lines.length - 1]) lines.push(w); else lines[lines.length - 1] = n; }
  if (lines.length > 2) { const h = Math.ceil(words.length / 2); lines = [words.slice(0, h).join(' '), words.slice(h).join(' ')]; }
  c.text = lines.join('\n'); return c;
}

function renderCueList() {
  const root = $('cueList'); root.innerHTML = '';
  state.cues.forEach((c, i) => {
    const div = document.createElement('div'); div.className = 'cue';
    div.innerHTML = `<div class='row'><input data-i='${i}' data-k='start' type='number' step='0.01' value='${c.start.toFixed(2)}'><input data-i='${i}' data-k='end' type='number' step='0.01' value='${c.end.toFixed(2)}'></div><textarea data-i='${i}' data-k='text'>${c.text}</textarea><div class='cueBtns'><button class='btn' data-a='split' data-i='${i}'>Split</button><button class='btn' data-a='mergePrev' data-i='${i}'>Merge ←</button><button class='btn' data-a='mergeNext' data-i='${i}'>Merge →</button></div><div class='small'>${fmt(c.start)} - ${fmt(c.end)}</div>`;
    root.appendChild(div);
  });
  $('cueCount').textContent = `${state.cues.length} cues`;
  root.querySelectorAll('input[data-k],textarea[data-k]').forEach(el => el.onchange = () => { const i = +el.dataset.i, k = el.dataset.k; state.cues[i][k] = k === 'text' ? el.value : Number(el.value); if (k !== 'text' && state.cues[i].end <= state.cues[i].start) state.cues[i].end = state.cues[i].start + .2; drawPreview($('video').currentTime || 0); });
  root.querySelectorAll('button[data-a]').forEach(b => b.onclick = () => { const i = +b.dataset.i, a = b.dataset.a; if (a === 'split') splitCue(i); if (a === 'mergePrev' && i > 0) mergeCue(i - 1, i); if (a === 'mergeNext' && i < state.cues.length - 1) mergeCue(i, i + 1); renderCueList(); });
}
function splitCue(i) { const c = state.cues[i]; const p = c.text.replace(/\n/g, ' ').split(/\s+/).filter(Boolean); if (p.length < 2) return; const m = Math.floor(p.length / 2), mid = (c.start + c.end) / 2; state.cues.splice(i, 1, autoWrapCue({ text: p.slice(0, m).join(' '), start: c.start, end: mid, words: [] }, 28), autoWrapCue({ text: p.slice(m).join(' '), start: mid, end: c.end, words: [] }, 28)); }
function mergeCue(a, b) { const c1 = state.cues[a], c2 = state.cues[b]; state.cues.splice(a, 2, autoWrapCue({ text: (c1.text + ' ' + c2.text).replace(/\s+/g, ' ').trim(), start: Math.min(c1.start, c2.start), end: Math.max(c1.end, c2.end), words: [] }, 28)); }

function applyStyle() {
  state.style = styleFromUI(); const s = state.style; const v = $('subtitlePreview');
  v.style.fontFamily = s.fontFamily;
  v.style.fontSize = s.fontSize + 'px';
  v.style.fontWeight = s.bold ? '800' : '500';
  v.style.fontStyle = s.italic ? 'italic' : 'normal';
  v.style.lineHeight = s.lineHeight;
  v.style.color = s.color;
  v.style.background = s.bgOn ? hexToRgba(s.bgColor, s.bgOpacity) : 'transparent';
  v.style.borderRadius = s.radius + 'px';
  v.style.padding = (s.bgOn ? `${s.padY}px ${s.padX}px` : '0');
  v.style.textShadow = `${s.shadow}px ${s.shadow}px ${Math.max(2, s.shadow * 2)}px #000, 0 0 ${s.outlineWidth}px #000`;
  v.style.position = 'absolute'; v.style.left = `${s.posX}%`; v.style.top = `${s.posY}%`; v.style.transform = 'translate(-50%,-50%)';
  $('xLbl').textContent = `${s.posX}%`; $('yLbl').textContent = `${s.posY}%`;
}

function drawPreview(t) {
  const cue = state.cues.find(c => t >= c.start && t <= c.end); const v = $('subtitlePreview');
  if (!cue) { v.style.opacity = '0'; v.textContent = ''; return; }
  let txt = cue.text;
  if (state.style.animation === 'highlight-word' && cue.words?.length) { const w = cue.words.find(x => t >= x.start && t <= x.end); if (w) txt = cue.text.replace(w.text, `【${w.text}】`); }
  if (!$('punct').checked) txt = txt.replace(/[.,!?;:]/g, '');
  v.textContent = txt; v.style.opacity = '1';
  v.style.transition = 'all .15s ease'; v.style.scale = '1';
  if (state.style.animation === 'fade') v.style.opacity = '0.9';
  if (state.style.animation === 'pop') v.style.scale = '1.06';
}

function setFile(file) { state.file = file; $('fileInfo').textContent = `${file.name} (${(file.size / 1024 / 1024).toFixed(1)}MB)`; $('video').src = URL.createObjectURL(file); log('File selected. Upload to continue.'); }
$('drop').onclick = () => $('fileInput').click();
['dragenter', 'dragover'].forEach(e => $('drop').addEventListener(e, (ev) => { ev.preventDefault(); $('drop').classList.add('drag'); }));
['dragleave', 'drop'].forEach(e => $('drop').addEventListener(e, (ev) => { ev.preventDefault(); $('drop').classList.remove('drag'); }));
$('drop').addEventListener('drop', (e) => { const f = e.dataTransfer.files?.[0]; if (f) setFile(f); });
$('fileInput').onchange = () => { const f = $('fileInput').files?.[0]; if (f) setFile(f); };
$('newVideoBtn').onclick = () => $('fileInput').click();

$('uploadBtn').onclick = async () => {
  if (!state.file) return log('Choose file first');
  const fd = new FormData(); fd.append('file', state.file);
  log('Uploading...');
  const r = await fetch('/api/upload', { method: 'POST', body: fd }); const d = await r.json();
  if (!r.ok) return log('Upload error: ' + (d.error || r.status));
  state.fileId = d.file_id; log(`Uploaded ID: ${state.fileId}`);
};

$('transcribeBtn').onclick = async () => {
  if (!state.fileId) return log('Upload first');
  log('Transcribing...');
  const r = await fetch('/api/transcribe', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file_id: state.fileId, language: $('lang').value || undefined }) });
  const d = await r.json();
  if (!r.ok) return log('Transcribe error: ' + (d.error || r.status));
  state.words = d.words || []; state.segments = d.segments || [];
  const s = styleFromUI(); state.cues = groupWordsToCues(state.words, s.maxWords, s.maxCueDur);
  renderCueList(); applyStyle(); log(`Done: ${state.words.length} words / ${state.cues.length} cues`);
};

$('applyStyleBtn').onclick = () => { state.cues = state.cues.map(c => autoWrapCue(c, 28)); applyStyle(); renderCueList(); log('Style applied'); };

$('exportBtnTop').onclick = async () => {
  if (!state.fileId || !state.cues.length) return log('Nothing to export');
  if ($('exportMode').value === 'frontend') return exportFrontend();
  log('Backend render...');
  const r = await fetch('/api/render', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ file_id: state.fileId, cues: state.cues, style: styleFromUI() }) });
  const d = await r.json();
  if (!r.ok) return log('Render error: ' + (d.error || r.status));
  const a = document.createElement('a'); a.href = '/api/download/' + d.render_id; a.download = 'subtitled.mp4'; a.click();
  log('Export done');
};

async function exportFrontend() {
  log('Frontend export...');
  const video = $('video'); const canvas = document.createElement('canvas'); const ctx = canvas.getContext('2d');
  canvas.width = video.videoWidth || 1280; canvas.height = video.videoHeight || 720;
  const stream = canvas.captureStream(30); const chunks = [];
  const rec = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' }); rec.ondataavailable = e => e.data.size && chunks.push(e.data); rec.start();
  const old = video.currentTime; video.currentTime = 0; await video.play();
  await new Promise(res => {
    const draw = () => {
      if (video.paused || video.ended) return res();
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const cue = state.cues.find(c => video.currentTime >= c.start && video.currentTime <= c.end);
      if (cue) {
        const s = styleFromUI();
        ctx.font = `${s.bold ? '800' : '500'} ${s.fontSize}px ${s.fontFamily}`;
        ctx.textAlign = 'center'; ctx.fillStyle = s.color; ctx.strokeStyle = '#000'; ctx.lineWidth = s.outlineWidth;
        cue.text.split('\n').forEach((line, i) => {
          const y = (canvas.height * s.posY / 100) + i * (s.fontSize * s.lineHeight);
          ctx.strokeText(line, canvas.width * (s.posX / 100), y);
          ctx.fillText(line, canvas.width * (s.posX / 100), y);
        });
      }
      requestAnimationFrame(draw);
    };
    draw();
  });
  rec.stop(); await new Promise(r => rec.onstop = r); video.pause(); video.currentTime = old;
  const blob = new Blob(chunks, { type: 'video/webm' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'subtitled-frontend.webm'; a.click();
  log('Frontend export done');
}

$('video').addEventListener('timeupdate', () => drawPreview($('video').currentTime || 0));
['font', 'fontSize', 'bold', 'italic', 'lineHeight', 'color', 'outlineWidth', 'shadow', 'bgOn', 'bgColor', 'bgOpacity', 'radius', 'padX', 'padY', 'posX', 'posY', 'anim', 'maxWords', 'maxCueDur', 'autoWrap', 'punct']
  .forEach(id => $(id).addEventListener('input', applyStyle));
$('centerXBtn').onclick = () => { $('posX').value = 50; applyStyle(); };
$('centerYBtn').onclick = () => { $('posY').value = 70; applyStyle(); };
document.querySelectorAll('.chip').forEach(ch => ch.onclick = () => setPreset(ch.dataset.preset));

setPreset('Classic');
applyStyle();
