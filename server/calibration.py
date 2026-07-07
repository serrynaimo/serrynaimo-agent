"""Speaker-gate calibration tool, served at /calibration by the bot runner.

Record positive samples (the enrolled speaker) and negative samples (voices to
ignore: background video, other people). "Run calibration" then:

1. embeds 2s windows of all samples with the same ECAPA encoder the gate uses,
2. scores positives leave-one-out and negatives against the voiceprint,
3. picks a threshold in the gap (warns if the clusters overlap),
4. rewrites enroll_samples/ from the positive windows,
5. updates .env and the running process env, so a client reconnect picks it up.
"""

import glob
import os
import re
import threading

import numpy as np
from fastapi import Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

BASE = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(BASE, "calibration_samples")
ENROLL_DIR = os.path.join(BASE, "enroll_samples")
ENV_PATH = os.path.join(BASE, ".env")

SR = 16000
WIN = 2 * SR   # 2s analysis windows
HOP = SR       # 1s hop

_lock = threading.Lock()


def _wav_dir(kind: str) -> str:
    d = os.path.join(SAMPLES, kind)
    os.makedirs(d, exist_ok=True)
    return d


def _decode_wav(data: bytes) -> np.ndarray:
    """Decode an uploaded WAV to float32 mono 16k."""
    import io
    import wave

    with wave.open(io.BytesIO(data), "rb") as w:
        sr = w.getframerate()
        ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if ch > 1:
        raw = raw.reshape(-1, ch).mean(axis=1)
    if sr != SR:
        n = int(round(len(raw) * SR / sr))
        raw = np.interp(np.linspace(0, len(raw), n, endpoint=False), np.arange(len(raw)), raw)
    return raw.astype(np.float32)


def _save_wav(path: str, samples: np.ndarray):
    import wave

    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())


def _windows(samples: np.ndarray) -> list[np.ndarray]:
    """Split into 2s windows (1s hop); keep a >=1s tail; drop silence."""
    out = []
    for i in range(0, max(1, len(samples) - WIN + 1), HOP):
        w = samples[i : i + WIN]
        if len(w) >= SR:
            out.append(w)
    if not out and len(samples) >= SR:
        out.append(samples)
    return [w for w in out if float(np.sqrt((w**2).mean())) > 0.008]


def _counts() -> dict:
    return {
        kind: len(glob.glob(os.path.join(SAMPLES, kind, "*.wav")))
        for kind in ("positive", "negative")
    }


def _update_env(threshold: float):
    """Set the calibrated values in .env and the running process."""
    updates = {
        "VOICE_MATCH_THRESHOLD": f"{threshold:.2f}",
        "VOICE_ENROLL_AUDIO": ENROLL_DIR,
        "VOICE_GATE_CALIBRATE": "0",
    }
    try:
        text = open(ENV_PATH).read()
        for key, value in updates.items():
            if re.search(rf"^{key}=", text, re.M):
                text = re.sub(rf"^{key}=.*$", f"{key}={value}", text, flags=re.M)
            else:
                text += f"\n{key}={value}\n"
        open(ENV_PATH, "w").write(text)
    except OSError as exc:
        logger.warning(f"calibration: could not update .env: {exc}")
    os.environ.update(updates)


def _run_calibration() -> dict:
    import torch
    from speechbrain.inference.speaker import EncoderClassifier

    from services_local import _load_wav_16k

    pos_files = sorted(glob.glob(os.path.join(SAMPLES, "positive", "*.wav")))
    neg_files = sorted(glob.glob(os.path.join(SAMPLES, "negative", "*.wav")))
    if not pos_files:
        return {"ok": False, "message": "Record at least one positive sample first."}

    encoder = EncoderClassifier.from_hparams(
        "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"}
    )

    def embed(s: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            e = encoder.encode_batch(torch.tensor(s)[None]).squeeze().numpy()
        return e / np.linalg.norm(e)

    pos_windows: list[np.ndarray] = []
    pos_src: list[str] = []
    for p in pos_files:
        for w in _windows(_load_wav_16k(p)):
            pos_windows.append(w)
            pos_src.append(os.path.basename(p))
    if len(pos_windows) < 4:
        return {"ok": False, "message": "Not enough positive audio — record a few more seconds."}
    if not neg_files:
        return {"ok": False, "message": "Record at least one negative sample (the voices to ignore) — they are needed for score normalization."}
    pos_embs = [embed(w) for w in pos_windows]
    neg_embs = []
    neg_src: list[str] = []
    for p in neg_files:
        for w in _windows(_load_wav_16k(p)):
            neg_embs.append(embed(w))
            neg_src.append(os.path.basename(p))
    if len(neg_embs) < 8:
        return {"ok": False, "message": "Not enough negative audio — record 20-30s of the voices to ignore."}

    total = np.sum(pos_embs, axis=0)
    voiceprint = total / np.linalg.norm(total)

    # Honest evaluation: split negatives into a cohort half (used for
    # normalization) and a test half (scored like unseen impostors).
    cohort_eval = np.stack(neg_embs[0::2])
    neg_test = neg_embs[1::2]

    def snorm(e: np.ndarray, raw: float, cohort: np.ndarray) -> float:
        cs = cohort @ e
        return float((raw - cs.mean()) / (cs.std() + 1e-6))

    pos_scores = []
    for e in pos_embs:
        rest = total - e
        rest = rest / np.linalg.norm(rest)
        pos_scores.append(snorm(e, float(np.dot(e, rest)), cohort_eval))
    neg_scores = [snorm(e, float(np.dot(e, voiceprint)), cohort_eval) for e in neg_test]

    # --- cost-optimal threshold (robust to single outlier windows) ---------
    # A lone bad window used to flip the verdict to "overlap". Instead, sweep
    # all candidate thresholds and pick the one minimizing errors, weighting
    # a false ACCEPT (impostor passes) twice as heavily as a false REJECT.
    npos, nneg = len(pos_scores), len(neg_scores)
    pos_sorted = np.sort(pos_scores)
    neg_sorted = np.sort(neg_scores)
    candidates = sorted(set(np.round(pos_scores + neg_scores, 3)))
    best_cost, zero_band = float("inf"), []
    best_t = candidates[0]
    for t in candidates:
        fa = float(np.mean(neg_sorted >= t))
        fr = float(np.mean(pos_sorted < t))
        cost = 2 * fa + fr
        if cost < best_cost - 1e-12:
            best_cost, best_t = cost, t
            zero_band = [t]
        elif abs(cost - best_cost) <= 1e-12:
            zero_band.append(t)
    # With a clean gap many thresholds are equally optimal: center for margin.
    threshold = round((zero_band[0] + zero_band[-1]) / 2, 2) if zero_band else round(best_t, 2)
    fa_pct = round(float(np.mean(neg_sorted >= threshold)) * 100, 1)
    fr_pct = round(float(np.mean(pos_sorted < threshold)) * 100, 1)
    clean = fa_pct == 0.0 and fr_pct == 0.0

    pos_min, p5 = float(pos_sorted[0]), float(np.percentile(pos_scores, 5))
    neg_max, n95 = float(neg_sorted[-1]), float(np.percentile(neg_scores, 95))

    # Per-file diagnostics: which clips produce the boundary-crossing windows.
    def _file_stats(scores, sources, agg):
        by_file: dict[str, list[float]] = {}
        for s, f in zip(scores, sources):
            by_file.setdefault(f, []).append(s)
        return {f: round(agg(v), 2) for f, v in by_file.items()}

    neg_test_src = neg_src[1::2]
    neg_by_file = _file_stats(neg_scores, neg_test_src, max)
    pos_by_file = _file_stats(pos_scores, pos_src, min)
    offenders = {
        "negative": [{"file": f, "max": v} for f, v in
                     sorted(neg_by_file.items(), key=lambda kv: -kv[1]) if v >= threshold][:3],
        "positive": [{"file": f, "min": v} for f, v in
                     sorted(pos_by_file.items(), key=lambda kv: kv[1]) if v < threshold][:3],
    }

    # Score histograms for the page (normalized per class).
    lo = float(min(pos_min, float(neg_sorted[0]))) - 0.2
    hi = float(max(neg_max, float(pos_sorted[-1]))) + 0.2
    bins = 36
    pos_hist, _ = np.histogram(pos_scores, bins=bins, range=(lo, hi))
    neg_hist, _ = np.histogram(neg_scores, bins=bins, range=(lo, hi))
    def _norm(h):
        m = int(h.max()) or 1
        return [round(float(v) / m, 3) for v in h]

    # Rebuild enrollment from positive windows (cap 60) + FULL negative cohort
    # for runtime S-norm scoring.
    with _lock:
        os.makedirs(ENROLL_DIR, exist_ok=True)
        for old in glob.glob(os.path.join(ENROLL_DIR, "*")):
            os.remove(old)
        for i, w in enumerate(pos_windows[:60], start=1):
            _save_wav(os.path.join(ENROLL_DIR, f"cal_{i:03d}.wav"), w)
        np.save(os.path.join(ENROLL_DIR, "cohort.npy"), np.stack(neg_embs[:200]))
    _update_env(threshold)

    logger.info(
        f"calibration: threshold {threshold} (FA {fa_pct}% FR {fr_pct}%, pos min "
        f"{pos_min:.2f} p5 {p5:.2f}, neg max {neg_max:.2f} p95 {n95:.2f}, "
        f"{npos} pos / {nneg} neg windows)"
    )
    if clean:
        message = "Clean separation. Disconnect and reconnect the client to apply."
    elif fa_pct <= 2 and fr_pct <= 5:
        message = (f"Good enough: {fa_pct}% of ignored-voice windows would pass and "
                   f"{fr_pct}% of yours would be rejected. Check the flagged clips "
                   "below to improve further.")
    else:
        message = (f"Distributions overlap: {fa_pct}% of ignored-voice windows pass, "
                   f"{fr_pct}% of yours get rejected. Listen to the flagged clips — "
                   "a negative clip that contains YOUR voice (or a positive clip "
                   "with someone else's) is the usual culprit. Delete it and rerun.")
    return {
        "ok": True,
        "clean": clean,
        "overlap": not clean,
        "threshold": threshold,
        "fa_pct": fa_pct,
        "fr_pct": fr_pct,
        "positive": {"windows": npos, "min": round(pos_min, 2),
                     "p5": round(p5, 2), "mean": round(float(np.mean(pos_scores)), 2)},
        "negative": {"windows": nneg, "max": round(neg_max, 2),
                     "p95": round(n95, 2), "mean": round(float(np.mean(neg_scores)), 2)},
        "hist": {"lo": round(lo, 2), "hi": round(hi, 2), "bins": bins,
                 "pos": _norm(pos_hist), "neg": _norm(neg_hist)},
        "offenders": offenders,
        "message": message,
    }


PAGE = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Speaker Gate Calibration</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
 body { font-family: -apple-system, sans-serif; max-width: 640px; margin: 2rem auto; padding: 0 1rem; color: #222; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 1.6rem; }
 button { font-size: 1rem; padding: .55rem 1.1rem; border-radius: 8px; border: 1px solid #bbb; background: #f6f6f6; cursor: pointer; margin-right: .5rem; }
 button.rec { background: #d33; color: #fff; border-color: #b22; }
 button:disabled { opacity: .45; cursor: default; }
 .hint { color: #666; font-size: .9rem; }
 .row { margin: .7rem 0; }
 #result { white-space: pre-wrap; background: #f4f4f8; border-radius: 8px; padding: 1rem; margin-top: 1rem; font-family: ui-monospace, monospace; font-size: .85rem; display: none; }
 .warn { color: #b00; font-weight: 600; }
 .good { color: #187a3c; font-weight: 600; }
 .count { font-weight: 600; }
 #hist { display: none; width: 100%; height: 130px; margin-top: .8rem; background: #fbfbfd; border: 1px solid #e2e2ea; border-radius: 8px; }
 .offender { background: #fff6e8; border: 1px solid #eeda9c; border-radius: 6px; padding: .5rem .7rem; margin-top: .5rem; font-family: -apple-system, sans-serif; white-space: normal; }
 .offender a { color: #b00; font-weight: 600; }
</style></head><body>
<h1>Speaker Gate Calibration</h1>
<p class="hint">Step 1: record <b>your voice</b> (several clips: near, far, short, long).
Step 2: record the <b>voices to ignore</b> (play the video, let others talk).
Step 3: run calibration.</p>

<h2>1 · Your voice <span class="count" id="posCount">0</span> clips</h2>
<div class="row">
 <button id="posBtn">● Record my voice</button>
 <button id="posClear">Clear</button>
</div>

<h2>2 · Voices to ignore <span class="count" id="negCount">0</span> clips</h2>
<div class="row">
 <button id="negBtn">● Record negatives</button>
 <button id="negClear">Clear</button>
</div>

<h2>3 · Calibrate</h2>
<div class="row"><button id="runBtn">Run calibration</button></div>
<div id="result"></div>
<canvas id="hist"></canvas>

<script>
let ctx, stream, proc, buf = [], active = null;

async function startRec(kind) {
  stream = await navigator.mediaDevices.getUserMedia({audio: true});
  ctx = new AudioContext();
  const src = ctx.createMediaStreamSource(stream);
  proc = ctx.createScriptProcessor(4096, 1, 1);
  buf = [];
  proc.onaudioprocess = e => buf.push(new Float32Array(e.inputBuffer.getChannelData(0)));
  src.connect(proc); proc.connect(ctx.destination);
  active = kind;
}

async function stopRec() {
  proc.disconnect(); stream.getTracks().forEach(t => t.stop());
  const sr = ctx.sampleRate; await ctx.close();
  const n = buf.reduce((a, b) => a + b.length, 0);
  const pcm = new Float32Array(n);
  let o = 0; for (const b of buf) { pcm.set(b, o); o += b.length; }
  // WAV encode (16-bit PCM)
  const dv = new DataView(new ArrayBuffer(44 + n * 2));
  const ws = (off, s) => { for (let i = 0; i < s.length; i++) dv.setUint8(off + i, s.charCodeAt(i)); };
  ws(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); ws(8, "WAVEfmt ");
  dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true);
  dv.setUint16(34, 16, true); ws(36, "data"); dv.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) dv.setInt16(44 + i * 2, Math.max(-1, Math.min(1, pcm[i])) * 32767, true);
  const kind = active; active = null;
  await fetch(`/calibration/sample/${kind}`, {method: "POST", body: dv.buffer});
  refresh();
}

function wire(kind, btnId) {
  const btn = document.getElementById(btnId);
  btn.onclick = async () => {
    if (active === kind) { btn.classList.remove("rec"); btn.textContent = btn.textContent.replace("■ Stop", "● Record"); await stopRec(); }
    else if (!active) { await startRec(kind); btn.classList.add("rec"); btn.textContent = btn.textContent.replace("● Record", "■ Stop"); }
  };
}
wire("positive", "posBtn"); wire("negative", "negBtn");

document.getElementById("posClear").onclick = () => fetch("/calibration/samples/positive", {method: "DELETE"}).then(refresh);
document.getElementById("negClear").onclick = () => fetch("/calibration/samples/negative", {method: "DELETE"}).then(refresh);

function drawHist(d) {
  const c = document.getElementById("hist");
  c.style.display = "block";
  const dpr = window.devicePixelRatio || 1;
  const W = c.clientWidth, H = 130;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, W, H);
  const {lo, hi, bins, pos, neg} = d.hist;
  const bw = W / bins;
  for (let i = 0; i < bins; i++) {
    g.fillStyle = "rgba(200,60,60,0.55)";
    g.fillRect(i * bw, H - neg[i] * (H - 18), bw - 1, neg[i] * (H - 18));
    g.fillStyle = "rgba(50,150,80,0.55)";
    g.fillRect(i * bw, H - pos[i] * (H - 18), bw - 1, pos[i] * (H - 18));
  }
  const tx = ((d.threshold - lo) / (hi - lo)) * W;
  g.strokeStyle = "#222"; g.setLineDash([4, 3]);
  g.beginPath(); g.moveTo(tx, 0); g.lineTo(tx, H); g.stroke();
  g.setLineDash([]);
  g.fillStyle = "#222"; g.font = "11px ui-monospace";
  g.fillText(`threshold ${d.threshold}`, Math.min(tx + 5, W - 110), 12);
  g.fillStyle = "rgba(200,60,60,0.9)"; g.fillText("■ voices to ignore", 6, H - 4);
  g.fillStyle = "rgba(50,150,80,0.9)"; g.fillText("■ your voice", 140, H - 4);
}

function offenderHtml(d) {
  const rows = [];
  for (const o of d.offenders.negative) {
    rows.push(`<div class="offender">ignore-clip <b>${o.file}</b> scores up to ${o.max} — ` +
      `if YOUR voice is audible in it, ` +
      `<a href="#" data-kind="negative" data-file="${o.file}">delete it</a> and rerun.</div>`);
  }
  for (const o of d.offenders.positive) {
    rows.push(`<div class="offender">your clip <b>${o.file}</b> scores down to ${o.min} — ` +
      `if it contains someone else / silence / noise, ` +
      `<a href="#" data-kind="positive" data-file="${o.file}">delete it</a> and rerun.</div>`);
  }
  return rows.join("");
}

document.getElementById("runBtn").onclick = async () => {
  const el = document.getElementById("result");
  el.style.display = "block";
  el.textContent = "Calibrating (loading encoder, embedding windows)…";
  const r = await fetch("/calibration/run", {method: "POST"});
  const d = await r.json();
  if (!d.ok) { el.innerHTML = `<span class="warn">${d.message}</span>`; return; }
  const verdict = d.clean
    ? '<span class="good">clean separation</span>'
    : `<span class="warn">${d.fa_pct}% false accepts · ${d.fr_pct}% false rejects</span>`;
  el.innerHTML =
    `threshold: <b>${d.threshold}</b>  ${verdict}\n` +
    `your voice:   ${d.positive.windows} windows  min ${d.positive.min}  p5 ${d.positive.p5}  mean ${d.positive.mean}\n` +
    `to ignore:    ${d.negative.windows} windows  max ${d.negative.max}  p95 ${d.negative.p95}  mean ${d.negative.mean}\n\n` +
    (d.clean ? d.message : `<span class="warn">${d.message}</span>`) +
    (d.offenders.negative.length + d.offenders.positive.length ? `\n\n` + offenderHtml(d) : "");
  drawHist(d);
  el.querySelectorAll("a[data-file]").forEach((a) => {
    a.onclick = async (ev) => {
      ev.preventDefault();
      await fetch(`/calibration/sample/${a.dataset.kind}/${a.dataset.file}`, {method: "DELETE"});
      a.closest(".offender").innerHTML = `deleted ${a.dataset.file} — run calibration again.`;
      refresh();
    };
  });
};

async function refresh() {
  const d = await (await fetch("/calibration/status")).json();
  document.getElementById("posCount").textContent = d.positive;
  document.getElementById("negCount").textContent = d.negative;
}
refresh();
</script></body></html>"""


def register_calibration(app):
    """Attach the calibration routes to the runner's FastAPI app."""

    @app.get("/calibration")
    async def calibration_page():
        return HTMLResponse(PAGE)

    @app.get("/calibration/status")
    async def calibration_status():
        return JSONResponse(_counts())

    @app.post("/calibration/sample/{kind}")
    async def calibration_sample(kind: str, request: Request):
        if kind not in ("positive", "negative"):
            return JSONResponse({"error": "kind must be positive|negative"}, status_code=400)
        data = await request.body()
        try:
            samples = _decode_wav(data)
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": f"bad audio: {exc}"}, status_code=400)
        if len(samples) < SR:
            return JSONResponse({"error": "clip too short (need >=1s)"}, status_code=400)
        d = _wav_dir(kind)
        n = len(glob.glob(os.path.join(d, "*.wav"))) + 1
        _save_wav(os.path.join(d, f"s_{n:03d}.wav"), samples)
        logger.info(f"calibration: saved {kind} sample #{n} ({len(samples)/SR:.1f}s)")
        return JSONResponse(_counts())

    @app.delete("/calibration/samples/{kind}")
    async def calibration_clear(kind: str):
        for p in glob.glob(os.path.join(SAMPLES, kind, "*.wav")):
            os.remove(p)
        return JSONResponse(_counts())

    @app.delete("/calibration/sample/{kind}/{name}")
    async def calibration_delete_one(kind: str, name: str):
        if kind not in ("positive", "negative") or not re.fullmatch(r"[\w.-]+\.wav", name):
            return JSONResponse({"error": "bad sample reference"}, status_code=400)
        path = os.path.join(SAMPLES, kind, name)
        if os.path.isfile(path):
            os.remove(path)
            logger.info(f"calibration: deleted {kind} sample {name}")
        return JSONResponse(_counts())

    @app.post("/calibration/run")
    async def calibration_run():
        import asyncio

        result = await asyncio.to_thread(_run_calibration)
        return JSONResponse(result)
