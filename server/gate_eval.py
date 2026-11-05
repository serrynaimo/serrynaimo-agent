"""Offline check of the speaker gate against other people talking over you.

Mixes held-out speech of the enrolled user (gate_samples/ and any *_enroll.wav
recordings beside this script, none of which are in the voiceprint) with the voices to ignore
(calibration_samples/negative/) the ways people talk over each other, plus a
simulated across-the-room version of the user, and runs the bot's own speaker
gate on every mix. Nothing here changes the bot; rerun it after recording new
calibration samples.

Two conditions: "recorded" voices are the ones in the S-norm cohort (people
recorded on the calibration page); "unrecorded" ones are hidden from a cohort
built from the other half of the negatives, standing in for visitors.

usage: uv run gate_eval.py [--words N]
    --words N  instead transcribe N clips' mixes with Qwen3-ASR and count
               words (uses the GPU alongside a running bot for a minute or so)
"""

import argparse
import asyncio
import difflib
import glob
import os
import re
import time

import numpy as np
from loguru import logger

logger.remove()  # the gate logs every decision; keep the report readable

import services_local as sl  # noqa: E402

SR = sl.QWEN3_ASR_SAMPLE_RATE
HERE = os.path.dirname(os.path.abspath(__file__))
SHAPES = {
    "mine_alone": "you alone",
    "theirs_alone": "them alone",
    "them_then_me": "they speak, then you",
    "me_then_them": "you, then they reply",
    "quick_reply": "a quick 'yeah' after you",
    "interjection": "a quick 'yeah' mid-sentence",
    "overlap_0dB": "both at once",
    "overlap_-6dB": "both at once, they're quieter",
    "far_alone": "you across the room, alone",
    "far_them_then_me": "across the room: they, then you",
    "far_me_then_them": "across the room: you, then they",
    "far_interjection": "across the room: 'yeah' mid-sentence",
}


def threshold_from_env() -> float:
    with open(os.path.join(HERE, ".env")) as f:
        for line in f:
            if line.startswith("VOICE_MATCH_THRESHOLD="):
                return float(line.split("=", 1)[1].strip())
    return 0.5


def gate(cohort):
    """The real Qwen3ASRSTTService gate methods, without loading the ASR."""
    svc = sl.Qwen3ASRSTTService.__new__(sl.Qwen3ASRSTTService)
    svc._encoder, svc._enrolled, _ = sl._cached_voiceprint(os.path.join(HERE, "enroll_samples"))
    svc._cohort = cohort
    svc._match_threshold = threshold_from_env()
    svc._calibrate = False
    svc._span_scores = {}
    return svc


async def judge(svc, x, prescored: bool):
    """Gate one utterance; returns (kept sample ranges or None, end-of-turn ms).

    prescored: fill the window scores the live loop computes while the user
    is still talking (all but the last window), so the timing is what the
    reply actually waits for.
    """
    svc._span_scores = {}
    win, hop = svc._span_geometry()
    if prescored:
        for s in range(0, x.size - win - hop + 1, hop):
            svc._span_scores[s] = svc._window_score(x[s : s + win])
    t = time.perf_counter()
    whole = svc._speaker_score(x)
    kept = await svc._enrolled_spans(x, whole)
    return (None if kept is None else kept[0]), 1000 * (time.perf_counter() - t)


def kept_mask(x, runs):
    keep = np.zeros(len(x), bool)
    for a, b in runs or []:
        keep[a:b] = True
    return keep


# ---- audio ---------------------------------------------------------------


def load(path):
    return sl._load_wav_16k(path).astype(np.float32)


def frame_rms(x):
    fr = x[: len(x) // 320 * 320].reshape(-1, 320)
    return np.sqrt((fr**2).mean(1))


def active(x):
    """Sample mask of speech (20 ms frames well above the clip's quiet floor)."""
    r = frame_rms(x)
    m = np.repeat(r > 0.15 * np.percentile(r, 95), 320)
    return np.pad(m, (0, len(x) - len(m)))


def level(x, target=0.05):
    a = active(x)
    return x * (target / float(np.sqrt((x[a] ** 2).mean())))


def chunks(x, secs, n_max):
    """Up to n_max speech-dense, non-overlapping chunks of `secs`."""
    L = int(secs * SR)
    p95 = np.percentile(frame_rms(x), 95)
    out = []
    for s in range(0, len(x) - L + 1, L):
        r = frame_rms(x[s : s + L])
        if ((r > 0.006) | (r > 0.15 * p95)).mean() > 0.6:
            out.append(x[s : s + L])
        if len(out) >= n_max:
            break
    return out


def far(x, rng, rt60=0.5, drr_db=-3.0, snr_db=20.0):
    """You across the room: a living-room reverb tail (direct path 3 dB under
    the reflections) plus room noise. Levels are equalised afterwards, like the
    browser's auto-gain, so distance shows up as echo and noise, not volume."""
    from scipy.signal import fftconvolve

    t = np.arange(int(rt60 * SR)) / SR
    h = rng.normal(0, 1, t.size) * np.exp(-6.9 * t / rt60)
    h[: int(0.003 * SR)] = 0
    h /= np.sqrt((h**2).sum())
    h[0] += 10 ** (drr_db / 20)
    y = fftconvolve(x, h)[: len(x) + int(0.2 * SR)].astype(np.float32)
    rms = float(np.sqrt((y[active(y)] ** 2).mean()))
    return y + rng.normal(0, rms * 10 ** (-snr_db / 20), len(y)).astype(np.float32)


def mix(shape, M, O, o, rng):
    """(audio, your-speech mask, their-speech mask). O: 2.5 s of them, o: 1 s."""
    if shape.startswith("far_"):
        M, shape = far(M, rng), shape[4:].replace("alone", "mine_alone")
    M, O, o = level(M), level(O), level(o)
    z = lambda secs: np.zeros(int(secs * SR), np.float32)  # noqa: E731
    if shape.startswith("overlap"):
        gain = 1.0 if shape == "overlap_0dB" else 10 ** (-6 / 20)
        start, L = int(0.25 * len(M)), min(len(O), int(0.6 * len(M)))
        x = M.copy()
        x[start : start + L] += gain * O[:L]
        theirs = np.zeros(len(M), bool)
        theirs[start : start + L] = active(O[:L])
        mine = active(M)
    else:
        if shape == "interjection":
            e = frame_rms(M)
            lo, hi = int(0.3 * len(e)), int(0.7 * len(e))
            cut = (lo + int(np.argmin(e[lo:hi]))) * 320
        parts = {
            "mine_alone": [(M, 1)],
            "theirs_alone": [(O, 2)],
            "me_then_them": [(M, 1), (z(0.3), 0), (O, 2)],
            "them_then_me": [(O, 2), (z(0.3), 0), (M, 1)],
            "quick_reply": [(M, 1), (z(0.3), 0), (o, 2)],
            "interjection": (
                lambda: [(M[:cut], 1), (z(0.2), 0), (o, 2), (z(0.2), 0), (M[cut:], 1)]
            ),
        }[shape]
        parts = parts() if callable(parts) else parts
        x = np.concatenate([p for p, _ in parts])
        mine = np.concatenate([active(p) if w == 1 else np.zeros(len(p), bool) for p, w in parts])
        theirs = np.concatenate([active(p) if w == 2 else np.zeros(len(p), bool) for p, w in parts])
    x = x + rng.normal(0, 0.0005, len(x)).astype(np.float32)
    return x.astype(np.float32), mine, theirs


def mine_clips():
    clips = [load(p) for p in sorted(glob.glob(os.path.join(HERE, "gate_samples", "*.wav")))]
    for path in sorted(glob.glob(os.path.join(HERE, "*_enroll.wav"))):
        clips += chunks(load(path), 4.0, 8)
    return clips


def cohort_from(paths):
    embed = sl._cached_voiceprint(os.path.join(HERE, "enroll_samples"))[0]
    wins = []
    for p in paths:  # calibration.py's windows: 2 s, 1 s hop, no silence
        x = load(p)
        for i in range(0, max(1, len(x) - 2 * SR + 1), SR):
            w = x[i : i + 2 * SR]
            if len(w) >= SR and float(np.sqrt((w**2).mean())) > 0.008:
                wins.append(embed(w))
    return np.stack(wins)


# ---- run -----------------------------------------------------------------


async def run():
    rng = np.random.default_rng(7)
    negs = sorted(glob.glob(os.path.join(HERE, "calibration_samples", "negative", "*.wav")))
    longs = {p: chunks(load(p), 2.5, 6) for p in negs}
    shorts = {p: chunks(load(p), 1.0, 10) for p in negs}
    negs = [p for p in negs if longs[p] and shorts[p]]  # skip recordings that are mostly pauses
    half = len(negs) // 2
    mine = mine_clips()
    deployed = sl._cached_voiceprint(os.path.join(HERE, "enroll_samples"))[2]
    conditions = {
        "recorded voices": (gate(deployed), negs, 3),
        "unrecorded voices": (gate(cohort_from(negs[:half])), negs[half:], 2),
    }
    print(
        f"{len(mine)} clips of you, {len(negs)} recordings of others, threshold {threshold_from_env()}"
    )
    timings = {True: [], False: []}
    for cname, (svc, pool, per_clip) in conditions.items():
        stats = {
            s: [0, 0.0, 0.0, 0.0, 0.0, 0] for s in SHAPES
        }  # accepted, mine kept/total, theirs kept/total, n
        for i, M in enumerate(mine):
            for j in range(per_clip):
                p = pool[(i + 3 * j) % len(pool)]
                O, o = longs[p][(i + j) % len(longs[p])], shorts[p][(2 * i + j) % len(shorts[p])]
                for shape in SHAPES:
                    x, m, t = mix(shape, M, O, o, rng)
                    kept, ms = await judge(svc, x, prescored=True)
                    timings[True].append(ms)
                    if i % 4 == 0 and j == 0:
                        timings[False].append((await judge(svc, x, prescored=False))[1])
                    k = kept_mask(x, kept)
                    st = stats[shape]
                    st[0] += kept is not None
                    st[1] += (m & k).sum()
                    st[2] += m.sum()
                    st[3] += (t & k).sum()
                    st[4] += t.sum()
                    st[5] += 1
        print(f"\n{cname}:  accepted | your speech kept | their speech through")
        for shape, label in SHAPES.items():
            a, mk, mt, tk, tt, n = stats[shape]
            mine_pct = f"{100 * mk / mt:5.1f}%" if mt else "    -"
            their_pct = f"{100 * tk / tt:5.1f}%" if tt else "    -"
            print(f"  {label:38} {100 * a / n:5.1f}%  {mine_pct:>8}  {their_pct:>8}")
    for pre, label in (
        (True, "end of turn (windows scored while you talk)"),
        (False, "if nothing was pre-scored"),
    ):
        ms = np.array(timings[pre])
        print(
            f"gate time, {label}: median {np.median(ms):.0f} ms, p95 {np.percentile(ms, 95):.0f} ms"
        )


async def words(n):
    """Transcribe mixes with the real recognizer and count crossed-over words."""
    from mlx_audio.stt.generate import load_model

    model = load_model(os.getenv("QWEN3_ASR_MODEL", "mlx-community/Qwen3-ASR-1.7B-bf16"))

    def asr(a):
        if a is None or len(a) < SR // 4:
            return []
        text = getattr(model.generate(a, verbose=False, language="en"), "text", "") or ""
        return re.findall(r"[a-z0-9']+", text.lower())

    def aligned(a, b):
        sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
        return {i for blk in sm.get_matching_blocks() for i in range(blk.a, blk.a + blk.size)}

    rng = np.random.default_rng(11)
    negs = sorted(glob.glob(os.path.join(HERE, "calibration_samples", "negative", "*.wav")))
    negs = [p for p in negs if chunks(load(p), 2.5, 6) and chunks(load(p), 1.0, 10)]
    mine = mine_clips()
    svc = gate(sl._cached_voiceprint(os.path.join(HERE, "enroll_samples"))[2])
    shapes = [
        "them_then_me",
        "me_then_them",
        "quick_reply",
        "interjection",
        "overlap_0dB",
        "far_them_then_me",
    ]
    tot = {s: [0, 0, 0, 0] for s in shapes}  # their words through, their words, yours lost, yours
    for i, M in enumerate(mine[k] for k in np.linspace(0, len(mine) - 1, n).astype(int)):
        p = negs[(3 * i) % len(negs)]
        L, S = chunks(load(p), 2.5, 6), chunks(load(p), 1.0, 10)
        O, o = L[i % len(L)], S[i % len(S)]
        yours = asr(M)
        for shape in shapes:
            x, _, _ = mix(shape, M, O, o, rng)
            other = o if shape in ("quick_reply", "interjection") else O
            theirs = [w for w in asr(other) if w not in set(yours)]
            runs = (await judge(svc, x, prescored=False))[0]
            out = asr(None if runs is None else svc._stitch(x, runs))
            hit = aligned(out, yours)
            t = tot[shape]
            t[0] += sum(1 for j, w in enumerate(out) if j not in hit and w in set(theirs))
            t[1] += len(theirs)
            t[2] += len(yours) - len(aligned(yours, out))
            t[3] += len(yours)
    print(f"\nwords, {n} clips (recorded voices): their words in your transcript | your words lost")
    for shape in shapes:
        a, b, c, d = tot[shape]
        print(f"  {SHAPES[shape]:38} {a:3}/{b:<3}  {c:3}/{d:<3}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--words", type=int, default=0, metavar="N")
    args = ap.parse_args()
    asyncio.run(words(args.words) if args.words else run())
