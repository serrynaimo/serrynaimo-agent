#
# Local, on-device (Apple Silicon / MLX) Pipecat services.
#
# Provides two custom Pipecat services built on top of the mlx-audio library so
# the whole voice pipeline can run locally on a Mac:
#
#   - Qwen3ASRSTTService : speech-to-text using Qwen3-ASR (mlx-audio)
#   - Qwen3TTSService    : text-to-speech using Qwen3-TTS (mlx-audio)
#
# Neither ships in Pipecat core, so they are implemented here against the
# mlx-audio model APIs. Both models are loaded once at construction time.
#

import asyncio
import base64
import io
import os
import re
import threading
import time
import wave
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from loguru import logger

from pipecat.frames.frames import (
    BotSpeakingFrame,
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    ErrorFrame,
    Frame,
    InterimTranscriptionFrame,
    InterruptionFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.turns.user_start import VADUserTurnStartStrategy
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.settings import STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transcriptions.language import Language
from pipecat.utils.text.base_text_filter import BaseTextFilter
from pipecat.utils.time import time_now_iso8601

# Qwen3-ASR runs on 16 kHz mono audio.
QWEN3_ASR_SAMPLE_RATE = 16000

# Largest available Qwen3 MLX checkpoints (1.7B, bf16 = highest precision).
DEFAULT_ASR_MODEL = "mlx-community/Qwen3-ASR-1.7B-bf16"
DEFAULT_TTS_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"


def set_thread_qos_user_interactive() -> bool:
    """Promote the calling thread to macOS USER_INTERACTIVE QoS.

    The scheduler then prefers it on performance cores under load, and Metal
    command queues created by this thread inherit the elevated QoS — which is
    what keeps speech synthesis smooth while LM Studio hammers the GPU.
    """
    try:
        import ctypes

        libc = ctypes.CDLL("/usr/lib/libSystem.dylib")
        QOS_CLASS_USER_INTERACTIVE = 0x21
        return libc.pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0) == 0
    except Exception:  # noqa: BLE001 — non-macOS or API change: best effort
        return False


class _MLXWorker:
    """A dedicated single-thread executor for one MLX model.

    MLX GPU streams are thread-local: arrays and streams created on one thread
    cannot be used from another ("There is no Stream(gpu, 1) in current thread").
    So each model is loaded AND run on the same dedicated thread. The thread
    runs at USER_INTERACTIVE QoS so speech stays smooth under system load.
    """

    def __init__(self, name: str):
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=name,
            initializer=set_thread_qos_user_interactive,
        )

    def run_sync(self, fn, *args):
        """Run fn on the MLX thread and block until done (for use in __init__)."""
        return self._executor.submit(fn, *args).result()

    def run_detached(self, fn, *args):
        """Queue fn on the MLX thread without waiting (e.g. background warmup)."""
        self._executor.submit(fn, *args)

    async def run(self, fn, *args):
        """Run fn on the MLX thread from async code."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    async def iterate(self, make_gen, stop: threading.Event) -> AsyncGenerator:
        """Bridge a blocking generator (run on the MLX thread) to an async iterator.

        mlx-audio generation is synchronous and GPU-bound; running it inline
        would block the event loop. Chunks are handed back over a queue. Setting
        `stop` (e.g. on barge-in) makes the worker abandon generation at the
        next chunk boundary so the thread frees up.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        sentinel = object()

        def worker():
            try:
                for item in make_gen():
                    if stop.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, item)
            except Exception as exc:  # noqa: BLE001 - surfaced to the caller below
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, sentinel)

        loop.run_in_executor(self._executor, worker)

        try:
            while True:
                item = await queue.get()
                if item is sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            # Consumer stopped early (interruption/barge-in): tell the worker.
            stop.set()


class ThinkTagFilter(BaseTextFilter):
    """Strips <think>...</think> spans from text before TTS.

    Reasoning is disabled at the source (LM Studio chat template), but if the
    model ever emits stray think tags into content they must not be spoken.
    Stateful: a block may span multiple filter calls.
    """

    START = "<think>"
    END = "</think>"

    def __init__(self):
        self._in_think = False

    async def filter(self, text: str) -> str:
        out = []
        remaining = text
        while remaining:
            if self._in_think:
                i = remaining.find(self.END)
                if i < 0:
                    break  # still inside a think block; drop everything
                remaining = remaining[i + len(self.END) :]
                self._in_think = False
            else:
                i = remaining.find(self.START)
                if i < 0:
                    out.append(remaining)
                    break
                out.append(remaining[:i])
                remaining = remaining[i + len(self.START) :]
                self._in_think = True
        return "".join(out)

    async def handle_interruption(self):
        self._in_think = False

    async def reset_interruption(self):
        self._in_think = False


class SpeakablePathFilter(BaseTextFilter):
    """Rewrites file paths and URLs into speakable short forms before TTS.

    The system prompt asks the model not to read paths aloud, but local
    models ignore that often enough that we enforce it mechanically. Only
    the spoken audio is affected — the transcript shown in the client keeps
    the full text.

    /Users/x/Desktop/report.pdf -> "report.pdf"; https://www.a.com/b?c -> "a.com".
    """

    # Two or more /-separated segments (or an absolute/home prefix): a path.
    _PATH = re.compile(r"(?:~/|/)?(?:[\w.\-]+/){2,}[\w.\-]+/?|(?:~|/)[\w.\-]+(?:/[\w.\-]+)+/?")
    _URL = re.compile(r"https?://(?:www\.)?([^/\s]+)\S*")

    async def filter(self, text: str) -> str:
        text = self._URL.sub(lambda m: m.group(1), text)
        return self._PATH.sub(
            lambda m: m.group(0).rstrip("/").rsplit("/", 1)[-1] or m.group(0), text
        )

    async def handle_interruption(self):
        pass

    async def reset_interruption(self):
        pass


class SpeakableSymbolFilter(BaseTextFilter):
    """Normalizes symbols the TTS mispronounces or stalls on.

    Qwen3-TTS can stop dead at em dashes, curly quotes, degree signs and
    similar typography. Everything is rewritten into plain speakable text;
    only the audio is affected, the on-screen transcript keeps the original.
    """

    _CURRENCY_WORDS = {"$": "dollars", "€": "euros", "£": "pounds"}
    _CURRENCY = re.compile(r"([$€£])\s?(\d[\d,.]*)")
    _RULES = [
        (re.compile(r"^#{1,6}\s+", re.M), ""),          # markdown headings
        (re.compile(r"\*{1,3}|_{2,3}|`+"), ""),         # emphasis / code marks
        (re.compile(r"[“”„«»\"]"), ""),                 # double quotes (before dashes)
        (re.compile(r"[‘’]"), "'"),
        (re.compile(r"\s*[—–―]\s*"), ", "),             # em/en dashes → pause
        (re.compile(r"\s+-\s+"), ", "),                 # spaced hyphen as dash
        (re.compile(r"…"), "... "),
        (re.compile(r"°\s*C\b"), " degrees Celsius"),
        (re.compile(r"°\s*F\b"), " degrees Fahrenheit"),
        (re.compile(r"°"), " degrees"),
        (re.compile(r"(?<=\d)\s?%"), " percent"),
        (re.compile(r"&"), " and "),
        (re.compile(r"±"), " plus or minus "),
        (re.compile(r"×"), " times "),
        (re.compile(r"(?<=\w)²"), " squared"),
        (re.compile(r"(?<=\w)³"), " cubed"),
        (re.compile(r"•"), ", "),
        # Emoji and pictographs: unspeakable — the TTS produces artifacts or
        # a whole utterance of nothing for an emoji-only "sentence".
        (re.compile(
            "["
            "\U0001f000-\U0001faff"   # emoji, symbols, pictographs
            "\U00002600-\U000027bf"   # misc symbols, dingbats
            "\U0001f1e6-\U0001f1ff"   # regional indicators (flags)
            "⬀-⯿←-⇿"  # arrows
            "︎️‍"      # variation selectors, ZWJ
            "]+"
        ), " "),
        (re.compile(r"\s{2,}"), " "),                   # tidy leftover spacing
    ]

    async def filter(self, text: str) -> str:
        text = self._CURRENCY.sub(
            lambda m: f"{m.group(2)} {self._CURRENCY_WORDS[m.group(1)]}", text
        )
        for pattern, repl in self._RULES:
            text = pattern.sub(repl, text)
        return text

    async def handle_interruption(self):
        pass

    async def reset_interruption(self):
        pass


def _float_to_pcm16(audio: np.ndarray) -> bytes:
    """Convert a float32 waveform in [-1, 1] to 16-bit little-endian PCM bytes."""
    audio = np.asarray(audio, dtype=np.float32).flatten()
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype("<i2").tobytes()


# Process-wide model cache. Pipecat constructs new service objects for every
# client session; without this, each reconnect would load another multi-GB
# copy of the models (the old copies stay pinned by their MLX threads), which
# stalls the new session until the WebRTC offer times out. Models and their
# dedicated MLX threads are created once and shared across sessions.
_MODEL_CACHE: dict = {}
_CACHE_LOCK = threading.Lock()


def _cached_model(kind: str, model_name: str, loader):
    """Return (worker, model, entry) for kind/model, loading once per process.

    The global lock only guards the cache dict; each entry has its own lock,
    so different models (ASR, TTS, voiceprint) can load in parallel.
    """
    key = (kind, model_name)
    with _CACHE_LOCK:
        entry = _MODEL_CACHE.get(key)
        if entry is None:
            entry = {"lock": threading.Lock(), "worker": None, "model": None, "warmed": set()}
            _MODEL_CACHE[key] = entry
    with entry["lock"]:
        if entry["worker"] is None:
            worker = _MLXWorker(f"mlx-{kind}")
            logger.info(f"Loading Qwen3-{kind.upper()} model: {model_name} (first run downloads weights)")
            entry["model"] = worker.run_sync(loader, model_name)
            entry["worker"] = worker
        else:
            logger.info(f"Reusing cached Qwen3-{kind.upper()} model: {model_name}")
    return entry["worker"], entry["model"], entry


def _load_wav_16k(path: str) -> np.ndarray:
    """Load a WAV as float32 mono 16 kHz (naive linear resample if needed)."""
    import wave as _wave

    with _wave.open(path, "rb") as w:
        sr = w.getframerate()
        s = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    if sr != QWEN3_ASR_SAMPLE_RATE:
        n = int(round(len(s) * QWEN3_ASR_SAMPLE_RATE / sr))
        s = np.interp(np.linspace(0, len(s), n, endpoint=False), np.arange(len(s)), s)
    return s.astype(np.float32)


def _tune_mlx_memory():
    """Cap MLX's buffer cache (global, idempotent).

    Unbounded cache growth over long sessions slowly degrades allocation and
    kernel-launch times — audible as TTS gapping out after a while.
    """
    try:
        import mlx.core as mx

        mx.set_cache_limit(2 * 1024**3)
    except Exception:  # noqa: BLE001 — best effort, API may move
        pass


def _load_tts_model(model_name: str):
    # Runs on the dedicated MLX thread (see _MLXWorker docstring).
    from mlx_audio.tts.utils import load_model

    _tune_mlx_memory()
    return load_model(model_name)


def preload_models(asr_model: str, tts_model: str, enroll_audio: str | None = None):
    """Load all local models concurrently (each on its own thread/worker).

    Called in a background thread at process boot: the web server comes up
    immediately while models load; a session starting early just blocks on
    the per-model cache locks until its models are ready.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Import the shared MLX stack once, single-threaded, before fanning out:
    # concurrent first-imports of the same modules deadlock Python's import
    # machinery (_ModuleLock).
    import mlx.core  # noqa: F401
    import mlx_lm.generate  # noqa: F401
    import mlx_audio.stt.generate  # noqa: F401
    import mlx_audio.tts.utils  # noqa: F401

    jobs = {
        "ASR": lambda: _cached_model("asr", asr_model, Qwen3ASRSTTService._load_model),
        "TTS": lambda: _cached_model("tts", tts_model, _load_tts_model),
    }
    if enroll_audio:
        jobs["voiceprint"] = lambda: _cached_voiceprint(enroll_audio)
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {ex.submit(fn): name for name, fn in jobs.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                fut.result()
                logger.info(f"Preload: {name} ready ({time.monotonic() - start:.1f}s)")
            except Exception as exc:  # noqa: BLE001 — services retry on first use
                logger.warning(f"Preload: {name} failed ({exc}); will retry on first session")
    logger.info(f"Preload complete in {time.monotonic() - start:.1f}s")


def _cached_voiceprint(enroll_audio: str):
    """Load the ECAPA speaker encoder and enrollment embedding once per process.

    ``enroll_audio`` may be a single WAV file or a directory of WAV utterances
    (e.g. gate_samples/); with a directory, the voiceprint is the average
    embedding across utterances, which is more robust.
    """
    # Include content mtimes in the key so re-running /calibration (which
    # rewrites the enrollment) takes effect on the next session, no restart.
    try:
        if os.path.isdir(enroll_audio):
            stamp = max((os.path.getmtime(os.path.join(enroll_audio, f))
                         for f in os.listdir(enroll_audio)), default=0)
        else:
            stamp = os.path.getmtime(enroll_audio)
    except OSError:
        stamp = 0
    key = ("voiceprint", enroll_audio, stamp)
    with _CACHE_LOCK:
        holder = _MODEL_CACHE.get(key)
        if holder is None:
            holder = {"lock": threading.Lock(), "entry": None}
            _MODEL_CACHE[key] = holder
    with holder["lock"]:
        entry = holder["entry"]
        if entry is None:
            import torch
            from speechbrain.inference.speaker import EncoderClassifier

            logger.info(f"Enrolling speaker voiceprint (ECAPA) from: {enroll_audio}")
            encoder = EncoderClassifier.from_hparams(
                "speechbrain/spkrec-ecapa-voxceleb", run_opts={"device": "cpu"}
            )

            def embed(samples: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    e = encoder.encode_batch(torch.tensor(samples)[None]).squeeze().numpy()
                return e / np.linalg.norm(e)

            if os.path.isdir(enroll_audio):
                paths = sorted(
                    p for p in (os.path.join(enroll_audio, f) for f in os.listdir(enroll_audio))
                    if p.endswith(".wav")
                )
                clips = [c for c in (_load_wav_16k(p) for p in paths)
                         if len(c) >= QWEN3_ASR_SAMPLE_RATE]  # skip clips under 1s
                if not clips:
                    raise ValueError(f"No usable WAV clips (>=1s) in {enroll_audio}")
                embeddings = [embed(c) for c in clips]
                logger.info(f"Voiceprint averaged from {len(embeddings)} utterances")
            else:
                embeddings = [embed(_load_wav_16k(enroll_audio))]
            voiceprint = np.mean(embeddings, axis=0)
            voiceprint /= np.linalg.norm(voiceprint)
            # Optional S-norm cohort (embeddings of voices to ignore), written
            # by the /calibration tool. Scoring against it cancels the shared
            # room/mic/channel component and separates far better than raw
            # cosine similarity.
            cohort = None
            cohort_path = (
                os.path.join(enroll_audio, "cohort.npy")
                if os.path.isdir(enroll_audio)
                else None
            )
            if cohort_path and os.path.isfile(cohort_path):
                cohort = np.load(cohort_path)
                logger.info(f"S-norm cohort loaded ({len(cohort)} embeddings)")
            entry = {"encoder": embed, "embedding": voiceprint, "cohort": cohort}
            holder["entry"] = entry
    return entry["encoder"], entry["embedding"], entry["cohort"]


# WAV-encoded filler clips, published by Qwen3TTSService.prime_phrases and
# served over HTTP for the client-side filler player.
_FILLER_WAVS: list[tuple[str, bytes]] = []


def filler_wavs() -> list[tuple[str, bytes]]:
    """Ordered (text, wav_bytes) filler clips; empty until priming finishes."""
    return list(_FILLER_WAVS)


def _norm_words(text: str) -> str:
    """Lowercased alphanumeric words, single-spaced — for echo comparison."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


class Qwen3ASRSTTService(SegmentedSTTService):
    """Speech-to-text using Qwen3-ASR via mlx-audio, fully local on Apple Silicon.

    Optional speaker gating: pass ``enroll_audio`` (a clip of the target
    speaker) and utterances whose voice doesn't match are silently dropped
    before transcription, so the bot only responds to that speaker.
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_ASR_MODEL,
        language: Language | None = Language.EN,
        context_prompt: str | None = None,
        enroll_audio: str | None = None,
        match_threshold: float = 0.5,
        calibrate: bool = False,
        wake_words: list[str] | None = None,
        wake_timeout_secs: float = 10.0,
        wake_giveup_secs: float = 20.0,
        wake_word_window: int = 10,
        interim_transcripts: bool = True,
        **kwargs,
    ):
        # Qwen3-ASR expects 16 kHz; force the segmented buffer to that rate.
        # STTSettings keeps pipecat's settings validation satisfied (we manage
        # model/language ourselves; None language = auto-detect).
        super().__init__(
            sample_rate=QWEN3_ASR_SAMPLE_RATE,
            settings=STTSettings(model=model, language=language),
            # Measured final-transcript latency after speech end: ~0.3-0.8s
            # (worst case ~1s when an interim pass occupies the worker).
            # Declaring it calibrates the turn-stop timing and silences the
            # per-utterance stop_secs warning.
            ttfs_p99_latency=1.0,
            **kwargs,
        )
        self._model_name = model
        self._language = language
        self._context_prompt = context_prompt
        # Normalized prompt for echo detection: on silence/unclear audio the
        # ASR sometimes "transcribes" its own biasing prompt verbatim.
        self._context_norm = _norm_words(context_prompt) if context_prompt else ""
        if context_prompt:
            logger.info(f"Qwen3-ASR context biasing: [{context_prompt[:120]}]")
        self._worker, self._model, _ = _cached_model("asr", model, self._load_model)
        self._encoder = self._enrolled = self._cohort = None
        self._match_threshold = match_threshold
        self._calibrate = calibrate
        if enroll_audio:
            if not os.path.exists(enroll_audio):
                raise FileNotFoundError(f"Speaker enrollment audio not found: {enroll_audio}")
            self._encoder, self._enrolled, self._cohort = _cached_voiceprint(enroll_audio)
            mode = "CALIBRATION (logging scores, dropping nothing)" if calibrate else f"threshold {match_threshold}"
            logger.info(f"Speaker gating enabled ({mode})")
        self._wake_words = [w.strip().lower() for w in (wake_words or []) if w.strip()]
        self._wake_timeout_secs = wake_timeout_secs
        # When the wake gate is closed, a single utterance that runs this long
        # without the wake word appearing is given up on: transcription of it
        # stops and nothing resumes until the next fresh turn. 0 disables it.
        self._wake_giveup_secs = wake_giveup_secs
        self._wake_word_window = wake_word_window
        # Monotonic time of the last activity that keeps the wake gate open:
        # bot speech, plus the user's own accepted speech (so continuing to
        # talk never lets the timeout close the gate mid-conversation).
        self._last_activity = float("-inf")
        # Explicit client mute: overrides ALL activity (bot speech, tools)
        # until the wake word is heard or the client unmutes.
        self._muted = False
        self._bot_speaking = False  # bot audio currently playing
        # Called when an utterance passes the speaker (and wake) gates —
        # i.e. the ENROLLED voice spoke. bot.py wires this to cancel
        # in-flight tool calls, which generic interruptions can't touch.
        self._on_verified_speech = None
        # Called (sync, fire-and-forget) when an utterance is DROPPED by the
        # speaker or wake gate. bot.py uses it to resume a pending tool
        # answer that the phantom turn blocked.
        self._on_dropped_speech = None
        # Async callable: stop ONLY the voice output (anyone may trigger it).
        self._on_voice_stop = None
        # Callable returning True while the agent has work in flight (tool
        # calls); keeps the wake window open during long silent work.
        self._agent_busy = None
        # Bot speech only opens the wake-gate window while the exchange is
        # voice-driven. A typed message flips this off, so the bot's spoken
        # reply to typed input does NOT unlock the gate for bystanders.
        self._voice_driven = True
        self._interim_task = None  # partial-transcription loop while user speaks
        self._interim_enabled = interim_transcripts
        # Gate state captured when the utterance STARTED: the wake-gate
        # decision must reflect when the user began speaking, not when the
        # transcript lands (talking + ASR latency would otherwise push an
        # in-window utterance past the deadline).
        self._utterance_started_active = False
        # Per-utterance: whether the wake word has been heard yet in this turn's
        # partials. Once true, the give-up timer is disarmed for the turn.
        self._utterance_wake_seen = False
        # LLM_AUDIO_INPUT=1 or omni: after ALL gates pass (speaker, wake,
        # mute), stash the utterance audio so the pipeline can hand it to an
        # audio-native LLM (e.g. Qwen3-Omni) instead of only the transcript.
        # ("omni" additionally makes the bot run Qwen3-Omni itself — bot.py.)
        self._llm_audio_input = (
            os.getenv("LLM_AUDIO_INPUT", "omni").strip().lower()
            in ("1", "true", "yes", "omni")
        )
        self._llm_audio: dict | None = None
        # Session-reset fence: utterances that STARTED before this stamp are
        # discarded wherever they surface (capture, interim, transcription),
        # so speech from the old session never leaks into a fresh one.
        self._utterance_started_at = float("-inf")
        self._abandoned_at = float("-inf")
        if self._wake_words:
            logger.info(
                f"Wake gate enabled: from {wake_timeout_secs:.0f}s after the bot last spoke, "
                f"utterances must contain one of {self._wake_words} "
                f"in the first {wake_word_window} words"
                + (
                    f"; a closed-gate turn with no wake word within "
                    f"{wake_giveup_secs:.0f}s is ignored until the next turn"
                    if wake_giveup_secs else ""
                )
            )
        logger.info("Qwen3-ASR model ready")

    @staticmethod
    def _load_model(model: str):
        # Runs on the dedicated MLX thread. mlx_lm's generation stream is
        # thread-local and was created at import time on the main thread;
        # re-create it here so generate_step works on this thread.
        import mlx.core as mx
        import mlx_lm.generate as mlx_lm_generate

        mlx_lm_generate.generation_stream = mx.new_thread_local_stream(mx.default_device())

        from mlx_audio.stt.generate import load_model

        _tune_mlx_memory()
        return load_model(model)

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction):
        # The output transport broadcasts Bot*SpeakingFrames upstream through
        # the whole pipeline, so the wake gate can track when the bot last
        # spoke without any extra wiring in bot.py. BotSpeakingFrame is a
        # periodic heartbeat while the bot talks, so the gate stays open
        # throughout a long answer and the countdown starts when it ends.
        if isinstance(
            frame, (BotSpeakingFrame, BotStartedSpeakingFrame, BotStoppedSpeakingFrame)
        ):
            if self._voice_driven:
                self._last_activity = time.monotonic()
            self._bot_speaking = not isinstance(frame, BotStoppedSpeakingFrame)
        await super().process_frame(frame, direction)

    def note_typed_message(self):
        """A typed message arrived: the bot's spoken reply must not open the
        voice window — the next voice utterance still needs the wake word."""
        self._voice_driven = False

    def note_proactive_speech(self):
        """The bot is about to speak on its OWN initiative (e.g. reading a
        notification), not in reply to the user. Its speech must NOT open the
        wake window — otherwise a proactive interjection would let bystanders
        (or ambient speech) address the agent wake-word-free for the timeout.
        Same mechanism as a typed message: mark the exchange not-voice-driven."""
        self._voice_driven = False

    def take_llm_audio(self) -> dict | None:
        """The gated utterance audio for the LLM ({text, b64}), consumed once."""
        audio, self._llm_audio = self._llm_audio, None
        return audio

    @staticmethod
    def _wav_b64(samples) -> str:
        """float32 mono @16k -> base64 WAV (16-bit PCM)."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(QWEN3_ASR_SAMPLE_RATE)
            w.writeframes(_float_to_pcm16(samples))
        return base64.b64encode(buf.getvalue()).decode()

    def abandon_utterances(self):
        """Discard any utterance in flight (session reset): everything that
        started speaking before this moment is dropped, whether it is still
        being captured, interim-transcribed, or already in the ASR."""
        self._abandoned_at = time.monotonic()

    def _utterance_abandoned(self, started_at: float) -> bool:
        return started_at < self._abandoned_at

    def bot_speaking(self) -> bool:
        """True while bot audio is actually playing (not while it thinks)."""
        return self._bot_speaking

    def set_verified_speech_hook(self, hook):
        """Async callable invoked for every utterance that passes the gates."""
        self._on_verified_speech = hook

    def set_dropped_speech_hook(self, hook):
        """Sync callable invoked when an utterance is dropped by a gate."""
        self._on_dropped_speech = hook

    def set_voice_stop_hook(self, hook):
        """Async callable that stops only the voice output."""
        self._on_voice_stop = hook

    def set_agent_busy_hook(self, hook):
        """Callable returning True while agent work (tool calls) is in flight."""
        self._agent_busy = hook

    def _notify_dropped(self):
        if self._on_dropped_speech is not None:
            try:
                self._on_dropped_speech()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Dropped-speech hook failed: {exc}")

    def last_activity(self) -> float:
        """Monotonic time of the last bot speech or accepted user speech."""
        return self._last_activity

    def require_wake_word(self):
        """Close the wake window now (client mute button): speech must carry
        the wake word again. Sticky — bot speech and running tools do NOT
        reopen the window; only the wake word or an explicit unmute does."""
        self._muted = True
        self._last_activity = float("-inf")

    def open_wake_window(self):
        """Open the wake window now (client unmute): speech is accepted
        without the wake word, with the usual idle timeout from here."""
        self._muted = False
        self._last_activity = time.monotonic()

    def conversation_active(self) -> bool:
        """True while the wake-gate window is open.

        The window runs from the last bot speech or accepted user speech —
        an ongoing exchange holds it open; only mutual silence closes it.
        Always True when no wake words are configured, so disabling the wake
        gate also restores unconditional VAD barge-in.
        """
        if not self._wake_words:
            return True
        # An explicit mute is sticky: neither bot speech nor in-flight tool
        # work reopens the window — only the wake word (or unmute) does.
        if self._muted:
            return False
        # Long silent work (tool calls in flight) counts as agent activity.
        if self._agent_busy is not None and self._agent_busy():
            return True
        return time.monotonic() - self._last_activity <= self._wake_timeout_secs

    async def _enrolled_prefix(self, samples: np.ndarray):
        """Trim an accepted utterance to the enrolled speaker's leading part.

        The utterance-level speaker gate scores the AVERAGE voice — if
        another person talks in the same VAD segment, the enrolled speaker's
        words can mask theirs and their words leak into the transcript.
        Sliding-window scoring finds the takeover point; audio after it is
        cut before transcription. Returns None if nothing usable remains.
        """
        sr = QWEN3_ASR_SAMPLE_RATE
        win, hop = int(1.6 * sr), int(0.8 * sr)
        if self._encoder is None or self._calibrate or samples.size <= win + hop:
            return samples
        threshold = self._match_threshold * 0.8  # short windows score lower
        end = samples.size
        for start in range(0, samples.size - win + 1, hop):
            chunk = samples[start:start + win]
            if float(np.sqrt(np.mean(chunk * chunk))) < 0.004:
                continue  # (near-)silence can't vote on speaker identity
            try:
                score = await asyncio.to_thread(self._speaker_score, chunk)
            except Exception:  # noqa: BLE001 — fail open
                return samples
            if score < threshold:
                end = start + hop
                break
        if end >= samples.size:
            return samples
        if end < int(1.0 * sr):
            logger.info("Speaker purity: segment taken over by another voice — dropped")
            return None
        logger.info(f"Speaker purity: trimmed mixed segment to first {end / sr:.1f}s")
        return samples[:end]

    def _has_wake_word(self, text: str) -> bool:
        words = re.findall(r"[\w']+", text.lower())
        # Anywhere in the first N words, or as the utterance's final word
        # ("...what do you think, Serry?").
        candidates = words[: self._wake_word_window] + words[-1:]
        # Substring match so "serry" also catches "serrynaimo" and "serry's".
        return any(wake in word for word in candidates for wake in self._wake_words)

    def _wake_giveup_armed(self) -> bool:
        """True while the current turn is a candidate for the give-up timer:
        wake gating is on with a give-up window, the turn began with the gate
        already closed (so it needs the wake word), and no wake word has been
        heard in it yet."""
        return bool(
            self._wake_words
            and self._wake_giveup_secs
            and not self._utterance_started_active
            and not self._utterance_wake_seen
        )

    def _mark_wake_seen(self):
        """A partial revealed the wake word mid-turn: open the window now so the
        rest of the turn (and its final transcript) is accepted, and disarm the
        give-up timer."""
        self._utterance_wake_seen = True
        self._muted = False
        self._last_activity = time.monotonic()
        logger.info("Wake gate: wake word heard (partial)")

    def _giveup_no_wake(self):
        """The wake word never came within the give-up window: abandon this turn
        everywhere it might still surface (partials + final), and stay quiet
        until the next turn starts after the natural pause. Reuses the session
        fence, which already drops in-flight utterances started before it."""
        logger.info(
            f"Wake gate: no wake word within {self._wake_giveup_secs:.0f}s — "
            "ignoring this turn and pausing transcription until the next one"
        )
        self._abandoned_at = time.monotonic()
        self._notify_dropped()

    # --- interim transcription: words appear while the user is speaking ----

    INTERIM_MIN_SECS = 1.0   # shortest partial worth transcribing
    INTERIM_INTERVAL = 0.9   # spacing between partial passes

    async def _handle_user_started_speaking(self, frame):
        await super()._handle_user_started_speaking(frame)
        self._utterance_started_active = self.conversation_active()
        self._utterance_started_at = time.monotonic()
        self._utterance_wake_seen = False
        # Speech that begins inside the window holds the window open, so a
        # user who keeps talking (with pauses) is never timed out mid-flow.
        if self._utterance_started_active:
            self._last_activity = time.monotonic()
        # Anyone speaking stops the VOICE instantly (voice only — generation
        # and tool work continue; the next sentence still gets spoken unless
        # the voice is stopped again).
        if self._bot_speaking and self._on_voice_stop is not None:
            try:
                await self._on_voice_stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Voice-stop hook failed: {exc}")
        # Interims flow even when the wake gate is closed — they are display
        # only: the turn strategies are configured to ignore interims, so a
        # gated utterance still can't interrupt or trigger the bot. If the
        # final transcript is dropped by the gate, the client fades the
        # pending text away.
        if self._interim_enabled and self._interim_task is None:
            self._interim_task = self.create_task(self._interim_loop())

    async def _handle_user_stopped_speaking(self, frame):
        task, self._interim_task = self._interim_task, None
        if task:
            await self.cancel_task(task)
        await super()._handle_user_stopped_speaking(frame)

    async def _interim_loop(self):
        """Transcribe the growing utterance buffer and push interim results.

        Qwen3-ASR has no native streaming, so this re-transcribes the
        accumulated segment every INTERIM_INTERVAL. Each partial passes the
        speaker gate before being shown. The final segmented pass still
        produces the authoritative TranscriptionFrame.
        """
        last_len = 0
        last_text = ""
        try:
            while self._user_speaking:
                await asyncio.sleep(self.INTERIM_INTERVAL)
                if not self._user_speaking:
                    return
                if self._utterance_abandoned(self._utterance_started_at):
                    return  # session reset — stop showing partials for it
                # Give-up timer: a closed-gate turn that has run this long with
                # no wake word is abandoned before we spend any more ASR on it.
                # Checked before the buffer/speaker work so even a non-matching
                # or near-silent monologue stops on time.
                if (
                    self._wake_giveup_armed()
                    and time.monotonic() - self._utterance_started_at >= self._wake_giveup_secs
                ):
                    self._giveup_no_wake()
                    return
                buf = bytes(self._audio_buffer)
                grew = len(buf) - last_len >= self.sample_rate  # >= 0.5s of new audio
                if len(buf) < int(self.sample_rate * 2 * self.INTERIM_MIN_SECS) or not grew:
                    continue
                last_len = len(buf)
                samples = np.frombuffer(buf, dtype="<i2").astype(np.float32) / 32768.0
                if self._encoder is not None and not self._calibrate:
                    try:
                        # Score only the RECENT audio: scoring the whole
                        # buffer lets the enrolled speaker's earlier words
                        # mask another voice taking over mid-utterance.
                        tail = samples[-int(2.0 * QWEN3_ASR_SAMPLE_RATE):]
                        score = await asyncio.to_thread(self._speaker_score, tail)
                        if score < self._match_threshold:
                            continue
                    except Exception:  # noqa: BLE001 — fail open like the final gate
                        pass
                try:
                    text = await self._worker.run(self._transcribe, samples)
                except Exception:  # noqa: BLE001 — partials are best-effort
                    continue
                # The wake word showing up in a partial opens the window mid-turn,
                # so a genuine long command that opens with the wake word is never
                # given up on (and its final transcript is accepted even if the
                # word later scrolls out of the wake-check window).
                if self._wake_giveup_armed() and text and self._has_wake_word(text):
                    self._mark_wake_seen()
                if self._user_speaking and text and text != last_text:
                    last_text = text
                    await self.push_frame(
                        InterimTranscriptionFrame(
                            text,
                            getattr(self, "_user_id", "") or "",
                            time_now_iso8601(),
                            self._language,
                        )
                    )
        except asyncio.CancelledError:
            pass

    def _log_gate_score(self, score: float, num_samples: int):
        """Append gate scores to a file for offline threshold calibration."""
        try:
            from datetime import datetime as _dt

            path = os.path.join(os.path.dirname(__file__), "gate_scores.log")
            with open(path, "a") as f:
                f.write(
                    f"{_dt.now().isoformat(timespec='seconds')} "
                    f"score={score:.3f} dur={num_samples / QWEN3_ASR_SAMPLE_RATE:.1f}s "
                    f"mode={'calibrate' if self._calibrate else 'gate'}\n"
                )
        except OSError:
            pass

    _capture_count = 0

    def _capture_utterance(self, samples: np.ndarray):
        """In calibrate mode, save utterance audio for pipeline re-enrollment."""
        try:
            import wave as _wave

            capture_dir = os.path.join(os.path.dirname(__file__), "gate_samples")
            os.makedirs(capture_dir, exist_ok=True)
            Qwen3ASRSTTService._capture_count += 1
            path = os.path.join(capture_dir, f"utt_{self._capture_count:03d}.wav")
            with _wave.open(path, "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(QWEN3_ASR_SAMPLE_RATE)
                w.writeframes((np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes())
        except OSError:
            pass

    def _speaker_score(self, samples: np.ndarray) -> float:
        """Similarity between this utterance and the enrolled voice.

        With a cohort (from /calibration) this is an S-normalized score
        (typical: enrolled speaker > 1.6, others < 1.4); without one it's the
        raw cosine similarity (same speaker ~0.3-0.6, others < 0.2).
        """
        embedding = self._encoder(samples)
        raw = float(np.dot(embedding, self._enrolled))
        if self._cohort is None:
            return raw
        cohort_scores = self._cohort @ embedding
        return float((raw - cohort_scores.mean()) / (cohort_scores.std() + 1e-6))

    def _transcribe(self, audio: np.ndarray) -> str:
        kwargs = {"verbose": False}
        if self._language is not None:
            kwargs["language"] = self._language.value.split("-")[0]
        if self._context_prompt:
            # Qwen3-ASR context biasing: expected vocabulary (names, jargon)
            # dramatically improves recognition of exactly those words.
            kwargs["system_prompt"] = self._context_prompt
        result = self._model.generate(audio, **kwargs)
        text = (getattr(result, "text", "") or "").strip()
        # On silence/unclear audio the ASR sometimes emits its biasing prompt
        # as the "transcript". A multi-word, in-order chunk of the prompt is
        # never real speech — drop it.
        if text and self._context_norm:
            t = _norm_words(text)
            if len(t.split()) >= 3 and t in self._context_norm:
                logger.info(f"ASR echoed its biasing prompt — dropped: [{text[:80]}]")
                return ""
        return text

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        # Session-reset fence: an utterance that began before the reset must
        # not surface in the fresh session.
        started_at = self._utterance_started_at
        if self._utterance_abandoned(started_at):
            logger.info("Session reset: dropped utterance captured before the reset")
            return
        # `audio` is a WAV container (16-bit mono @ self.sample_rate) built by
        # SegmentedSTTService. Decode it to a normalized float32 array.
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                frames = wav.readframes(wav.getnframes())
            samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        except Exception as exc:  # noqa: BLE001
            yield ErrorFrame(error=f"Qwen3-ASR audio decode error: {exc}")
            return

        if samples.size == 0:
            return

        # Tripwire: near-silent input means a mic/OS-level problem — the ASR
        # will confabulate text from whispers, which looks like model failure.
        rms = float(np.sqrt(np.mean(samples * samples)))
        if rms < 0.006:
            logger.warning(
                f"Utterance audio is nearly silent (rms={rms:.4f}) — check the "
                "microphone device and input volume; transcription will be unreliable"
            )

        # Speaker gate: only transcribe utterances from the enrolled voice.
        if self._encoder is not None:
            try:
                score = await asyncio.to_thread(self._speaker_score, samples)
            except Exception as exc:  # noqa: BLE001 — fail open, never lock the user out
                logger.warning(f"Speaker check failed (letting utterance through): {exc}")
            else:
                self._log_gate_score(score, samples.size)
                if self._calibrate:
                    self._capture_utterance(samples)
                    logger.info(f"Speaker gate CALIBRATION: score {score:.2f} (nothing dropped)")
                elif score < self._match_threshold:
                    logger.info(f"Speaker gate: dropped utterance (score {score:.2f} < {self._match_threshold})")
                    self._notify_dropped()
                    return
                else:
                    logger.info(f"Speaker gate: accepted utterance (score {score:.2f})")

        # The gate passed on average — now cut anything after another voice
        # takes over, so other people's words never enter the transcript.
        samples = await self._enrolled_prefix(samples)
        if samples is None:
            self._notify_dropped()
            return

        await self.start_ttfb_metrics()
        await self.start_processing_metrics()
        try:
            text = await self._worker.run(self._transcribe, samples)
        except Exception as exc:  # noqa: BLE001
            yield ErrorFrame(error=f"Qwen3-ASR transcription error: {exc}")
            return
        finally:
            await self.stop_ttfb_metrics()
            await self.stop_processing_metrics()

        if not text:
            return

        # Re-check the reset fence: a session reset during the speaker gate
        # or ASR run must still discard this transcript.
        if self._utterance_abandoned(started_at):
            logger.info(f"Session reset: dropped in-flight transcript [{text[:60]}]")
            return

        # Wake gate: after a lull, only utterances addressed to the bot by
        # name get through. Runs after the speaker gate, on the transcript —
        # but judged by when the utterance STARTED, so speech begun inside
        # the window is never discarded for taking a while to say or parse.
        if self._wake_words and not self._utterance_started_active and not self.conversation_active():
            if not self._has_wake_word(text):
                logger.info(f"Wake gate: ignored utterance without wake word: [{text[:80]}]")
                self._notify_dropped()
                return
            logger.info("Wake gate: wake word heard")
            self._muted = False  # the wake word lifts an explicit mute
        # Verified-speaker interruption: raw VAD never interrupts anything —
        # noise and other voices can neither stop speech nor cancel work.
        # An utterance that reaches this point passed the speaker (and wake)
        # gates, so it interrupts everything: speech (if the early barge-in
        # check didn't already), generation, and in-flight tool calls.
        if self._on_verified_speech is not None:
            try:
                # Enrolled speaker confirmed: cancel in-flight tool calls,
                # which are immune to generic interruptions.
                await self._on_verified_speech()
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Verified-speech hook failed: {exc}")
        await self.broadcast_frame(InterruptionFrame)
        # An accepted VOICE utterance keeps the conversation window open and
        # returns the exchange to voice-driven mode.
        self._voice_driven = True
        self._last_activity = time.monotonic()

        logger.debug(f"Qwen3-ASR transcription: [{text}]")
        if self._llm_audio_input:
            self._llm_audio = {"text": text, "b64": self._wav_b64(samples)}
        yield TranscriptionFrame(
            text,
            getattr(self, "_user_id", "") or "",
            time_now_iso8601(),
            self._language,
        )


class VoiceOnlyInterruptor(FrameProcessor):
    """Passthrough processor placed between the LLM and the TTS.

    ``stop_voice()`` interrupts ONLY the voice: the InterruptionFrame is
    pushed downstream from here, so synthesis and playback stop, while the
    LLM upstream never sees it — generation continues, its text still shows
    on screen, and the next sentences still get spoken unless the voice is
    stopped again.
    """

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)

    async def stop_voice(self):
        logger.info("Voice-only interruption: speech detected over bot audio")
        frame = InterruptionFrame()
        # Mark it: generation continues after this frame, so downstream
        # consumers (the verbatim-trail hook in bot.py) must not treat it
        # like a generation-cancelling interruption.
        frame.voice_only = True
        await self.push_frame(frame, FrameDirection.DOWNSTREAM)


class GatedInterruptionVADTurnStartStrategy(VADUserTurnStartStrategy):
    """VAD turn start with a dynamic barge-in gate.

    Speech always starts a user turn (so STT segmentation and aggregation
    behave exactly like stock pipecat), but it only interrupts when the gate
    callable says so — wired to "bot audio is playing", so raw VAD can cut
    off speech instantly but never cancels silent work (LLM generation, tool
    calls). For utterances during silent work, Qwen3ASRSTTService broadcasts
    the interruption itself once the speaker gate confirms the enrolled voice.
    """

    def __init__(self, gate_open, **kwargs):
        super().__init__(**kwargs)
        self._gate_open = gate_open

    async def trigger_user_turn_started(self):
        self._enable_interruptions = self._gate_open()
        await super().trigger_user_turn_started()


class Qwen3TTSService(TTSService):
    """Text-to-speech using Qwen3-TTS via mlx-audio, fully local on Apple Silicon.

    Model variants:
    - Base: voice cloning — pass ``ref_audio`` (short ~3-10s clip of the target
      voice) with ``ref_text`` (its exact transcript); the cloning prompt is
      cached after first use. Without a reference it speaks in a default timbre.
    - CustomVoice: named speakers via ``voice`` (serena, vivian, ryan, ...);
      ``instruct`` optionally adds emotion/style direction.
    - VoiceDesign: describe the voice in ``instruct`` (required), e.g.
      "A warm British woman in her forties with a refined RP accent".
    """

    def __init__(
        self,
        *,
        model: str = DEFAULT_TTS_MODEL,
        voice: str | None = None,
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        language: str = "auto",
        temperature: float = 0.9,
        top_k: int = 50,
        speed: float = 1.0,
        stream_interval: float = 2.5,
        cont_stream_interval: float | None = None,
        stream_context_size: int = 50,
        fade_in_ms: int = 40,
        **kwargs,
    ):
        if (ref_audio is None) != (ref_text is None):
            raise ValueError("Voice cloning requires both ref_audio and ref_text")
        if ref_audio is not None and not os.path.isfile(ref_audio):
            raise FileNotFoundError(f"Reference audio not found: {ref_audio}")
        self._worker, self._model, cache_entry = _cached_model("tts", model, _load_tts_model)
        # Model reports its own output sample rate (Qwen3-TTS 12Hz -> 24 kHz).
        native_sr = int(getattr(self._model, "sample_rate", 24000))
        # Let the caller override, but default the pipeline to the model's rate.
        kwargs.setdefault("sample_rate", native_sr)
        # TTSSettings keeps pipecat's settings validation satisfied (we manage
        # model/voice ourselves; language is handled via lang_code).
        super().__init__(
            settings=TTSSettings(model=model, voice=voice or "default", language=None),
            **kwargs,
        )
        self._native_sr = native_sr
        self._voice = voice
        self._instruct = instruct
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._language = language
        self._temperature = temperature
        self._top_k = top_k
        self._speed = speed
        self._stream_interval = stream_interval
        # Continuation sentences (audio already playing) get bigger chunks:
        # their TTFB is inaudible and the headroom prevents underruns.
        self._cont_stream_interval = max(cont_stream_interval or 0.0, stream_interval, 2.5)
        self._last_audio_ts = float("-inf")
        self._stream_context_size = stream_context_size
        self._fade_in_samples = int(native_sr * fade_in_ms / 1000)
        if ref_audio:
            logger.info(f"Qwen3-TTS voice cloning enabled from: {ref_audio}")

        # Warmup: synthesize a short utterance once so Metal kernels are
        # compiled and the voice-clone (ICL) prompt cache is populated before
        # the first real message — cold starts are audibly rougher. Done once
        # per (model, voice reference); reconnecting sessions skip it.
        warm_key = (voice, instruct, ref_audio, ref_text)
        # Phrase cache lives on the global model cache entry (keyed by voice
        # config), so reconnecting sessions reuse it instead of re-synthesizing.
        self._phrase_cache = cache_entry.setdefault("phrases", {}).setdefault(warm_key, {})
        if warm_key not in cache_entry["warmed"]:
            cache_entry["warmed"].add(warm_key)

            def _warmup():
                for _ in self._make_gen("Hello."):
                    pass
                logger.info("Qwen3-TTS warmup complete")

            # Detached: warmup runs on the TTS worker in the background; a
            # first utterance arriving early simply queues behind it instead
            # of the whole session start blocking on it.
            self._worker.run_detached(_warmup)
        logger.info(f"Qwen3-TTS model ready (sample_rate={native_sr}, warming in background)")

    def can_generate_metrics(self) -> bool:
        return True

    def prime_phrases(self, phrases: list[str], publish_filler_wavs: bool = True):
        """Pre-synthesize static phrases (fillers) and cache their PCM.

        The client fetches these as WAVs (via /filler/) and plays them
        entirely locally — they never enter the pipeline or the conversation.
        Runs detached on the TTS worker, after the warmup.

        With ``publish_filler_wavs=False`` the phrases are only cached for
        instant server-side playback (run_tts / TTSSpeakFrame) and do NOT
        join the client's filler rotation.
        """

        def _prime():
            for text in phrases:
                key = text.strip()
                if key in self._phrase_cache:
                    continue
                chunks = []
                for result in self._make_gen(text):
                    audio = getattr(result, "audio", None)
                    if audio is not None:
                        chunks.append(np.asarray(audio, dtype=np.float32).flatten())
                if not chunks:
                    continue
                samples = np.concatenate(chunks)
                # Trim the trailing near-silence the generator pads short
                # phrases with: it keeps "bot speaking" true for extra
                # seconds, and pipecat defers the next LLM round until the
                # bot stops speaking. Windowed RMS, not per-sample peaks —
                # the cloned voice's breath/noise floor defeats a peak check.
                win = int(0.02 * self._native_sr)
                n_win = len(samples) // win
                if n_win:
                    rms = np.sqrt(
                        (samples[: n_win * win].reshape(n_win, win) ** 2).mean(axis=1)
                    )
                    voiced = np.where(rms > 0.02)[0]
                    if voiced.size:
                        samples = samples[: (voiced[-1] + 1) * win + int(0.1 * self._native_sr)]
                if self._fade_in_samples and samples.size:
                    n = min(self._fade_in_samples, samples.size)
                    ramp = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
                    samples[:n] *= ramp
                self._phrase_cache[key] = _float_to_pcm16(samples)
            if not publish_filler_wavs:
                logger.info(f"Qwen3-TTS phrase cache primed ({len(self._phrase_cache)} phrases)")
                return
            # Publish WAV-encoded clips for the client-side filler player.
            global _FILLER_WAVS
            wavs = []
            for text in phrases:
                pcm = self._phrase_cache.get(text.strip())
                if not pcm:
                    continue
                buf = io.BytesIO()
                with wave.open(buf, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(self._native_sr)
                    w.writeframes(pcm)
                wavs.append((text, buf.getvalue()))
            _FILLER_WAVS = wavs
            logger.info(f"Qwen3-TTS phrase cache primed ({len(self._phrase_cache)} phrases)")

        self._worker.run_detached(_prime)

    def _make_gen(self, text: str, interval: float | None = None):
        return self._model.generate(
            text=text,
            voice=self._voice,
            instruct=self._instruct,
            ref_audio=self._ref_audio,
            ref_text=self._ref_text,
            lang_code=self._language,
            temperature=self._temperature,
            top_k=self._top_k,
            speed=self._speed,
            stream=True,
            # Longer first-decode window + more decoder context: streamed
            # Qwen3-TTS is prone to a rough/unstable start before the (cloned)
            # voice locks in, and to chunk-boundary discontinuities.
            streaming_interval=interval or self._stream_interval,
            streaming_context_size=self._stream_context_size,
            verbose=False,
        )

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        if not text.strip():
            return  # e.g. an emoji-only sentence, emptied by the speech transforms
        cached = self._phrase_cache.get(text.strip())
        if cached is not None:
            logger.debug(f"Qwen3-TTS cached phrase: [{text}]")
            await self.start_tts_usage_metrics(text)
            yield TTSAudioRawFrame(cached, self._native_sr, 1, context_id=context_id)
            return

        # Adaptive chunking: only the FIRST sentence of a turn needs a small
        # first chunk (that's the audible latency). Follow-up sentences
        # synthesize while earlier audio is still playing, so their TTFB is
        # hidden — bigger chunks there mean real headroom against underruns
        # (generation runs barely above real time) and fewer chunk joins.
        continuation = (time.monotonic() - self._last_audio_ts) < 2.0
        interval = self._cont_stream_interval if continuation else self._stream_interval

        logger.debug(f"Qwen3-TTS generating ({interval:.1f}s chunks): [{text}]")
        await self.start_ttfb_metrics()
        await self.start_tts_usage_metrics(text)

        measuring_ttfb = True
        first_chunk = True
        stop = threading.Event()
        try:
            async for result in self._worker.iterate(lambda: self._make_gen(text, interval), stop):
                audio = getattr(result, "audio", None)
                if audio is None:
                    continue
                self._last_audio_ts = time.monotonic()
                samples = np.asarray(audio, dtype=np.float32).flatten()
                if first_chunk and samples.size and self._fade_in_samples:
                    # Short cosine ramp: removes the click/rough onset at the
                    # start of each utterance without touching actual speech.
                    n = min(self._fade_in_samples, samples.size)
                    ramp = (1 - np.cos(np.linspace(0, np.pi, n))) / 2
                    samples = samples.copy()
                    samples[:n] *= ramp
                pcm = _float_to_pcm16(samples)
                if not pcm:
                    continue
                first_chunk = False
                if measuring_ttfb:
                    await self.stop_ttfb_metrics()
                    measuring_ttfb = False
                yield TTSAudioRawFrame(pcm, self._native_sr, 1, context_id=context_id)
        except Exception as exc:  # noqa: BLE001
            yield ErrorFrame(error=f"Qwen3-TTS synthesis error: {exc}")
