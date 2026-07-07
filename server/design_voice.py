"""Design a voice once, so the bot can lock onto it via cloning.

VoiceDesign re-designs the voice from the text description on EVERY generation,
so using it live makes the voice drift between utterances. This script runs the
design step once: it generates N candidate clips from QWEN3_TTS_INSTRUCT (or
--instruct), you listen and pick one, then point the bot's Base model at it:

    uv run design_voice.py            # generate 3 candidates
    afplay voice_ref_1.wav            # listen to each
    # then in .env:
    #   QWEN3_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16
    #   QWEN3_TTS_REF_AUDIO=<absolute path to the winner>
    #   QWEN3_TTS_REF_TEXT=<REF_TEXT printed below>
"""

import argparse
import os
import wave

import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

# Rich, varied sentence: cloning quality tracks the phonetic coverage of the clip.
REF_TEXT = (
    "Hello! Lovely to meet you. I'm your assistant, and I must say, "
    "it's a rather brilliant day for a good conversation."
)

DESIGN_MODEL = "mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instruct", default=os.getenv("QWEN3_TTS_INSTRUCT"))
    parser.add_argument("--candidates", type=int, default=3)
    parser.add_argument("--prefix", default="voice_ref", help="output file prefix")
    parser.add_argument("--model", default=DESIGN_MODEL)
    args = parser.parse_args()

    if not args.instruct:
        raise SystemExit("No voice description: set QWEN3_TTS_INSTRUCT in .env or pass --instruct")

    print(f"Designing from: {args.instruct!r}\n")
    from mlx_audio.tts.utils import load_model

    model = load_model(args.model)
    sr = int(getattr(model, "sample_rate", 24000))

    for i in range(1, args.candidates + 1):
        chunks = []
        for r in model.generate(
            text=REF_TEXT, instruct=args.instruct, lang_code="auto", verbose=False
        ):
            audio = np.asarray(r.audio, dtype=np.float32).flatten()
            chunks.append(np.clip(audio, -1.0, 1.0))
        pcm = (np.concatenate(chunks) * 32767.0).astype("<i2").tobytes()
        path = os.path.abspath(f"{args.prefix}_{i}.wav")
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr)
            w.writeframes(pcm)
        print(f"candidate {i}: {path} ({len(pcm) / 2 / sr:.1f}s)  ->  afplay {path}")

    print("\nPick your favourite, then set in .env:")
    print("  QWEN3_TTS_MODEL=mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16")
    print("  QWEN3_TTS_REF_AUDIO=<path of the winner>")
    print(f"  QWEN3_TTS_REF_TEXT={REF_TEXT}")


if __name__ == "__main__":
    main()
