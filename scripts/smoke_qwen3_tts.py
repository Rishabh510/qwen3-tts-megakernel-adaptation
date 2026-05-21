import os
import time

import soundfile as sf
import torch

from qwen_tts import Qwen3TTSModel


MODEL_ID = os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
OUT_DIR = os.environ.get("OUT_DIR", "outputs")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    attn_impl = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    print(f"loading model={MODEL_ID} attn_implementation={attn_impl}")
    tts = Qwen3TTSModel.from_pretrained(
        MODEL_ID,
        device_map="cuda:0",
        dtype=torch.bfloat16,
        attn_implementation=attn_impl,
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    wavs, sr = tts.generate_custom_voice(
        text="The quick brown fox jumps over the lazy dog.",
        language="English",
        speaker="Ryan",
        max_new_tokens=256,
    )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    out_path = os.path.join(OUT_DIR, "qwen3_tts_smoke.wav")
    sf.write(out_path, wavs[0], sr)
    audio_seconds = len(wavs[0]) / float(sr)
    print(f"wrote={out_path}")
    print(f"sample_rate={sr}")
    print(f"audio_seconds={audio_seconds:.3f}")
    print(f"elapsed_seconds={elapsed:.3f}")
    print(f"rtf={elapsed / audio_seconds:.3f}")


if __name__ == "__main__":
    main()
