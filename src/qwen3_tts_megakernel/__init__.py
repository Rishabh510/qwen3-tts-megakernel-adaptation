"""Qwen3-TTS megakernel adaptation package."""

__all__ = ["StreamingSynthesizer", "SynthesizerConfig"]


def __getattr__(name):
    if name in __all__:
        from .synthesizer import StreamingSynthesizer, SynthesizerConfig

        exports = {
            "StreamingSynthesizer": StreamingSynthesizer,
            "SynthesizerConfig": SynthesizerConfig,
        }
        return exports[name]
    raise AttributeError(name)
