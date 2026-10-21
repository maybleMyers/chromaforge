"""Compatibility shim for huggingface_hub >= 1.0.

huggingface_hub 1.x removed the deprecated ``HfFolder`` class (and other
legacy helpers) that gradio 4.40 still imports at module level, which breaks
``import gradio``. Re-attach a minimal implementation so gradio can import
and its optional HF OAuth login keeps working. Import this before gradio.
"""

import os

import huggingface_hub as _hf_hub


if not hasattr(_hf_hub, "HfFolder"):
    from pathlib import Path as _Path

    def _hf_home() -> _Path:
        home = os.environ.get("HF_HOME") or os.environ.get("HUGGINGFACE_HUB_CACHE")
        return _Path(home).expanduser() if home else _Path("~/.cache/huggingface").expanduser()

    class HfFolder:
        """Minimal replacement for the removed huggingface_hub.HfFolder."""

        @staticmethod
        def _token_path() -> _Path:
            return _hf_home() / "token"

        @classmethod
        def save_token(cls, token: str):
            path = cls._token_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(token, encoding="utf-8")

        @classmethod
        def get_token(cls):
            try:
                return cls._token_path().read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

        @classmethod
        def delete_token(cls):
            cls._token_path().unlink(missing_ok=True)

    _hf_hub.HfFolder = HfFolder
