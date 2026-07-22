import hashlib
import importlib.metadata
import os
import sys
from pathlib import Path

import mlx.core as mx
import platformdirs


class WanPromptEmbedStore:
    # Disk-backed EXACT cache for Wan UMT5 prompt embeds. A cache hit skips
    # loading the ~11 GB torch text encoder entirely. Keys are derived from
    # the tokenized inputs (which capture prompt text AND tokenizer config)
    # plus a fingerprint of the text-encoder weight files, so any change to
    # the prompt, tokenizer, sequence length, precision, or encoder snapshot
    # produces a different key. Values are a few MB of safetensors each.

    DEFAULT_MAX_ENTRIES = 64

    def __init__(self, *, enabled: bool = True, cache_dir: Path | None = None, max_entries: int = DEFAULT_MAX_ENTRIES):
        self.enabled = enabled
        self.cache_dir = cache_dir or self.default_cache_dir()
        self.max_entries = max(1, int(max_entries))

    @staticmethod
    def default_cache_dir() -> Path:
        return Path(platformdirs.user_cache_dir(appname="mflux")) / "prompt_embeds" / "wan"

    @staticmethod
    def compute_text_encoder_fingerprint(text_encoder_path: Path) -> str:
        # Weight-file names, sizes, and mtimes identify the encoder snapshot
        # without reading gigabytes. Revision changes move the snapshot path;
        # local edits change size or mtime. The torch/transformers versions
        # join the fingerprint because their CPU bf16 kernels define the
        # exact embed values — an upgrade must invalidate, not serve stale.
        # importlib.metadata keeps the no-torch-import-on-hit property.
        digest = hashlib.sha256()
        digest.update(str(text_encoder_path).encode("utf-8"))
        for package in ("torch", "transformers"):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "absent"
            digest.update(f"{package}={version};".encode("utf-8"))
        try:
            for entry in sorted(text_encoder_path.iterdir(), key=lambda p: p.name):
                if not entry.is_file():
                    continue
                stat = entry.stat()
                digest.update(f"{entry.name}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8"))
        except OSError:
            pass
        return digest.hexdigest()

    @staticmethod
    def compute_key(
        *,
        encoder_fingerprint: str,
        input_ids_bytes: bytes,
        attention_mask_bytes: bytes,
        max_sequence_length: int,
        precision: str,
    ) -> str:
        digest = hashlib.sha256()
        # Length-prefix every field so adjacent variable-length byte streams
        # cannot alias across field boundaries.
        for field in (
            encoder_fingerprint.encode("utf-8"),
            input_ids_bytes,
            attention_mask_bytes,
            f"{max_sequence_length}:{precision}".encode("utf-8"),
        ):
            digest.update(len(field).to_bytes(8, "little"))
            digest.update(field)
        return digest.hexdigest()

    def load(self, key: str) -> mx.array | None:
        if not self.enabled:
            return None
        path = self._entry_path(key)
        if not path.exists():
            return None
        try:
            embeds = mx.load(str(path))["embeds"]
            path.touch()  # LRU recency
            return embeds
        except Exception as exc:
            # Corrupt entries fail loud, then heal by re-encoding.
            print(f"WanPromptEmbedStore: dropping corrupt cache entry {path.name}: {exc}", file=sys.stderr)
            try:
                path.unlink()
            except OSError:
                pass
            return None

    def store(self, key: str, embeds: mx.array) -> None:
        if not self.enabled:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self._entry_path(key)
            # Write-then-rename keeps concurrent readers (parallel scene
            # processes) from ever seeing a half-written entry.
            temp_path = self.cache_dir / f".tmp-{os.getpid()}-{key}.safetensors"
            mx.save_safetensors(str(temp_path), {"embeds": embeds})
            os.replace(temp_path, path)
            self._prune()
        except Exception as exc:
            print(f"WanPromptEmbedStore: failed to persist prompt embeds: {exc}", file=sys.stderr)

    def _entry_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.safetensors"

    def _prune(self) -> None:
        entries = sorted(
            (p for p in self.cache_dir.glob("*.safetensors") if p.is_file()),
            key=lambda p: p.stat().st_mtime_ns,
        )
        while len(entries) > self.max_entries:
            oldest = entries.pop(0)
            try:
                oldest.unlink()
            except OSError:
                break
