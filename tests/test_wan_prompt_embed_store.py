import mlx.core as mx
import numpy as np

from mflux.models.wan.prompt_embed_store import WanPromptEmbedStore


def _store(tmp_path, **kwargs):
    return WanPromptEmbedStore(cache_dir=tmp_path / "cache", **kwargs)


def _key(prompt: str = "a cat", fingerprint: str = "abc") -> str:
    ids = np.arange(8, dtype=np.int64).reshape(1, 8)
    mask = np.ones((1, 8), dtype=np.int64)
    return WanPromptEmbedStore.compute_key(
        encoder_fingerprint=fingerprint,
        input_ids_bytes=str(ids.shape).encode() + ids.tobytes() + prompt.encode(),
        attention_mask_bytes=str(mask.shape).encode() + mask.tobytes(),
        max_sequence_length=512,
        precision="bfloat16",
    )


def test_round_trip_is_exact(tmp_path):
    store = _store(tmp_path)
    embeds = mx.array(np.random.default_rng(0).standard_normal((1, 8, 16)).astype(np.float32)).astype(mx.bfloat16)
    key = _key()

    assert store.load(key) is None
    store.store(key, embeds)
    loaded = store.load(key)

    assert loaded is not None
    assert loaded.dtype == mx.bfloat16
    assert bool(mx.array_equal(loaded, embeds))


def test_disabled_store_is_inert(tmp_path):
    store = _store(tmp_path, enabled=False)
    key = _key()
    store.store(key, mx.zeros((1, 2, 2)))

    assert store.load(key) is None
    assert not (tmp_path / "cache").exists()


def test_corrupt_entry_is_dropped_and_reencoded(tmp_path, capsys):
    store = _store(tmp_path)
    key = _key()
    store.cache_dir.mkdir(parents=True, exist_ok=True)
    corrupt_path = store.cache_dir / f"{key}.safetensors"
    corrupt_path.write_bytes(b"not a safetensors file")

    assert store.load(key) is None
    # Loud failure, then healed: the corrupt file is gone.
    assert not corrupt_path.exists()
    assert "corrupt" in capsys.readouterr().err


def test_prune_keeps_only_max_entries(tmp_path):
    store = _store(tmp_path, max_entries=3)
    for index in range(5):
        store.store(_key(prompt=f"prompt {index}"), mx.zeros((1, 2, 2)))

    remaining = list(store.cache_dir.glob("*.safetensors"))
    assert len(remaining) == 3
    # The newest entries survive.
    assert store.load(_key(prompt="prompt 4")) is not None


def test_different_keys_do_not_collide(tmp_path):
    store = _store(tmp_path)
    a = mx.ones((1, 2, 2))
    b = mx.zeros((1, 2, 2))
    store.store(_key(prompt="a"), a)
    store.store(_key(prompt="b"), b)

    assert bool(mx.array_equal(store.load(_key(prompt="a")), a))
    assert bool(mx.array_equal(store.load(_key(prompt="b")), b))


def test_store_failure_degrades_to_warning_not_raise(tmp_path, capsys):
    # A read-only cache dir must not break generation: store() warns to
    # stderr and the entry simply is not cached.
    import os

    store = _store(tmp_path)
    store.cache_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(store.cache_dir, 0o500)
    try:
        store.store(_key(), mx.zeros((1, 2, 2)))
    finally:
        os.chmod(store.cache_dir, 0o700)

    assert store.load(_key()) is None
    assert "failed to persist" in capsys.readouterr().err


def test_fingerprint_tracks_file_changes(tmp_path):
    encoder_dir = tmp_path / "text_encoder"
    encoder_dir.mkdir()
    (encoder_dir / "model.safetensors").write_bytes(b"weights-v1")
    first = WanPromptEmbedStore.compute_text_encoder_fingerprint(encoder_dir)

    (encoder_dir / "model.safetensors").write_bytes(b"weights-v2-longer")
    second = WanPromptEmbedStore.compute_text_encoder_fingerprint(encoder_dir)

    assert first != second
