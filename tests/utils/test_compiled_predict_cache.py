from types import SimpleNamespace

from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.utils.compiled_predict_cache import CompiledPredictCache


class TestCompiledPredictCache:
    def test_same_key_and_token_reuses_the_built_callable(self):
        cache = CompiledPredictCache()
        token = object()
        builds = []

        def build():
            fn = lambda: None  # noqa: E731
            builds.append(fn)
            return fn

        first = cache.get_or_build(key=("edit", False), weights_token=token, build=build)
        second = cache.get_or_build(key=("edit", False), weights_token=token, build=build)

        assert first is second
        assert len(builds) == 1

    def test_distinct_keys_get_distinct_entries(self):
        cache = CompiledPredictCache()
        token = object()
        cache.get_or_build(key=("edit", False), weights_token=token, build=lambda: object())
        cache.get_or_build(key=("edit", True), weights_token=token, build=lambda: object())

        assert len(cache) == 2

    def test_token_replacement_drops_every_entry(self):
        cache = CompiledPredictCache()
        stale = cache.get_or_build(key="k", weights_token=object(), build=lambda: object())
        fresh = cache.get_or_build(key="k", weights_token=object(), build=lambda: object())

        assert fresh is not stale
        assert len(cache) == 1

    def test_clear_forces_rebuild(self):
        cache = CompiledPredictCache()
        token = object()
        stale = cache.get_or_build(key="k", weights_token=token, build=lambda: object())
        cache.clear()
        fresh = cache.get_or_build(key="k", weights_token=token, build=lambda: object())

        assert fresh is not stale


class TestMemorySaverDropsCompiledPredicts:
    def test_delete_transformer_clears_compiled_cache_before_release(self):
        model = SimpleNamespace(
            tiling_config=None,
            transformer=object(),
            compiled_predict_cache=CompiledPredictCache(),
        )
        model.compiled_predict_cache.get_or_build(
            key="k",
            weights_token=model.transformer,
            build=lambda: object(),
        )
        saver = MemorySaver(model=model, keep_transformer=False)

        saver._delete_transformer()

        assert model.transformer is None
        assert len(model.compiled_predict_cache) == 0

    def test_delete_transformer_tolerates_models_without_the_cache(self):
        model = SimpleNamespace(tiling_config=None, transformer=object())
        saver = MemorySaver(model=model, keep_transformer=False)

        saver._delete_transformer()

        assert model.transformer is None
