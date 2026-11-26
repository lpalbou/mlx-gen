import gc

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten, tree_unflatten

from mflux.models.flux2.variants.edit.flux2_klein_edit import Flux2KleinEdit
from mflux.models.flux2.variants.txt2img.flux2_klein import Flux2Klein
from mflux.utils.apple_silicon import AppleSiliconUtil
from mflux.utils.compiled_predict_cache import CompiledPredictCache

WIDTH = 1024
LAYERS = 4
WEIGHT_BYTES = LAYERS * (WIDTH * WIDTH + WIDTH) * 4
TOKEN_COUNTS = (64, 256, 128)


class _TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(WIDTH, WIDTH) for _ in range(LAYERS)]

    def __call__(self, hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids, guidance=None):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


def _lazily_loaded_transformer(tmp_path) -> _TinyTransformer:
    # Loaded like WeightLoader loads a package: mx.load leaves every array unevaluated.
    path = tmp_path / "weights.safetensors"
    if not path.exists():
        mx.save_safetensors(str(path), dict(tree_flatten(_TinyTransformer().parameters())))
    transformer = _TinyTransformer()
    transformer.update(tree_unflatten(list(mx.load(str(path)).items())))
    return transformer


def _settled_active_bytes() -> int:
    gc.collect()
    mx.clear_cache()
    return mx.get_active_memory()


def _txt2img_call(predict, tokens: int) -> mx.array:
    return predict(
        latents=mx.ones((1, tokens, WIDTH)),
        latent_ids=mx.zeros((1, tokens, 4)),
        prompt_embeds=mx.ones((1, 8, WIDTH)),
        text_ids=mx.zeros((1, 8, 4)),
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance=1.0,
        timestep=mx.array([1.0]),
    )


def _edit_call(predict, tokens: int) -> mx.array:
    return predict(
        latents=mx.ones((1, tokens, WIDTH)),
        image_latents=mx.ones((1, tokens, WIDTH)),
        latent_ids=mx.zeros((1, tokens, 4)),
        image_latent_ids=mx.zeros((1, tokens, 4)),
        prompt_embeds=mx.ones((1, 8, WIDTH)),
        text_ids=mx.zeros((1, 8, 4)),
        negative_prompt_embeds=None,
        negative_text_ids=None,
        guidance=1.0,
        timestep=mx.array([1.0]),
    )


@pytest.mark.fast
@pytest.mark.parametrize(
    "build_predict, call",
    [(Flux2Klein._predict, _txt2img_call), (Flux2KleinEdit._predict, _edit_call)],
    ids=["txt2img", "edit"],
)
def test_compiled_predict_holds_one_weight_copy_across_input_shapes(monkeypatch, tmp_path, build_predict, call):
    monkeypatch.setattr(AppleSiliconUtil, "is_m1_or_m2", staticmethod(lambda: False))
    transformer = _lazily_loaded_transformer(tmp_path)
    baseline = _settled_active_bytes()

    predict = build_predict(transformer)
    for tokens in TOKEN_COUNTS:
        mx.eval(call(predict, tokens))

    # Before the fix every new input shape traced its own evaluated copy: 3 shapes held 3 copies.
    assert _settled_active_bytes() - baseline < 1.5 * WEIGHT_BYTES
    del predict
    assert _settled_active_bytes() - baseline > 0.9 * WEIGHT_BYTES, "the weights live in the module"


@pytest.mark.fast
def test_materialized_weights_give_the_outputs_of_the_lazy_trace(tmp_path):
    inputs = {tokens: mx.random.normal((1, tokens, WIDTH), key=mx.random.key(tokens)) for tokens in TOKEN_COUNTS}

    lazy = _lazily_loaded_transformer(tmp_path)
    lazy_predict = mx.compile(lambda x: lazy(x, None, None, None, None))
    lazy_outputs = {tokens: lazy_predict(x) for tokens, x in inputs.items()}
    mx.eval(lazy_outputs)

    materialized = _lazily_loaded_transformer(tmp_path)
    fixed_predict = CompiledPredictCache.compile_over_weights(
        lambda x: materialized(x, None, None, None, None), materialized
    )
    fixed_outputs = {tokens: fixed_predict(x) for tokens, x in inputs.items()}

    for tokens in TOKEN_COUNTS:
        assert mx.array_equal(lazy_outputs[tokens], fixed_outputs[tokens]).item()
