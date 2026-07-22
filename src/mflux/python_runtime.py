from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mflux.callbacks import ProgressCallback, ProgressEvent
from mflux.cli.output_paths import normalize_output_template, resolve_output_path
from mflux.models.common.config import ModelConfig
from mflux.task_inference import GenerationPlan, TaskInferenceError, resolve_generation_plan


@dataclass(frozen=True)
class GeneratedOutput:
    seed: int
    task: str
    artifact: Any
    item_index: int
    item_count: int
    output_path: Path | None = None
    saved_path: Path | None = None


@dataclass(frozen=True)
class LoadedGenerationModel:
    plan: GenerationPlan
    model_config: ModelConfig
    runtime_id: str
    cache_key_base: str
    cache_key: str
    model: Any

    def generate_output(
        self,
        *,
        seed: int,
        output: str | Path | None = None,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **generate_kwargs: Any,
    ) -> GeneratedOutput:
        return self.generate_outputs(
            seeds=[seed],
            output=output,
            overwrite=overwrite,
            progress_callback=progress_callback,
            save_kwargs=save_kwargs,
            **generate_kwargs,
        )[0]

    def generate_outputs(
        self,
        *,
        seeds: Sequence[int],
        output: str | Path | None = None,
        overwrite: bool = False,
        progress_callback: ProgressCallback | None = None,
        save_kwargs: dict[str, Any] | None = None,
        **generate_kwargs: Any,
    ) -> list[GeneratedOutput]:
        return _RuntimeGenerationExecutor.generate_outputs(
            loaded=self,
            seeds=seeds,
            output=output,
            overwrite=overwrite,
            progress_callback=progress_callback,
            save_kwargs=save_kwargs,
            generate_kwargs=generate_kwargs,
        )


@dataclass(frozen=True)
class GenerationRuntimePlan:
    plan: GenerationPlan
    model_config: ModelConfig
    runtime_id: str
    cache_key_base: str
    _definition: "_RuntimeDefinition" = field(repr=False, compare=False)

    def cache_key(
        self,
        *,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> str:
        parts = {
            "base": self.cache_key_base,
            "quantize": quantize,
            "model_path": model_path,
            "lora_paths": lora_paths or [],
            "lora_scales": lora_scales or [],
            "lora_target_roles": lora_target_roles or [],
        }
        if model_kwargs:
            # Constructor extras change the loaded model's behavior, so hosts
            # that dedupe by cache_key must see them in the identity.
            parts["model_kwargs"] = {key: repr(value) for key, value in sorted(model_kwargs.items())}
        return json.dumps(parts, sort_keys=True, separators=(",", ":"))

    def load(
        self,
        *,
        quantize: int | None = None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        lora_target_roles: list[str] | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ) -> Any:
        if lora_target_roles is not None and self.plan.family != "wan":
            raise ValueError("lora_target_roles is only supported for Wan runtimes.")
        model_class = self._definition.load_class()
        kwargs: dict[str, Any] = {
            "quantize": quantize,
            "model_path": model_path,
            "lora_paths": lora_paths,
            "lora_scales": lora_scales,
            "model_config": self.model_config,
            **self._definition.extra_kwargs,
        }
        if self.plan.family == "wan":
            kwargs["lora_target_roles"] = lora_target_roles
        # Host-provided constructor extras (for example Wan
        # keep_text_encoder_resident / prompt_embed_disk_cache) come last so
        # embedding apps can reach model-specific controls through the
        # public wrapper without bespoke construction code.
        if model_kwargs:
            kwargs.update(model_kwargs)
        return model_class(**kwargs)


@dataclass(frozen=True)
class _RuntimeDefinition:
    runtime_id: str
    import_path: str
    extra_kwargs: dict[str, Any] = field(default_factory=dict)

    def load_class(self) -> type[Any]:
        module_name, class_name = self.import_path.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)


class _RuntimeGenerationExecutor:
    @staticmethod
    def generate_outputs(
        *,
        loaded: LoadedGenerationModel,
        seeds: Sequence[int],
        output: str | Path | None,
        overwrite: bool,
        progress_callback: ProgressCallback | None,
        save_kwargs: dict[str, Any] | None,
        generate_kwargs: dict[str, Any],
    ) -> list[GeneratedOutput]:
        resolved_seeds = [int(seed) for seed in seeds]
        if not resolved_seeds:
            raise ValueError("seeds must contain at least one value.")
        if len(set(resolved_seeds)) != len(resolved_seeds):
            raise ValueError("Duplicate seeds are not supported for Python multi-output generation.")
        output_template = _RuntimeGenerationExecutor._normalized_output_template(
            loaded=loaded,
            seeds=resolved_seeds,
            output=output,
        )
        generate_method = _RuntimeGenerationExecutor._generate_method(loaded)
        results: list[GeneratedOutput] = []
        item_count = len(resolved_seeds)

        for item_index, seed in enumerate(resolved_seeds, start=1):
            output_path = (
                resolve_output_path(str(output_template), overwrite=overwrite, seed=seed)
                if output_template is not None
                else None
            )
            last_event: ProgressEvent | None = None
            model_failed = False
            unsubscribe = None
            if progress_callback is not None:

                def on_event(event: ProgressEvent) -> None:
                    nonlocal last_event, model_failed
                    last_event = event
                    if event.phase == "failed":
                        model_failed = True
                    progress_callback(event)

                unsubscribe = _RuntimeGenerationExecutor._subscribe_progress(
                    loaded=loaded,
                    seed=seed,
                    item_index=item_index,
                    item_count=item_count,
                    output_path=output_path,
                    emit_generated=output_path is not None,
                    progress_callback=on_event,
                )
            try:
                artifact = generate_method(seed=seed, **generate_kwargs)
                saved_path = None
                if output_path is not None:
                    _RuntimeGenerationExecutor._emit_terminal_progress(
                        phase="save",
                        task=getattr(artifact, "task", None) or loaded.plan.task,
                        seed=seed,
                        item_index=item_index,
                        item_count=item_count,
                        output_path=output_path,
                        last_event=last_event,
                        progress_callback=progress_callback,
                    )
                    saved_path = artifact.save(path=output_path, overwrite=True, **(save_kwargs or {}))
                    _RuntimeGenerationExecutor._emit_terminal_progress(
                        phase="complete",
                        task=getattr(artifact, "task", None) or loaded.plan.task,
                        seed=seed,
                        item_index=item_index,
                        item_count=item_count,
                        output_path=saved_path or output_path,
                        last_event=last_event,
                        progress_callback=progress_callback,
                    )
                results.append(
                    GeneratedOutput(
                        seed=seed,
                        task=getattr(artifact, "task", None) or loaded.plan.task,
                        artifact=artifact,
                        item_index=item_index,
                        item_count=item_count,
                        output_path=output_path,
                        saved_path=saved_path,
                    )
                )
            except Exception:
                if not model_failed:
                    _RuntimeGenerationExecutor._emit_terminal_progress(
                        phase="failed",
                        task=last_event.task if last_event is not None else loaded.plan.task,
                        seed=seed,
                        item_index=item_index,
                        item_count=item_count,
                        output_path=output_path,
                        last_event=last_event,
                        progress_callback=progress_callback,
                    )
                raise
            finally:
                if unsubscribe is not None:
                    unsubscribe()
        return results

    @staticmethod
    def _generate_method(loaded: LoadedGenerationModel):
        if loaded.plan.task in {"text-to-image", "image-to-image"}:
            return getattr(loaded.model, "generate_image")
        if loaded.plan.task in {"text-to-video", "image-to-video", "video-to-video"}:
            return getattr(loaded.model, "generate_video")
        raise TaskInferenceError(f"Unsupported runtime task {loaded.plan.task!r} for Python generation execution.")

    @staticmethod
    def _normalized_output_template(
        *,
        loaded: LoadedGenerationModel,
        seeds: list[int],
        output: str | Path | None,
    ) -> str | None:
        if output is None:
            return None
        return normalize_output_template(
            str(output),
            is_video=loaded.plan.task in {"text-to-video", "image-to-video", "video-to-video"},
            include_seed=len(seeds) > 1,
        )

    @staticmethod
    def _subscribe_progress(
        *,
        loaded: LoadedGenerationModel,
        seed: int,
        item_index: int,
        item_count: int,
        output_path: Path | None,
        emit_generated: bool,
        progress_callback: ProgressCallback | None,
    ):
        if progress_callback is None:
            return None
        callbacks = getattr(loaded.model, "callbacks", None)
        if callbacks is None or not hasattr(callbacks, "subscribe_progress"):
            return None

        def on_progress(event: ProgressEvent) -> None:
            phase = "generated" if emit_generated and event.phase == "complete" else event.phase
            progress_callback(
                replace(
                    event,
                    phase=phase,
                    seed=seed,
                    item_index=item_index,
                    item_count=item_count,
                    output_path=str(output_path) if output_path is not None else None,
                )
            )

        return callbacks.subscribe_progress(on_progress)

    @staticmethod
    def _emit_terminal_progress(
        *,
        phase: str,
        task: str,
        seed: int,
        item_index: int,
        item_count: int,
        output_path: Path | None,
        last_event: ProgressEvent | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        if progress_callback is None:
            return
        progress_callback(
            ProgressEvent(
                phase=phase,
                frame=last_event.frame if last_event is not None else None,
                total_frames=last_event.total_frames if last_event is not None else None,
                step=last_event.step if last_event is not None else 0,
                total_steps=last_event.total_steps if last_event is not None else 0,
                task=task,
                timestep=last_event.timestep if last_event is not None else None,
                seed=seed,
                item_index=item_index,
                item_count=item_count,
                output_path=str(output_path) if output_path is not None else None,
            )
        )


def resolve_generation_runtime(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
    image_count: int = 0,
    video_count: int = 0,
    task: str | None = "auto",
    i2i_mode: str | None = "auto",
    has_image_strength: bool = False,
    has_video_strength: bool = False,
    has_video_mask: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
) -> GenerationRuntimePlan:
    plan = resolve_generation_plan(
        model=model,
        model_config=model_config,
        family=family,
        base_model=base_model,
        image_count=image_count,
        video_count=video_count,
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
        has_video_strength=has_video_strength,
        has_video_mask=has_video_mask,
        has_mask=has_mask,
        has_control_image=has_control_image,
        has_outpaint=has_outpaint,
        has_reframe=has_reframe,
        has_lora=has_lora,
    )
    return resolve_generation_runtime_for_plan(
        plan=plan,
        model=model,
        model_config=model_config,
        base_model=base_model,
    )


def resolve_generation_runtime_for_plan(
    *,
    plan: GenerationPlan,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    base_model: str | None = None,
) -> GenerationRuntimePlan:
    resolved_model_config = model_config or _model_config_for_plan(plan=plan, model=model, base_model=base_model)
    definition = _runtime_definition_for_plan(plan, model_config=resolved_model_config)
    cache_key_base = _cache_key_base(
        runtime_id=definition.runtime_id,
        model_name=resolved_model_config.model_name,
        control_model=plan.control_model,
    )
    return GenerationRuntimePlan(
        plan=plan,
        model_config=resolved_model_config,
        runtime_id=definition.runtime_id,
        cache_key_base=cache_key_base,
        _definition=definition,
    )


def load_generation_model(
    *,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    family: str | None = None,
    base_model: str | None = None,
    image_count: int = 0,
    video_count: int = 0,
    task: str | None = "auto",
    i2i_mode: str | None = "auto",
    has_image_strength: bool = False,
    has_video_strength: bool = False,
    has_video_mask: bool = False,
    has_mask: bool = False,
    has_control_image: bool = False,
    has_outpaint: bool = False,
    has_reframe: bool = False,
    has_lora: bool = False,
    quantize: int | None = None,
    model_path: str | None = None,
    lora_paths: list[str] | None = None,
    lora_scales: list[float] | None = None,
    lora_target_roles: list[str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> LoadedGenerationModel:
    runtime = resolve_generation_runtime(
        model=model,
        model_config=model_config,
        family=family,
        base_model=base_model,
        image_count=image_count,
        video_count=video_count,
        task=task,
        i2i_mode=i2i_mode,
        has_image_strength=has_image_strength,
        has_video_strength=has_video_strength,
        has_video_mask=has_video_mask,
        has_mask=has_mask,
        has_control_image=has_control_image,
        has_outpaint=has_outpaint,
        has_reframe=has_reframe,
        has_lora=has_lora,
    )
    return load_generation_model_for_plan(
        plan=runtime.plan,
        model=model,
        model_config=runtime.model_config,
        base_model=base_model,
        quantize=quantize,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        lora_target_roles=lora_target_roles,
        model_kwargs=model_kwargs,
    )


def load_generation_model_for_plan(
    *,
    plan: GenerationPlan,
    model: str | None = None,
    model_config: ModelConfig | None = None,
    base_model: str | None = None,
    quantize: int | None = None,
    model_path: str | None = None,
    lora_paths: list[str] | None = None,
    lora_scales: list[float] | None = None,
    lora_target_roles: list[str] | None = None,
    model_kwargs: dict[str, Any] | None = None,
) -> LoadedGenerationModel:
    runtime = resolve_generation_runtime_for_plan(
        plan=plan,
        model=model,
        model_config=model_config,
        base_model=base_model,
    )
    cache_key = runtime.cache_key(
        quantize=quantize,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        lora_target_roles=lora_target_roles,
        model_kwargs=model_kwargs,
    )
    model_instance = runtime.load(
        quantize=quantize,
        model_path=model_path,
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        lora_target_roles=lora_target_roles,
        model_kwargs=model_kwargs,
    )
    return LoadedGenerationModel(
        plan=runtime.plan,
        model_config=runtime.model_config,
        runtime_id=runtime.runtime_id,
        cache_key_base=runtime.cache_key_base,
        cache_key=cache_key,
        model=model_instance,
    )


def _model_config_for_plan(
    *,
    plan: GenerationPlan,
    model: str | None,
    base_model: str | None,
) -> ModelConfig:
    resolved_name = model or plan.model_override or plan.model_name
    if resolved_name is None:
        raise TaskInferenceError(
            "Cannot resolve a model config for this generation plan. Pass model_config=... or a concrete model name."
        )
    return ModelConfig.from_name(resolved_name, base_model=base_model)


def _cache_key_base(*, runtime_id: str, model_name: str, control_model: str | None) -> str:
    parts = [runtime_id, model_name]
    if control_model is not None:
        parts.append(control_model)
    return "::".join(parts)


def _runtime_definition_for_plan(plan: GenerationPlan, model_config: ModelConfig | None = None) -> _RuntimeDefinition:
    if plan.handler_id == "bonsai.generate":
        return _RuntimeDefinition(
            runtime_id="bonsai-image",
            import_path="mflux.models.bonsai_image.variants.bonsai_image.BonsaiImage",
        )
    if plan.handler_id == "ernie-image.generate":
        return _RuntimeDefinition(
            runtime_id="ernie-image-turbo",
            import_path="mflux.models.ernie_image.variants.ernie_image_turbo.ErnieImageTurbo",
        )
    if plan.handler_id == "z-image.generate":
        return _RuntimeDefinition(
            runtime_id="z-image",
            import_path="mflux.models.z_image.variants.z_image.ZImage",
        )
    if plan.handler_id == "z-image-turbo.generate":
        return _RuntimeDefinition(
            runtime_id="z-image-turbo",
            import_path="mflux.models.z_image.variants.ZImageTurbo",
        )
    if plan.handler_id == "fibo.generate":
        return _RuntimeDefinition(
            runtime_id="fibo",
            import_path="mflux.models.fibo.variants.txt2img.fibo.FIBO",
        )
    if plan.handler_id == "wan.generate":
        if model_config is not None and bool(model_config.transformer_overrides.get("supports_vace", False)):
            return _RuntimeDefinition(
                runtime_id="wan-vace",
                import_path="mflux.models.wan.variants.wan_vace.WanVace",
            )
        return _RuntimeDefinition(
            runtime_id="wan2.2-ti2v",
            import_path="mflux.models.wan.variants.wan2_2_ti2v.Wan2_2_TI2V",
        )
    if plan.handler_id == "qwen.edit":
        return _RuntimeDefinition(
            runtime_id="qwen.edit",
            import_path="mflux.models.qwen.variants.edit.qwen_image_edit.QwenImageEdit",
        )
    if plan.handler_id == "qwen.generate":
        if plan.control_model is not None:
            return _RuntimeDefinition(
                runtime_id="qwen.controlnet",
                import_path="mflux.models.qwen.variants.controlnet.qwen_image_controlnet.QwenImageControlNet",
                extra_kwargs={"controlnet_model": plan.control_model},
            )
        return _RuntimeDefinition(
            runtime_id="qwen.image",
            import_path="mflux.models.qwen.variants.txt2img.qwen_image.QwenImage",
        )
    if plan.handler_id == "flux2.generate":
        return _RuntimeDefinition(
            runtime_id="flux2.klein",
            import_path="mflux.models.flux2.variants.txt2img.flux2_klein.Flux2Klein",
        )
    if plan.handler_id == "flux2.edit":
        if plan.capability_id == "flux2.outpaint":
            return _RuntimeDefinition(
                runtime_id="flux2.klein-outpaint",
                import_path="mflux.models.flux2.variants.edit.flux2_klein_outpaint.Flux2KleinOutpaint",
            )
        if plan.capability_id == "flux2.inpaint":
            return _RuntimeDefinition(
                runtime_id="flux2.klein-inpaint",
                import_path="mflux.models.flux2.variants.edit.flux2_klein_inpaint.Flux2KleinInpaint",
            )
        return _RuntimeDefinition(
            runtime_id="flux2.klein-edit",
            import_path="mflux.models.flux2.variants.edit.flux2_klein_edit.Flux2KleinEdit",
        )
    raise TaskInferenceError(f"Unsupported runtime handler {plan.handler_id!r} for Python generation loading.")
