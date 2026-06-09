from pathlib import Path

from huggingface_hub.utils import LocalEntryNotFoundError

from mflux.cli.defaults.defaults import MFLUX_LORA_CACHE_DIR
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRAApplicationError
from mflux.models.common.resolution.lora_resolution import LoraResolution


class LoRACompatibility:
    @staticmethod
    def validate_for_model_config(
        *,
        model_config: ModelConfig,
        lora_paths: list[str] | None,
        selected_model: str | None = None,
    ) -> None:
        if not lora_paths:
            return

        selected_class = LoRACompatibility._model_class(model_config.model_name)
        if selected_class is None and model_config.base_model is not None:
            selected_class = LoRACompatibility._model_class(model_config.base_model)
        if selected_class is None and selected_model is not None:
            selected_class = LoRACompatibility._model_class(selected_model)

        selected_label = selected_model or model_config.model_name
        for lora_path in lora_paths:
            repo_id = LoRACompatibility._repo_id_from_path(lora_path)
            if repo_id is None:
                continue
            for adapter_base_model in LoRACompatibility._cached_base_models(repo_id):
                adapter_class = LoRACompatibility._model_class(adapter_base_model)
                if adapter_class is None or selected_class is None or adapter_class == selected_class:
                    continue
                if adapter_class == "flux2-dev":
                    raise LoRAApplicationError(
                        f"LoRA {repo_id} targets {adapter_base_model}, but {selected_label} resolves to "
                        "FLUX.2 Klein. MLX-Gen does not currently support FLUX.2-dev; use a LoRA trained "
                        "for FLUX.2 Klein or add first-class FLUX.2-dev support."
                    )
                raise LoRAApplicationError(
                    f"LoRA {repo_id} targets {adapter_base_model}, which is not compatible with "
                    f"selected model {selected_label}."
                )

    @staticmethod
    def _repo_id_from_path(path: str) -> str | None:
        if path.startswith(("./", "../", "~/", "/")):
            return None
        if path.startswith("hf:"):
            path = path[3:]
        if ":" in path and "/" in path:
            return path.split(":", 1)[0]
        if path.count("/") == 1:
            return path
        return None

    @staticmethod
    def _cached_base_models(repo_id: str) -> tuple[str, ...]:
        try:
            snapshot = LoraResolution._load_cached_snapshot(repo_id, ["README.md"], MFLUX_LORA_CACHE_DIR)
        except LocalEntryNotFoundError:
            return ()
        readme_path = snapshot / "README.md"
        if not readme_path.exists():
            return ()
        try:
            return LoRACompatibility._base_models_from_card(readme_path)
        except OSError:
            return ()

    @staticmethod
    def _base_models_from_card(readme_path: Path) -> tuple[str, ...]:
        lines = readme_path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].strip() != "---":
            return ()

        frontmatter: list[str] = []
        for line in lines[1:]:
            if line.strip() == "---":
                break
            frontmatter.append(line)

        base_models: list[str] = []
        in_base_model_list = False
        for line in frontmatter:
            stripped = line.strip()
            if stripped.startswith("base_model:"):
                value = stripped.split(":", 1)[1].strip().strip("'\"")
                if value:
                    base_models.append(value)
                    in_base_model_list = False
                else:
                    in_base_model_list = True
                continue
            if in_base_model_list and stripped.startswith("- "):
                base_models.append(stripped[2:].strip().strip("'\""))
                continue
            if in_base_model_list and not stripped.startswith("- "):
                in_base_model_list = False
        return tuple(base_models)

    @staticmethod
    def _model_class(model_name: str | None) -> str | None:
        if model_name is None:
            return None
        normalized = model_name.lower()
        checks = (
            ("black-forest-labs/flux.2-dev", "flux2-dev"),
            ("flux.2-dev", "flux2-dev"),
            ("qwen/qwen-image-edit-2511", "qwen-image-edit-2511"),
            ("qwen-image-edit-2511", "qwen-image-edit-2511"),
            ("qwen/qwen-image-edit-2509", "qwen-image-edit-2509"),
            ("qwen-image-edit-2509", "qwen-image-edit-2509"),
            ("qwen/qwen-image-edit", "qwen-image-edit"),
            ("qwen-image-edit", "qwen-image-edit"),
            ("qwen/qwen-image", "qwen-image"),
            ("qwen-image", "qwen-image"),
            ("black-forest-labs/flux.2-klein-4b", "flux2-klein-4b"),
            ("flux.2-klein-4b", "flux2-klein-4b"),
            ("flux2-klein-4b", "flux2-klein-4b"),
            ("black-forest-labs/flux.2-klein-9b", "flux2-klein-9b"),
            ("flux.2-klein-9b", "flux2-klein-9b"),
            ("flux2-klein-9b", "flux2-klein-9b"),
            ("z-image-turbo", "z-image-turbo"),
            ("z-image", "z-image"),
        )
        for marker, model_class in checks:
            if marker in normalized:
                return model_class
        return None
