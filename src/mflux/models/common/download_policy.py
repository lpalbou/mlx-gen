import shlex
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

_DOWNLOADS_ALLOWED: ContextVar[bool] = ContextVar("mlx_gen_downloads_allowed", default=False)


class DownloadRequiredError(FileNotFoundError):
    def __init__(
        self,
        repo_id: str,
        *,
        path: str | None = None,
        artifact: str = "model",
        message: str | None = None,
        download_command: str | None = None,
        prepare_command: str | None = None,
    ):
        self.repo_id = repo_id
        self.artifact = artifact
        self.download_command = download_command or explicit_download_command(repo_id, artifact=artifact)
        if prepare_command is not None:
            self.prepare_command = prepare_command
        elif message is not None:
            self.prepare_command = None
        elif artifact.lower() != "lora":
            self.prepare_command = explicit_prepare_command(repo_id, path=path)
        else:
            self.prepare_command = None
        super().__init__(message or explicit_download_hint(repo_id, path=path, artifact=artifact))


def downloads_enabled() -> bool:
    return _DOWNLOADS_ALLOWED.get()


@contextmanager
def allow_downloads() -> Iterator[None]:
    token = _DOWNLOADS_ALLOWED.set(True)
    try:
        yield
    finally:
        _DOWNLOADS_ALLOWED.reset(token)


def is_huggingface_repo_id(path: str | None) -> bool:
    return path is not None and "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))


def explicit_download_hint(repo_id: str, *, path: str | None = None, artifact: str = "model") -> str:
    local_path = path or f"./models/{_local_model_dir_name(repo_id)}"
    download_command = explicit_download_command(repo_id, artifact=artifact)
    prepare_command = explicit_prepare_command(repo_id, path=local_path)
    if artifact.lower() == "lora":
        return (
            "MLX-Gen will not download LoRA files during generation.\n"
            "Download the required LoRA before starting the workflow:\n"
            f"  {download_command}\n"
            f"Then run generation again with the same LoRA reference, or use the downloaded file path."
        )
    return (
        f"MLX-Gen will not download {artifact} files during generation.\n"
        "Download the required files before starting the workflow:\n"
        f"  {download_command}\n"
        "For a reusable local MLX-Gen model folder, run:\n"
        f"  {prepare_command}\n"
        f"Then run generation again with --model {shlex.quote(repo_id)} or --model {shlex.quote(local_path)}."
    )


def explicit_download_command(repo_id: str, *, artifact: str = "model") -> str:
    suffix = " --all-files" if artifact.lower() == "lora" else ""
    return f"mlxgen download --model {shlex.quote(repo_id)}{suffix}"


def explicit_prepare_command(repo_id: str, *, path: str | None = None, quantize: int = 8) -> str:
    local_path = path or f"./models/{_local_model_dir_name(repo_id)}"
    return f"mlxgen prepare --model {shlex.quote(repo_id)} --path {shlex.quote(local_path)} -q {quantize}"


def raise_download_required(repo_id: str, *, artifact: str = "model") -> None:
    raise DownloadRequiredError(repo_id, artifact=artifact)


def direct_url_download_hint(component_name: str, url: str) -> str:
    command = explicit_direct_url_download_command(component_name)
    display_name = _component_display_name(component_name)
    return (
        f"MLX-Gen will not download {display_name} during generation.\n"
        "Download the required files before starting the workflow:\n"
        f"  {command}\n"
        f"Source URL: {url}\n"
        "Then run the generation or depth command again."
    )


def explicit_direct_url_download_command(component_name: str) -> str:
    return f"mlxgen download --model {shlex.quote(_local_model_dir_name(component_name))}"


def raise_direct_url_download_required(component_name: str, url: str) -> None:
    raise DownloadRequiredError(
        component_name,
        artifact=component_name,
        message=direct_url_download_hint(component_name, url),
        download_command=explicit_direct_url_download_command(component_name),
        prepare_command=None,
    )


def _local_model_dir_name(repo_id: str) -> str:
    return Path(repo_id).name.lower().replace("_", "-")


def _component_display_name(component_name: str) -> str:
    if component_name == "depth_pro":
        return "Depth Pro weights"
    return component_name.replace("_", " ")
