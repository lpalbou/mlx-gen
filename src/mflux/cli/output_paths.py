from pathlib import Path

from mflux.utils.image_util import ImageUtil


def ensure_output_stem_token(output: str, *, token: str, suffix: str) -> str:
    output_path = Path(output)
    normalized_stem = output_path.stem.replace("{image_name}", "{input_name}")
    if token in normalized_stem:
        return str(output_path.with_stem(normalized_stem))
    return str(output_path.with_stem(normalized_stem + suffix))


def normalize_output_template(
    output: str,
    *,
    is_video: bool = False,
    include_seed: bool = False,
    include_input_name: bool = False,
) -> str:
    normalized = "video.mp4" if is_video and output == "image.png" else output
    if include_seed:
        normalized = ensure_output_stem_token(
            normalized,
            token="{seed}",
            suffix="_seed_{seed}",
        )
    if include_input_name:
        normalized = ensure_output_stem_token(
            normalized,
            token="{input_name}",
            suffix="_{input_name}",
        )
    return normalized


def format_output_template(
    output: str,
    *,
    seed: int | None = None,
    input_name: str | None = None,
) -> str:
    values: dict[str, object] = {}
    if seed is not None:
        values["seed"] = seed
    if input_name is not None:
        values["input_name"] = input_name
        values["image_name"] = input_name
    return output.format(**values) if values else output


def resolve_output_path(
    output: str,
    *,
    overwrite: bool,
    seed: int | None = None,
    input_name: str | None = None,
) -> Path:
    return ImageUtil.resolve_output_path(
        path=format_output_template(output, seed=seed, input_name=input_name),
        overwrite=overwrite,
    )
