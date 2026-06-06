import json
import logging
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE

from mflux.models.common.download_policy import downloads_enabled, raise_download_required
from mflux.models.common.resolution.actions import PathAction, Rule

logger = logging.getLogger(__name__)


class PathResolution:
    RULES = frozenset(
        {
            Rule(priority=0, name="none", check="is_none", action=PathAction.LOCAL),
            Rule(priority=1, name="local", check="exists_locally", action=PathAction.LOCAL),
            Rule(priority=2, name="local_prepared_hf", check="has_local_prepared_hf", action=PathAction.LOCAL_PREPARED),
            Rule(priority=3, name="hf_cached", check="is_hf_cached", action=PathAction.HUGGINGFACE_CACHED),
            Rule(priority=4, name="hf_download", check="is_hf_format", action=PathAction.HUGGINGFACE),
            Rule(priority=5, name="error", check="always", action=PathAction.ERROR),
        }
    )

    @staticmethod
    def resolve(path: str | None, patterns: list[str] | None = None) -> Path | None:
        if patterns is None:
            patterns = ["*.safetensors"]

        for rule in sorted(PathResolution.RULES, key=lambda r: r.priority):
            if PathResolution._check(rule.check, path, patterns):
                logger.debug(f"Path resolution: '{path}' → rule '{rule.name}' ({rule.action.value})")
                return PathResolution._execute(rule.action, path, patterns)

        raise ValueError(f"No rule matched for path: {path}")

    @staticmethod
    def _is_hf_format(path: str | None) -> bool:
        return path is not None and "/" in path and path.count("/") == 1 and not path.startswith(("./", "../", "~/"))

    @staticmethod
    def _check(check: str, path: str | None, patterns: list[str]) -> bool:
        if check == "is_none":
            return path is None
        if check == "exists_locally":
            if path is None:
                return False
            local_path = Path(path).expanduser()
            if not local_path.exists():
                return False
            # Warn if directory exists but contains no matching files
            if local_path.is_dir():
                has_matching_files = any(list(local_path.glob(p)) for p in patterns)
                if not has_matching_files:
                    print(
                        f"⚠️  Directory '{path}' exists but contains no files matching {patterns}. "
                        f"Model loading may fail."
                    )
            return True
        if check == "is_hf_cached":
            if not PathResolution._is_hf_format(path):
                return False
            # Check if we have a complete cached snapshot
            return PathResolution._find_complete_cached_snapshot(path, patterns) is not None
        if check == "has_local_prepared_hf":
            if not PathResolution._is_hf_format(path):
                return False
            return PathResolution._find_local_prepared_model(path, patterns) is not None
        if check == "is_hf_format":
            return PathResolution._is_hf_format(path)
        if check == "always":
            return True
        return False

    @staticmethod
    def _execute(action: PathAction, path: str | None, patterns: list[str]) -> Path | None:
        if action == PathAction.LOCAL:
            return Path(path).expanduser() if path else None
        if action == PathAction.LOCAL_PREPARED:
            prepared_path = PathResolution._find_local_prepared_model(path, patterns)
            if prepared_path:
                return prepared_path
            raise_download_required(path)
        if action == PathAction.HUGGINGFACE_CACHED:
            # Find the best complete cached snapshot
            cached_path = PathResolution._find_complete_cached_snapshot(path, patterns)
            if cached_path:
                return cached_path
            # Fallback to standard snapshot_download (shouldn't happen if _check passed)
            return Path(snapshot_download(repo_id=path, allow_patterns=patterns, local_files_only=True))
        if action == PathAction.HUGGINGFACE:
            if not downloads_enabled():
                raise_download_required(path)
            print(f"Downloading model from HuggingFace: {path}...")
            return Path(snapshot_download(repo_id=path, allow_patterns=patterns))
        if action == PathAction.ERROR:
            raise FileNotFoundError(
                f"Model not found: '{path}'. "
                f"If local path, make sure it exists. "
                f"If HuggingFace repo, use 'org/model' format."
            )
        return None

    @staticmethod
    def _find_local_prepared_model(repo_id: str | None, patterns: list[str]) -> Path | None:
        if repo_id is None:
            return None

        repo_name = repo_id.split("/")[-1]
        candidates = [
            Path("models") / repo_name,
            Path.home() / "models" / repo_name,
        ]
        required_subdirs = PathResolution._get_required_subdirs_with_safetensors(patterns)

        for candidate in candidates:
            candidate = candidate.expanduser()
            if not candidate.is_dir():
                continue
            if PathResolution._is_snapshot_complete(candidate, required_subdirs, patterns):
                return candidate.resolve()
        return None

    @staticmethod
    def _find_complete_cached_snapshot(repo_id: str | None, patterns: list[str]) -> Path | None:
        if repo_id is None:
            return None

        # Build the cache directory path for this repo
        # HuggingFace cache structure: {cache_dir}/models--{org}--{model}/snapshots/{revision}/
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_dir = Path(HF_HUB_CACHE) / repo_cache_name / "snapshots"

        if not repo_cache_dir.exists():
            return None

        # Extract subdirectories that need safetensors files (e.g., "vae/*.safetensors" → "vae")
        required_subdirs = PathResolution._get_required_subdirs_with_safetensors(patterns)

        # Check each snapshot for completeness, prefer more recent ones
        snapshots = sorted(repo_cache_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)

        for snapshot_path in snapshots:
            if not snapshot_path.is_dir():
                continue
            if PathResolution._is_snapshot_complete(snapshot_path, required_subdirs, patterns):
                logger.debug(f"Found complete cached snapshot: {snapshot_path}")
                return snapshot_path

        return None

    @staticmethod
    def _get_required_subdirs_with_safetensors(patterns: list[str]) -> set[str]:
        subdirs = set()
        for pattern in patterns:
            # Only care about safetensors patterns
            if "*.safetensors" not in pattern:
                continue
            # Handle patterns like "vae/*.safetensors"
            if "/" in pattern:
                subdir = pattern.split("/")[0]
                # Only add if it's a real subdir name (not a glob pattern itself)
                if "*" not in subdir:
                    subdirs.add(subdir)
        return subdirs

    @staticmethod
    def _is_snapshot_complete(
        snapshot_path: Path, required_subdirs: set[str], patterns: list[str] | None = None
    ) -> bool:
        if patterns:
            for pattern in patterns:
                if "*.safetensors" in pattern:
                    continue
                if not PathResolution._pattern_has_valid_match(snapshot_path, pattern):
                    return False

        if not required_subdirs:
            # No specific subdirs required - check that all patterns are satisfied
            if patterns:
                for pattern in patterns:
                    if not PathResolution._pattern_has_valid_match(snapshot_path, pattern):
                        return False
                return True
            else:
                # Fallback: just check for any safetensors
                return any(snapshot_path.glob("**/*.safetensors"))

        for subdir in required_subdirs:
            subdir_path = snapshot_path / subdir
            if not subdir_path.exists():
                return False
            if not PathResolution._subdir_safetensors_complete(subdir_path):
                return False

        return True

    @staticmethod
    def _pattern_has_valid_match(snapshot_path: Path, pattern: str) -> bool:
        matches = list(snapshot_path.glob(pattern))
        return any(PathResolution._path_exists(match) for match in matches)

    @staticmethod
    def _subdir_safetensors_complete(subdir_path: Path) -> bool:
        index_files = sorted(subdir_path.glob("*.safetensors.index.json"))
        if index_files:
            return all(PathResolution._safetensors_index_complete(index_path) for index_path in index_files)
        return any(
            f.name.endswith(".safetensors") and PathResolution._path_exists(f)
            for f in subdir_path.iterdir()
        )

    @staticmethod
    def _safetensors_index_complete(index_path: Path) -> bool:
        if not PathResolution._path_exists(index_path):
            return False
        try:
            index = json.loads(index_path.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        expected_files = set(weight_map.values())
        return all(PathResolution._path_exists(index_path.parent / filename) for filename in expected_files)

    @staticmethod
    def _path_exists(path: Path) -> bool:
        if path.is_symlink():
            return os.path.exists(path)
        return path.exists()
