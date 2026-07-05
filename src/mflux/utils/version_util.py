from __future__ import annotations

import importlib.metadata
import re
from pathlib import Path

import toml


class VersionUtil:
    PACKAGED_RELEASE_DATE = "2026-07-06"
    _version: str | None = None
    _release_date: str | None = None

    @staticmethod
    def get_mflux_version() -> str:
        if VersionUtil._version is None:
            VersionUtil._version = VersionUtil._scan_pyproject() or VersionUtil._get_installed_version() or "unknown"
        return VersionUtil._version

    @staticmethod
    def get_mflux_release_date() -> str:
        if VersionUtil._release_date is not None:
            return VersionUtil._release_date
        version = VersionUtil.get_mflux_version()
        VersionUtil._release_date = (
            VersionUtil._scan_changelog_release_date(version) or VersionUtil.PACKAGED_RELEASE_DATE
        )
        return VersionUtil._release_date

    @staticmethod
    def format_cli_release_label() -> str:
        version = VersionUtil.get_mflux_version()
        if version == "unknown":
            return "MLX-Gen"
        return f"MLX-Gen {version} ({VersionUtil.get_mflux_release_date()})"

    @staticmethod
    def _scan_pyproject() -> str | None:
        current_dir = Path(__file__).resolve().parent
        for parent in current_dir.parents:
            pyproject_path = parent / "pyproject.toml"
            if pyproject_path.exists():
                try:
                    data = toml.load(pyproject_path)
                    return data.get("project", {}).get("version")
                except Exception:  # noqa: BLE001
                    return None
        return None

    @staticmethod
    def _get_installed_version() -> str | None:
        for distribution_name in ("mlx-gen", "abstractvision-mflux", "mflux"):
            version = VersionUtil._read_installed_distribution_version(distribution_name)
            if version is not None:
                return version
        return None

    @staticmethod
    def _read_installed_distribution_version(distribution_name: str) -> str | None:
        try:
            return importlib.metadata.version(distribution_name)
        except importlib.metadata.PackageNotFoundError:
            return None

    @staticmethod
    def _scan_changelog_release_date(version: str) -> str | None:
        if not version or version == "unknown":
            return None

        version_pattern = re.compile(rf"^## \[{re.escape(version)}\](?:\s*-\s*(.+))?.*$")
        current_dir = Path(__file__).resolve().parent
        for parent in current_dir.parents:
            changelog_path = parent / "CHANGELOG.md"
            if not changelog_path.exists():
                continue
            try:
                for line in changelog_path.read_text(encoding="utf-8").splitlines():
                    match = version_pattern.match(line)
                    if match:
                        date = match.group(1)
                        return date.strip() if date else None
            except Exception:  # noqa: BLE001
                return None
        return None
