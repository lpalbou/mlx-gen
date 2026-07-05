import argparse
from pathlib import Path

import toml

from mflux.cli.completions.generator import CompletionGenerator

# Commands that intentionally have no static completion script. The mlxgen router entrypoints
# are subcommand-based (generate|upscale|capabilities|...) and forward most options to the
# selected backend, so a flat completion built from the router parser would list only the 15
# routing options and mislead users; completion for them is deferred until the generator can
# express subcommand-aware completion.
EXCLUDED_SCRIPTS = {
    "mlxgen",
    "mlxgen-generate",
    "mlx-gen",
    "mlx-generate",
}


def _project_scripts() -> dict:
    pyproject = Path(__file__).resolve().parents[2] / "pyproject.toml"
    return toml.load(pyproject)["project"]["scripts"]


def test_every_console_script_has_a_completion_or_documented_exclusion():
    scripts = set(_project_scripts())
    generator = CompletionGenerator()
    covered = set(generator.commands) | EXCLUDED_SCRIPTS

    missing = sorted(scripts - covered)
    assert not missing, f"console scripts without shell completion coverage: {missing}"


def test_no_stale_completion_commands():
    scripts = set(_project_scripts())
    generator = CompletionGenerator()

    stale = sorted(set(generator.commands) - scripts)
    assert not stale, f"completion commands with no console script: {stale}"


def test_every_completion_command_yields_a_real_parser():
    generator = CompletionGenerator()
    for command in generator.commands:
        parser = generator.create_parser_for_command(command)
        actions = [action for action in parser._actions if not isinstance(action, argparse._HelpAction)]
        assert actions, f"{command} produces an empty completion (no non-help parser actions)"
