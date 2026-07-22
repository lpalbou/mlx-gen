from __future__ import annotations

import json
import sys
from collections.abc import Callable, Collection
from pathlib import Path
from typing import TextIO

from mflux.callbacks import ProgressEvent
from mflux.models.common.download_policy import DownloadRequiredError


def cli_print(message: str, *, json_events: bool, error: bool = False) -> None:
    stream = sys.stderr if json_events or error else sys.stdout
    print(message, file=stream)


class CliArgumentError(ValueError):
    def __init__(self, message: str, *, usage: str | None = None):
        super().__init__(message)
        self.usage = usage


class CliRuntimeEventStream:
    def __init__(
        self,
        *,
        enabled: bool,
        command: str,
        model: str | None = None,
        seed: int | None = None,
        stream: TextIO | None = None,
    ):
        self.enabled = enabled
        self._sink = (
            _JsonlEventSink(
                stream=stream or sys.stdout,
                command=command,
                model=model,
                seed=seed,
            )
            if enabled
            else None
        )
        self._output_path: str | None = None
        self._last_event: ProgressEvent | None = None
        self._last_task: str | None = None

    def set_output_path(self, output_path: str | Path) -> None:
        self._output_path = str(output_path)

    def subscribe_model(
        self,
        model,
        *,
        map_complete_to_generated: bool,
        suppress_terminal_phases: Collection[str] = (),
    ) -> Callable[[], None] | None:
        if not self.enabled:
            return None
        callbacks = getattr(model, "callbacks", None)
        if callbacks is None or not hasattr(callbacks, "subscribe_progress"):
            return None
        suppressed = set(suppress_terminal_phases)

        def on_progress(event: ProgressEvent) -> None:
            self.handle_progress(
                event,
                map_complete_to_generated=map_complete_to_generated,
                suppress_terminal_phases=suppressed,
            )

        return callbacks.subscribe_progress(on_progress)

    def handle_progress(
        self,
        event: ProgressEvent,
        *,
        map_complete_to_generated: bool = False,
        suppress_terminal_phases: Collection[str] = (),
    ) -> None:
        if not self.enabled or self._sink is None:
            return
        self._last_event = event
        if event.task is not None:
            self._last_task = event.task
        if event.phase in suppress_terminal_phases:
            return
        phase = "generated" if map_complete_to_generated and event.phase == "complete" else event.phase
        self._sink.emit(
            phase=phase,
            task=event.task,
            step=event.step,
            total_steps=event.total_steps,
            progress=event.progress,
            frame=event.frame,
            total_frames=event.total_frames,
            frame_progress=event.frame_progress,
            timestep=event.timestep,
            output_path=self._output_path if phase in {"save", "complete", "failed"} else None,
        )

    def emit_save(
        self,
        *,
        task: str | None = None,
        health_check: str | None = None,
        fps: int | float | None = None,
        width: int | None = None,
        height: int | None = None,
        total_frames: int | None = None,
    ) -> None:
        # Save events optionally carry the output's fps/frame/dimension facts
        # so embedded hosts can build artifact metadata without re-decoding
        # the file, plus a `health_check: "skipped"` marker when the caller
        # opted out of the post-save validation decode.
        self._emit_terminal_event(
            phase="save",
            task=task,
            health_check=health_check,
            fps=fps,
            width=width,
            height=height,
            total_frames_override=total_frames,
        )

    def emit_complete(self, *, task: str | None = None) -> None:
        self._emit_terminal_event(phase="complete", task=task)

    def emit_failed(
        self,
        *,
        task: str | None = None,
        error: BaseException | None = None,
        diagnostics_path: str | Path | None = None,
    ) -> None:
        if not self.enabled or self._sink is None:
            return
        event = self._last_event
        resolved_task = task or self._last_task
        self._sink.emit(
            phase="failed",
            task=resolved_task,
            step=event.step if event is not None else 0,
            total_steps=event.total_steps if event is not None else 0,
            progress=event.progress if event is not None else 0.0,
            frame=event.frame if event is not None else None,
            total_frames=event.total_frames if event is not None else None,
            frame_progress=event.frame_progress if event is not None else None,
            timestep=event.timestep if event is not None else None,
            output_path=self._output_path,
            diagnostics_path=str(diagnostics_path) if diagnostics_path is not None else None,
            error=str(error) if error is not None else None,
            error_type=type(error).__name__ if error is not None else None,
            remediation=_remediation_for_error(error),
        )

    def _emit_terminal_event(
        self,
        *,
        phase: str,
        task: str | None,
        health_check: str | None = None,
        fps: int | float | None = None,
        width: int | None = None,
        height: int | None = None,
        total_frames_override: int | None = None,
    ) -> None:
        if not self.enabled or self._sink is None:
            return
        event = self._last_event
        resolved_task = task or self._last_task
        step = event.step if event is not None else 0
        total_steps = event.total_steps if event is not None else 0
        frame = event.frame if event is not None else None
        total_frames = event.total_frames if event is not None else None
        if total_frames_override is not None:
            total_frames = total_frames_override
        frame_progress = event.frame_progress if event is not None else None
        timestep = event.timestep if event is not None else None
        progress = 1.0 if total_steps <= 0 else min(1.0, max(0.0, step / total_steps))
        self._sink.emit(
            phase=phase,
            task=resolved_task,
            step=step,
            total_steps=total_steps,
            progress=progress,
            frame=frame,
            total_frames=total_frames,
            frame_progress=frame_progress,
            timestep=timestep,
            health_check=health_check,
            fps=fps,
            width=width,
            height=height,
            output_path=self._output_path,
        )


# Preserve the older export name for callers that imported the first draft.
CliEventStream = CliRuntimeEventStream


class _JsonlEventSink:
    def __init__(
        self,
        *,
        stream: TextIO,
        command: str,
        model: str | None,
        seed: int | None,
    ):
        self._stream = stream
        self._command = command
        self._model = model
        self._seed = seed

    def emit(
        self,
        *,
        phase: str,
        task: str | None,
        step: int,
        total_steps: int,
        progress: float,
        frame: int | None = None,
        total_frames: int | None = None,
        frame_progress: float | None = None,
        timestep: int | float | None = None,
        output_path: str | None = None,
        diagnostics_path: str | None = None,
        error: str | None = None,
        error_type: str | None = None,
        remediation: dict[str, object] | None = None,
        health_check: str | None = None,
        fps: int | float | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        payload = {
            "type": "runtime",
            "command": self._command,
            "model": self._model,
            "seed": self._seed,
            "phase": phase,
            "task": task,
            "step": step,
            "total_steps": total_steps,
            "progress": progress,
            "frame": frame,
            "total_frames": total_frames,
            "frame_progress": frame_progress,
            "timestep": timestep,
            "output_path": output_path,
            "diagnostics_path": diagnostics_path,
            "error": error,
            "error_type": error_type,
            "remediation": remediation,
            "health_check": health_check,
            "fps": fps,
            "width": width,
            "height": height,
        }
        compact = {key: value for key, value in payload.items() if value is not None}
        self._stream.write(json.dumps(compact) + "\n")
        self._stream.flush()


def emit_cli_failure_event_for_argv(
    *,
    prog: str,
    argv: list[str],
    error: BaseException,
    task: str | None = None,
    diagnostics_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> None:
    events = CliRuntimeEventStream(
        enabled=True,
        command=_command_for_prog(prog),
        model=_option_value(argv, "--model") or _option_value(argv, "-m"),
        seed=_seed_from_argv(argv),
    )
    resolved_output_path = output_path or _option_value(argv, "--output")
    if resolved_output_path is not None:
        events.set_output_path(resolved_output_path)
    events.emit_failed(task=task, error=error, diagnostics_path=diagnostics_path)


def _command_for_prog(prog: str) -> str:
    if "upscale" in prog:
        return "mlxgen upscale"
    if "generate" in prog:
        return "mlxgen generate"
    return prog


def _option_value(argv: list[str], option_name: str) -> str | None:
    for index, token in enumerate(argv):
        if token == option_name:
            if index + 1 >= len(argv):
                return None
            return argv[index + 1]
        if token.startswith(f"{option_name}="):
            return token.split("=", 1)[1]
    return None


def _seed_from_argv(argv: list[str]) -> int | None:
    value = _option_value(argv, "--seed") or _option_value(argv, "-s")
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _remediation_for_error(error: BaseException | None) -> dict[str, object] | None:
    if error is None:
        return None
    if isinstance(error, DownloadRequiredError):
        remediation = {
            "kind": "download-required",
            "repo_id": error.repo_id,
            "artifact": error.artifact,
            "download_command": error.download_command,
        }
        if error.prepare_command is not None:
            remediation["prepare_command"] = error.prepare_command
        return remediation
    if isinstance(error, CliArgumentError):
        remediation = {"kind": "cli-usage"}
        if error.usage is not None:
            remediation["usage"] = error.usage.strip()
        return remediation
    return None
