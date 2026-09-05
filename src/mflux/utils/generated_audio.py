"""Audio generated alongside a video (MiniMax-H3 emits a stereo 32 kHz track with every clip)."""

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class GeneratedAudio:
    waveform: np.ndarray  # (channels, samples) float32 in [-1, 1]
    sample_rate: int

    def __post_init__(self):
        if self.waveform.ndim != 2:
            raise ValueError("GeneratedAudio.waveform must be (channels, samples).")
        if self.sample_rate <= 0:
            raise ValueError("GeneratedAudio.sample_rate must be positive.")

    @property
    def channels(self) -> int:
        return int(self.waveform.shape[0])

    @property
    def num_samples(self) -> int:
        return int(self.waveform.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.num_samples / float(self.sample_rate)

    def trimmed(self, duration_seconds: float) -> "GeneratedAudio":
        """The first `duration_seconds` of the track (the audio VAE rounds up to whole latent frames)."""
        samples = min(self.num_samples, int(round(duration_seconds * self.sample_rate)))
        return GeneratedAudio(waveform=self.waveform[:, :samples], sample_rate=self.sample_rate)

    def to_pcm16(self) -> np.ndarray:
        """Interleaved `(samples, channels)` int16 samples."""
        clipped = np.clip(self.waveform.astype(np.float32), -1.0, 1.0)
        return np.ascontiguousarray((clipped.T * 32767.0).round().astype(np.int16))

    def save_wav(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(self.channels)
            handle.setsampwidth(2)
            handle.setframerate(self.sample_rate)
            handle.writeframes(self.to_pcm16().tobytes())
        return path

    def metadata(self) -> dict:
        return {
            "audio_present": True,
            "audio_source": "generated",
            "audio_channels": self.channels,
            "audio_sample_rate": self.sample_rate,
            "audio_duration_seconds": round(self.duration_seconds, 4),
        }
