#!/usr/bin/env python3
"""Analyze WAV files with full-file sliding-window FFTs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import statistics
import wave
from pathlib import Path
from xml.sax.saxutils import escape

import numpy as np
from scipy.signal import stft


DEFAULT_DATA_DIR = Path("Data")
DEFAULT_RESULTS_DIR = Path("outputs")
FFT_SIZE = 8192
WAVEFORM_POINTS = 4000
MAX_SPECTRUM_HZ = 2000.0
DEFAULT_ANALYSIS_SAMPLE_RATE_HZ = 5000
RECOMMENDED_MIN_FREQUENCY_HZ = 50.0
SPECTROGRAM_FFT_SIZE = 2048
SPECTROGRAM_BANDS = 180
SPECTROGRAM_COLUMNS = 320
RELEVANT_MIN_HZ = 400.0
RELEVANT_THRESHOLD_FRACTION = 0.10
SMOOTHING_BINS = 9
FILTER_BLOCK_SIZE = 4096
ZOOM_TRIGGER_FRACTION = 0.40
SILENCE_MAGNITUDE = 1e-9


def fft(values: list[complex]) -> list[complex]:
    """Educational reference FFT. Active analysis uses numpy.fft.rfft."""
    n = len(values)
    if n <= 1:
        return values
    if n & (n - 1):
        raise ValueError("FFT input length must be a power of two")

    even = fft(values[0::2])
    odd = fft(values[1::2])
    result = [0j] * n
    half = n // 2

    for k in range(half):
        angle = -2 * math.pi * k / n
        twiddle = complex(math.cos(angle), math.sin(angle)) * odd[k]
        result[k] = even[k] + twiddle
        result[k + half] = even[k] - twiddle

    return result


def decode_pcm_frames(raw: bytes, sample_width: int, channels: int) -> list[float]:
    """Convert WAV PCM bytes to mono samples between about -1.0 and 1.0."""
    bytes_per_frame = sample_width * channels
    frame_count = len(raw) // bytes_per_frame
    samples: list[float] = []

    for frame_index in range(frame_count):
        offset = frame_index * bytes_per_frame
        frame_total = 0.0

        for channel in range(channels):
            start = offset + channel * sample_width
            sample_bytes = raw[start : start + sample_width]

            if sample_width == 1:
                sample_value = sample_bytes[0] - 128
                scale = 128.0
            else:
                sample_value = int.from_bytes(sample_bytes, "little", signed=True)
                scale = float(1 << (8 * sample_width - 1))

            frame_total += sample_value / scale

        samples.append(frame_total / channels)

    return samples


def read_mono_wav(path: Path) -> tuple[list[float], int, int]:
    """Read a WAV file as mono normalized samples."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())

    return decode_pcm_frames(raw, sample_width, channels), sample_rate, channels


def resample_samples_polyphase(
    samples: list[float],
    original_sample_rate: int,
    analysis_sample_rate: int,
) -> list[float]:
    """Resample audio with scipy's anti-aliased polyphase filter."""
    if analysis_sample_rate == original_sample_rate:
        return samples[:]
    if analysis_sample_rate < 2:
        raise ValueError("analysis sample rate must be at least 2 Hz")

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:
        raise RuntimeError(
            "Resampling requires scipy. Install scipy or pass --no-resample."
        ) from exc

    divisor = math.gcd(original_sample_rate, analysis_sample_rate)
    up = analysis_sample_rate // divisor
    down = original_sample_rate // divisor
    resampled = resample_poly(samples, up, down)
    return [float(sample) for sample in resampled]


def choose_analysis_audio(
    samples: list[float],
    original_sample_rate: int,
    requested_sample_rate: int,
    resample_enabled: bool,
    method: str,
) -> tuple[list[float], int, bool]:
    if not resample_enabled or requested_sample_rate == original_sample_rate:
        return samples[:], original_sample_rate, False
    if method != "polyphase":
        raise ValueError("Only --resample-method polyphase is currently supported")
    return (
        resample_samples_polyphase(samples, original_sample_rate, requested_sample_rate),
        requested_sample_rate,
        True,
    )


def clamp_frequency_settings(
    analysis_sample_rate: int,
    min_frequency_hz: float,
    max_frequency_hz: float | None,
) -> tuple[float, float | None, list[str]]:
    warnings = []
    nyquist_hz = analysis_sample_rate / 2
    if max_frequency_hz is not None and max_frequency_hz > nyquist_hz:
        warnings.append(
            f"max_frequency_hz {max_frequency_hz:.1f} is above Nyquist "
            f"{nyquist_hz:.1f}; clamped to {nyquist_hz:.1f} Hz."
        )
        max_frequency_hz = nyquist_hz
    if max_frequency_hz is not None and analysis_sample_rate < 2.5 * max_frequency_hz:
        warnings.append(
            f"analysis sample rate {analysis_sample_rate} Hz is low for "
            f"max_frequency_hz {max_frequency_hz:.1f}; consider at least "
            f"{math.ceil(2.5 * max_frequency_hz)} Hz."
        )
    if min_frequency_hz >= nyquist_hz:
        warnings.append(
            f"min_frequency_hz {min_frequency_hz:.1f} is at or above Nyquist; "
            "lowering it to keep at least one analyzable bin."
        )
        min_frequency_hz = max(0.0, nyquist_hz / 10)
    return min_frequency_hz, max_frequency_hz, warnings


def write_wav_from_mono(
    samples: list[float],
    sample_rate: int,
    channels: int,
    output_path: Path,
) -> None:
    """Write normalized mono samples as 16-bit PCM WAV, duplicating channels if needed."""
    max_abs = max((abs(sample) for sample in samples), default=0.0)
    gain = 0.98 / max_abs if max_abs > 0.98 else 1.0

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for sample in samples:
            value = int(max(-1.0, min(1.0, sample * gain)) * 32767)
            sample_bytes = value.to_bytes(2, "little", signed=True)
            for _ in range(channels):
                frames.extend(sample_bytes)

        wav_file.writeframes(bytes(frames))


def hann_window(size: int) -> list[float]:
    return [
        0.5 - 0.5 * math.cos(2 * math.pi * index / (size - 1))
        for index in range(size)
    ]


def power_of_two_at_most(value: int) -> int:
    if value < 1:
        return 0
    return 1 << (value.bit_length() - 1)


def frame_starts(sample_count: int, fft_size: int, hop_size: int) -> list[int]:
    if sample_count <= fft_size:
        return [0]

    starts = list(range(0, sample_count - fft_size + 1, hop_size))
    final_start = sample_count - fft_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def frequency_bin_range(
    sample_rate: int,
    fft_size: int,
    min_frequency_hz: float,
    max_frequency_hz: float | None,
) -> range:
    min_bin = max(1, math.ceil(min_frequency_hz * fft_size / sample_rate))
    max_bin = fft_size // 2
    if max_frequency_hz is not None:
        max_bin = min(max_bin, math.floor(max_frequency_hz * fft_size / sample_rate))
    if max_bin < min_bin:
        return range(0)
    return range(min_bin, max_bin + 1)


def fft_bins_from_samples(
    samples: list[float],
    sample_rate: int,
    fft_size: int,
) -> list[tuple[float, float]]:
    """First-window real FFT using numpy.fft.rfft."""
    actual_fft_size = min(fft_size, power_of_two_at_most(len(samples)))
    if actual_fft_size < 2:
        return []

    block = samples[:actual_fft_size]
    if len(block) < actual_fft_size:
        block = block + [0.0] * (actual_fft_size - len(block))

    windowed = np.asarray(block, dtype=float) * np.hanning(actual_fft_size)
    spectrum = np.fft.rfft(windowed)
    frequencies = np.fft.rfftfreq(actual_fft_size, d=1 / sample_rate)
    return [
        (float(frequency), float(magnitude))
        for frequency, magnitude in zip(frequencies, np.abs(spectrum))
    ]


def analyze_dominant_frequency_over_time(
    wav_path: Path,
    fft_size: int,
    hop_size: int | None = None,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float | None = None,
) -> dict[str, object]:
    """Analyze dominant frequencies across the whole file.

    First-window FFT looks at only the beginning of a recording. This function
    instead uses overlapping Hann-windowed FFT frames across the full file,
    then averages each FFT bin's magnitude over all frames. The global dominant
    frequency is the bin with the largest mean magnitude.
    """
    samples, sample_rate, channels = read_mono_wav(wav_path)
    return analyze_samples_over_time(
        wav_path.name,
        samples,
        sample_rate,
        channels,
        fft_size,
        hop_size,
        min_frequency_hz,
        max_frequency_hz,
    )


def analyze_samples_over_time(
    name: str,
    samples: list[float],
    sample_rate: int,
    channels: int,
    fft_size: int,
    hop_size: int | None = None,
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float | None = None,
) -> dict[str, object]:
    """Analyze dominant frequencies across a sample array."""
    duration = len(samples) / sample_rate if sample_rate else 0.0
    actual_fft_size = min(fft_size, power_of_two_at_most(len(samples)))
    if actual_fft_size < 2:
        actual_fft_size = 2
    hop_size = hop_size or actual_fft_size // 2
    hop_size = max(1, min(hop_size, actual_fft_size))

    starts = frame_starts(len(samples), actual_fft_size, hop_size)
    window = np.hanning(actual_fft_size)
    bin_range = frequency_bin_range(
        sample_rate,
        actual_fft_size,
        min_frequency_hz,
        max_frequency_hz,
    )
    frequencies = np.fft.rfftfreq(actual_fft_size, d=1 / sample_rate)
    magnitude_sums = np.zeros(len(frequencies), dtype=float)
    frames: list[dict[str, float]] = []

    for start in starts:
        block = samples[start : start + actual_fft_size]
        if len(block) < actual_fft_size:
            block = block + [0.0] * (actual_fft_size - len(block))

        spectrum = np.fft.rfft(np.asarray(block, dtype=float) * window)
        magnitudes = np.abs(spectrum)
        magnitude_sums += magnitudes
        frame_best_frequency = 0.0
        frame_best_magnitude = 0.0

        for bin_index in bin_range:
            if bin_index >= len(magnitudes):
                continue
            magnitude = float(magnitudes[bin_index])
            if bin_index in bin_range and magnitude > frame_best_magnitude:
                frame_best_frequency = float(frequencies[bin_index])
                frame_best_magnitude = magnitude

        frames.append(
            {
                "time_seconds": (start + actual_fft_size / 2) / sample_rate,
                "dominant_frequency_hz": frame_best_frequency,
                "dominant_magnitude": frame_best_magnitude,
            }
        )

    frames_analyzed = max(1, len(starts))
    average_spectrum = [
        (float(frequency), float(magnitude / frames_analyzed))
        for frequency, magnitude in zip(frequencies, magnitude_sums)
    ]
    candidate_bins = [
        (frequency, magnitude)
        for index, (frequency, magnitude) in enumerate(average_spectrum)
        if index in bin_range
    ]
    global_frequency, global_magnitude = max(
        candidate_bins,
        key=lambda item: item[1],
        default=(0.0, 0.0),
    )
    frame_frequencies = [frame["dominant_frequency_hz"] for frame in frames]
    strongest_frame = max(frames, key=lambda frame: frame["dominant_magnitude"], default={})

    return {
        "file": name,
        "sample_rate_hz": sample_rate,
        "channels": channels,
        "duration_seconds": duration,
        "fft_size": actual_fft_size,
        "hop_size": hop_size,
        "min_frequency_hz": min_frequency_hz,
        "max_frequency_hz": max_frequency_hz,
        "number_of_frames_analyzed": len(frames),
        "global_dominant_frequency_hz": global_frequency,
        "global_dominant_magnitude": global_magnitude,
        "median_frame_dominant_frequency_hz": statistics.median(frame_frequencies) if frame_frequencies else 0.0,
        "mean_frame_dominant_frequency_hz": statistics.fmean(frame_frequencies) if frame_frequencies else 0.0,
        "std_frame_dominant_frequency_hz": statistics.pstdev(frame_frequencies) if len(frame_frequencies) > 1 else 0.0,
        "strongest_frame_dominant_frequency_hz": float(strongest_frame.get("dominant_frequency_hz", 0.0)),
        "strongest_frame_time_seconds": float(strongest_frame.get("time_seconds", 0.0)),
        "strongest_frame_magnitude": float(strongest_frame.get("dominant_magnitude", 0.0)),
        "frames": frames,
        "average_spectrum": average_spectrum,
        "samples": samples,
    }


def dominant_frequency(bins: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the strongest non-DC bin from a spectrum."""
    if len(bins) < 2:
        return 0.0, 0.0
    return max(bins[1:], key=lambda item: item[1])


def smoothed_magnitudes(magnitudes: list[float], window_size: int = SMOOTHING_BINS) -> list[float]:
    half_window = window_size // 2
    smoothed = []

    for index in range(len(magnitudes)):
        start = max(0, index - half_window)
        end = min(len(magnitudes), index + half_window + 1)
        smoothed.append(sum(magnitudes[start:end]) / (end - start))

    return smoothed


def relevant_frequency_cutoff(
    bins: list[tuple[float, float]],
    max_search_hz: float = MAX_SPECTRUM_HZ,
    minimum_hz: float = RELEVANT_MIN_HZ,
    threshold_fraction: float = RELEVANT_THRESHOLD_FRACTION,
) -> float:
    """Estimate where the useful average spectrum has died off."""
    visible_bins = [(frequency, magnitude) for frequency, magnitude in bins[1:] if frequency <= max_search_hz]
    if not visible_bins:
        return minimum_hz

    frequencies = [frequency for frequency, _ in visible_bins]
    smoothed = smoothed_magnitudes([magnitude for _, magnitude in visible_bins])
    threshold = max(smoothed) * threshold_fraction
    cutoff = max(
        (frequency for frequency, magnitude in zip(frequencies, smoothed) if magnitude >= threshold),
        default=minimum_hz,
    )

    return min(max(cutoff, minimum_hz), max_search_hz)


def detect_useful_frequency_range(
    spectrum_bins: list[tuple[float, float]],
    min_frequency_hz: float = 20.0,
    max_frequency_hz: float | None = None,
    threshold_fraction: float = 0.05,
    padding_fraction: float = 0.25,
    minimum_span_hz: float = 100.0,
    hard_plot_ceiling_hz: float = MAX_SPECTRUM_HZ,
) -> dict[str, float | str | bool]:
    """Find the frequency band that contains visually useful energy."""
    valid_bins = [
        (frequency, magnitude)
        for frequency, magnitude in spectrum_bins
        if frequency >= min_frequency_hz
        and frequency > 0
        and (max_frequency_hz is None or frequency <= max_frequency_hz)
    ]
    if not valid_bins:
        return {
            "useful_min_frequency_hz": 0.0,
            "useful_max_frequency_hz": min(hard_plot_ceiling_hz, minimum_span_hz),
            "peak_frequency_hz": 0.0,
            "peak_magnitude": 0.0,
            "low_frequency_dominant": False,
            "reason": "no_valid_frequency_bins",
        }

    peak_frequency_hz, peak_magnitude = max(valid_bins, key=lambda item: item[1])
    if peak_magnitude <= SILENCE_MAGNITUDE:
        return {
            "useful_min_frequency_hz": 0.0,
            "useful_max_frequency_hz": min(hard_plot_ceiling_hz, minimum_span_hz),
            "peak_frequency_hz": peak_frequency_hz,
            "peak_magnitude": peak_magnitude,
            "low_frequency_dominant": peak_frequency_hz < 100,
            "reason": "no_strong_frequency_detected",
        }

    threshold = peak_magnitude * threshold_fraction
    useful_frequencies = [
        frequency for frequency, magnitude in valid_bins
        if magnitude >= threshold
    ]
    raw_min = min(useful_frequencies)
    raw_max = max(useful_frequencies)
    span = max(raw_max - raw_min, minimum_span_hz)
    padded_min = max(0.0, raw_min - span * padding_fraction)
    padded_max = raw_max + span * padding_fraction

    if padded_max - padded_min < minimum_span_hz:
        center = (padded_min + padded_max) / 2
        padded_min = max(0.0, center - minimum_span_hz / 2)
        padded_max = padded_min + minimum_span_hz

    ceiling = hard_plot_ceiling_hz
    if max_frequency_hz is not None:
        ceiling = min(ceiling, max_frequency_hz)
    if padded_max <= ceiling:
        useful_max = min(ceiling, padded_max)
    else:
        useful_max = padded_max

    useful_min = 0.0 if useful_max <= 250 else padded_min
    reason = "low_frequency_dominant" if raw_max < 100 or peak_frequency_hz < 100 else "useful_energy_band_detected"
    return {
        "useful_min_frequency_hz": useful_min,
        "useful_max_frequency_hz": max(useful_max, minimum_span_hz),
        "peak_frequency_hz": peak_frequency_hz,
        "peak_magnitude": peak_magnitude,
        "low_frequency_dominant": raw_max < 100 or peak_frequency_hz < 100,
        "reason": reason,
    }


def hz_for_filename(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.001:
        return str(int(rounded))
    return f"{value:.1f}".replace(".", "p")


def zoom_needed(useful_max_hz: float, full_max_hz: float) -> bool:
    return full_max_hz > 0 and useful_max_hz < full_max_hz * ZOOM_TRIGGER_FRACTION


def audit_row(
    source_file: str,
    output_type: str,
    output_file: Path,
    full_y_min_hz: float,
    full_y_max_hz: float,
    useful_range: dict[str, float | str | bool],
    dominant_frequency_hz: float,
    recommended_min_frequency_hz: float,
    zoomed_version_created: bool,
    filtered: bool = False,
) -> dict[str, str | float | bool]:
    warnings = []
    recommendations = []
    useful_max = float(useful_range["useful_max_frequency_hz"])
    peak_magnitude = float(useful_range["peak_magnitude"])

    if dominant_frequency_hz < recommended_min_frequency_hz:
        warnings.append("dominant frequency below min recommended frequency")
        warnings.append("low frequency may be rumble/artifact")
        recommendations.append("Inspect zoomed low-frequency plots and consider whether vibration, wind, or handling noise is expected.")
    if zoom_needed(useful_max, full_y_max_hz):
        warnings.append("plot y-axis too wide for visible energy")
        recommendations.append("Use the zoomed plot for visual inspection.")
    if peak_magnitude <= SILENCE_MAGNITUDE:
        warnings.append("no strong frequency detected")
        recommendations.append("Check whether the source audio is silent or extremely low level.")
    if filtered:
        warnings.append("dominant-sine debug file should show a narrow horizontal band")
        useful_span = useful_max - float(useful_range["useful_min_frequency_hz"])
        if useful_span > max(200.0, dominant_frequency_hz):
            warnings.append("filtered energy spread wider than expected")
            recommendations.append("Filtered output may not be isolating a single tone cleanly.")
        else:
            recommendations.append("Sparse spectrogram is expected for dominant-sine debug audio.")

    return {
        "source_file": source_file,
        "output_type": output_type,
        "output_file": str(output_file),
        "full_y_min_hz": full_y_min_hz,
        "full_y_max_hz": full_y_max_hz,
        "useful_min_frequency_hz": float(useful_range["useful_min_frequency_hz"]),
        "useful_max_frequency_hz": useful_max,
        "peak_frequency_hz": float(useful_range["peak_frequency_hz"]),
        "peak_magnitude": peak_magnitude,
        "dominant_frequency_hz": dominant_frequency_hz,
        "low_frequency_dominant": bool(useful_range["low_frequency_dominant"]),
        "zoomed_version_created": zoomed_version_created,
        "warning": "; ".join(dict.fromkeys(warnings)),
        "recommendation": " ".join(dict.fromkeys(recommendations)),
    }


def filter_to_single_frequency(
    samples: list[float],
    sample_rate: int,
    target_frequency_hz: float,
    block_size: int = FILTER_BLOCK_SIZE,
) -> list[float]:
    """Reconstruct audio using only one target frequency with a moving amplitude estimate."""
    if not samples or target_frequency_hz <= 0:
        return [0.0] * len(samples)

    block_size = min(block_size, power_of_two_at_most(len(samples)))
    if block_size < 2:
        return [0.0] * len(samples)

    hop_size = block_size // 2
    window = hann_window(block_size)
    output = [0.0] * len(samples)
    weights = [0.0] * len(samples)
    angular_frequency = 2 * math.pi * target_frequency_hz / sample_rate

    for start in range(0, len(samples), hop_size):
        block = samples[start : start + block_size]
        if len(block) < block_size:
            block = block + [0.0] * (block_size - len(block))

        cos_projection = 0.0
        sin_projection = 0.0
        weight_total = 0.0

        for index, sample in enumerate(block):
            absolute_index = start + index
            angle = angular_frequency * absolute_index
            weight = window[index]
            cos_projection += sample * weight * math.cos(angle)
            sin_projection += sample * weight * math.sin(angle)
            weight_total += weight

        if weight_total == 0:
            continue

        cos_amplitude = 2 * cos_projection / weight_total
        sin_amplitude = 2 * sin_projection / weight_total

        for index in range(block_size):
            output_index = start + index
            if output_index >= len(samples):
                break

            angle = angular_frequency * output_index
            weight = window[index]
            value = cos_amplitude * math.cos(angle) + sin_amplitude * math.sin(angle)
            output[output_index] += value * weight
            weights[output_index] += weight

        if start + block_size >= len(samples):
            break

    for index, weight in enumerate(weights):
        if weight:
            output[index] /= weight

    return output


def filter_frequency_band(
    samples: list[float],
    sample_rate: int,
    low_cutoff_hz: float | None,
    high_cutoff_hz: float | None,
    notch_frequency_hz: float = 0.0,
    notch_harmonics: int = 0,
) -> list[float]:
    """Preserve a frequency band while suppressing low hum/rumble."""
    if not samples:
        return []

    try:
        from scipy.signal import butter, filtfilt, iirnotch, sosfilt, sosfiltfilt
    except ImportError as exc:
        raise RuntimeError("Band filtering requires scipy.") from exc

    nyquist_hz = sample_rate / 2
    low = low_cutoff_hz if low_cutoff_hz and low_cutoff_hz > 0 else None
    high = high_cutoff_hz if high_cutoff_hz and high_cutoff_hz < nyquist_hz else None
    filtered = samples[:]

    if low is not None and high is not None and low >= high:
        raise ValueError("Band filter low cutoff must be lower than high cutoff")

    if low is not None or high is not None:
        if low is not None and high is not None:
            wn: float | list[float] = [low / nyquist_hz, high / nyquist_hz]
            btype = "bandpass"
        elif low is not None:
            wn = low / nyquist_hz
            btype = "highpass"
        else:
            wn = high / nyquist_hz  # type: ignore[operator]
            btype = "lowpass"
        sos = butter(6, wn, btype=btype, output="sos")
        filtered = list(sosfiltfilt(sos, filtered) if len(filtered) > 64 else sosfilt(sos, filtered))

    if notch_frequency_hz > 0 and notch_harmonics > 0:
        for harmonic in range(1, notch_harmonics + 1):
            frequency_hz = notch_frequency_hz * harmonic
            if frequency_hz >= nyquist_hz:
                break
            b, a = iirnotch(frequency_hz / nyquist_hz, 30)
            filtered = list(filtfilt(b, a, filtered) if len(filtered) > 64 else filtered)

    return [float(sample) for sample in filtered]


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}">{escape(text)}</text>'
    )


def waveform_trace(
    samples: list[float],
    sample_rate: int,
    point_count: int = WAVEFORM_POINTS,
) -> tuple[list[tuple[float, float]], float]:
    duration = len(samples) / sample_rate if sample_rate else 0.0
    bucket_size = max(1, math.ceil(len(samples) / point_count))
    points = []

    for start in range(0, len(samples), bucket_size):
        bucket = samples[start : start + bucket_size]
        if bucket:
            midpoint = len(bucket) // 2
            points.append(((start + midpoint) / sample_rate, bucket[midpoint]))

    return points, duration


def write_waveform_svg_from_samples(
    samples: list[float],
    sample_rate: int,
    title: str,
    output_path: Path,
) -> None:
    points, duration = waveform_trace(samples, sample_rate)
    width = 1000
    height = 420
    left = 112
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    center_y = top + plot_height / 2
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Waveform: {title}", 20),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Time (seconds)", 13),
        svg_text(32, top + plot_height / 2, "Amplitude", 13),
    ]

    for tick in range(6):
        time_value = duration * tick / 5 if duration else 0.0
        x = left + (time_value / duration) * plot_width if duration else left
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#cbd5e1"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{time_value:.1f}", 11))

    for amplitude in (-1.0, -0.5, 0.0, 0.5, 1.0):
        y = center_y - amplitude * (plot_height / 2)
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{amplitude:.1f}", 11, "end"))

    waveform_points = []
    for time_seconds, amplitude in points:
        x = left + (time_seconds / duration) * plot_width if duration else left
        y = center_y - amplitude * (plot_height / 2)
        waveform_points.append(f"{x:.1f},{y:.1f}")

    elements.append(f'<line x1="{left}" y1="{center_y:.1f}" x2="{width - right}" y2="{center_y:.1f}" stroke="#64748b" stroke-width="1"/>')
    if waveform_points:
        elements.append(f'<polyline points="{" ".join(waveform_points)}" fill="none" stroke="#2563eb" stroke-width="1.25"/>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_spectrum_svg_from_bins(
    title: str,
    bins: list[tuple[float, float]],
    fft_size: int,
    output_path: Path,
    max_spectrum_hz: float = MAX_SPECTRUM_HZ,
    subtitle: str = "",
    zoomed: bool = False,
) -> None:
    frequency_limit = max(1.0, max_spectrum_hz)
    visible_bins = [item for item in bins if item[0] <= frequency_limit]
    max_magnitude = max((magnitude for _, magnitude in visible_bins), default=1.0)
    max_magnitude = max(max_magnitude, 1e-12)
    peak_frequency_hz, _ = dominant_frequency(visible_bins)
    width = 1000
    height = 420
    left = 112
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Average FFT Spectrum: {title}", 20),
        svg_text(
            width / 2,
            52,
            f"{'Zoomed to useful frequency range. ' if zoomed else ''}FFT size {fft_size}, peak {peak_frequency_hz:.2f} Hz, range 0-{frequency_limit:.0f} Hz{(' - ' + subtitle) if subtitle else ''}",
            12,
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Frequency (Hz)", 13),
        svg_text(32, top + plot_height / 2, "Mean magnitude", 13),
    ]

    for tick in range(6):
        frequency = frequency_limit * tick / 5
        x = left + (frequency / frequency_limit) * plot_width
        magnitude = max_magnitude * tick / 5
        y = top + plot_height - (magnitude / max_magnitude) * plot_height
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#cbd5e1"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{frequency:.0f}", 11))
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{magnitude:.1f}", 11, "end"))

    points = []
    for frequency, magnitude in visible_bins:
        x = left + (frequency / frequency_limit) * plot_width
        y = top + plot_height - (magnitude / max_magnitude) * plot_height
        points.append(f"{x:.1f},{y:.1f}")

    if points:
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#0f766e" stroke-width="1.5"/>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_db_spectrum_svg_from_bins(
    title: str,
    bins: list[tuple[float, float]],
    fft_size: int,
    output_path: Path,
    max_spectrum_hz: float = MAX_SPECTRUM_HZ,
    subtitle: str = "",
) -> None:
    db_bins = [
        (frequency, 20 * math.log10(max(magnitude, 1e-12)))
        for frequency, magnitude in bins
    ]
    frequency_limit = max(1.0, max_spectrum_hz)
    visible_bins = [item for item in db_bins if item[0] <= frequency_limit]
    min_db = min((magnitude for _, magnitude in visible_bins), default=-120.0)
    max_db = max((magnitude for _, magnitude in visible_bins), default=0.0)
    if max_db - min_db < 1:
        min_db = max_db - 1
    width = 1000
    height = 420
    left = 112
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Average FFT Spectrum dB: {title}", 20),
        svg_text(
            width / 2,
            52,
            f"FFT size {fft_size}, range 0-{frequency_limit:.0f} Hz{(' - ' + subtitle) if subtitle else ''}",
            12,
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Frequency (Hz)", 13),
        svg_text(32, top + plot_height / 2, "dB magnitude", 13),
    ]

    for tick in range(6):
        frequency = frequency_limit * tick / 5
        db_value = min_db + (max_db - min_db) * tick / 5
        x = left + (frequency / frequency_limit) * plot_width
        y = top + plot_height - ((db_value - min_db) / (max_db - min_db)) * plot_height
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#cbd5e1"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{frequency:.0f}", 11))
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{db_value:.1f}", 11, "end"))

    points = []
    for frequency, db_value in visible_bins:
        x = left + (frequency / frequency_limit) * plot_width
        y = top + plot_height - ((db_value - min_db) / (max_db - min_db)) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    if points:
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#be123c" stroke-width="1.5"/>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_spectrum_variability_svg(
    title: str,
    frequency_bins: list[float],
    center_values: list[float],
    low_values: list[float],
    high_values: list[float],
    output_path: Path,
    max_spectrum_hz: float,
    subtitle: str = "",
) -> None:
    triples = [
        (frequency, center, low, high)
        for frequency, center, low, high in zip(frequency_bins, center_values, low_values, high_values)
        if frequency <= max_spectrum_hz
    ]
    if not triples:
        triples = [(0.0, 0.0, 0.0, 0.0)]
    max_magnitude = max((high for _, _, _, high in triples), default=1.0)
    max_magnitude = max(max_magnitude, 1e-12)
    width = 1000
    height = 420
    left = 112
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    frequency_limit = max(1.0, max_spectrum_hz)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Segment Spectrum Variability: {title}", 20),
        svg_text(width / 2, 52, f"Numerical segment spectra averaged; range 0-{frequency_limit:.0f} Hz{(' - ' + subtitle) if subtitle else ''}", 12),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Frequency (Hz)", 13),
        svg_text(32, top + plot_height / 2, "Mean magnitude", 13),
    ]
    upper = []
    lower = []
    center = []
    for frequency, center_value, low, high in triples:
        x = left + (frequency / frequency_limit) * plot_width
        upper.append(f"{x:.1f},{top + plot_height - (high / max_magnitude) * plot_height:.1f}")
        lower.append(f"{x:.1f},{top + plot_height - (low / max_magnitude) * plot_height:.1f}")
        center.append(f"{x:.1f},{top + plot_height - (center_value / max_magnitude) * plot_height:.1f}")
    if upper and lower:
        elements.append(f'<polygon points="{" ".join(upper + list(reversed(lower)))}" fill="#99f6e4" opacity="0.65"/>')
    if center:
        elements.append(f'<polyline points="{" ".join(center)}" fill="none" stroke="#0f766e" stroke-width="1.6"/>')
    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def pyplot_module():
    os.environ.setdefault("MPLCONFIGDIR", str(Path("/private/tmp") / "frequency_analyzer_matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_waveform_png_from_samples(
    samples: list[float],
    sample_rate: int,
    title: str,
    output_path: Path,
) -> None:
    plt = pyplot_module()
    points, duration = waveform_trace(samples, sample_rate)
    times = [time for time, _ in points]
    amplitudes = [amplitude for _, amplitude in points]
    figure, axis = plt.subplots(figsize=(10, 4.2), dpi=140)
    axis.plot(times, amplitudes, color="#2563eb", linewidth=0.9)
    axis.axhline(0, color="#64748b", linewidth=0.8)
    axis.set_title(f"Waveform: {title}")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, max(duration, 0.001))
    axis.set_ylim(-1.05, 1.05)
    axis.grid(True, color="#cbd5e1", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_spectrum_png_from_bins(
    title: str,
    bins: list[tuple[float, float]],
    output_path: Path,
    max_spectrum_hz: float,
    db_scale: bool = False,
    subtitle: str = "",
) -> None:
    plt = pyplot_module()
    visible_bins = [(frequency, magnitude) for frequency, magnitude in bins if frequency <= max_spectrum_hz]
    frequencies = [frequency for frequency, _ in visible_bins]
    if db_scale:
        magnitudes = [20 * math.log10(max(magnitude, 1e-12)) for _, magnitude in visible_bins]
        ylabel = "dB magnitude"
        color = "#be123c"
    else:
        magnitudes = [magnitude for _, magnitude in visible_bins]
        ylabel = "Mean magnitude"
        color = "#0f766e"
    figure, axis = plt.subplots(figsize=(10, 4.2), dpi=140)
    axis.plot(frequencies, magnitudes, color=color, linewidth=1.1)
    axis.set_title(f"Average FFT Spectrum: {title}")
    if subtitle:
        axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel(ylabel)
    axis.set_xlim(0, max(1.0, max_spectrum_hz))
    axis.grid(True, color="#cbd5e1", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_spectrogram_png_from_samples(
    samples: list[float],
    sample_rate: int,
    title: str,
    output_path: Path,
    fft_size: int,
    max_frequency_hz: float,
    subtitle: str = "",
) -> None:
    plt = pyplot_module()
    cells, duration, frequency_limit, actual_fft_size = spectrogram_cells_from_samples(
        samples,
        sample_rate,
        fft_size,
        max_frequency_hz,
    )
    data = [
        [math.log10(1 + value) for value in row]
        for row in zip(*cells)
    ] if cells else [[0.0]]
    figure, axis = plt.subplots(figsize=(10, 5.6), dpi=140)
    image = axis.imshow(
        data,
        origin="lower",
        aspect="auto",
        extent=[0, max(duration, 0.001), 0, frequency_limit],
        cmap="viridis",
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Relative magnitude (log scale)")
    axis.set_title(f"Frequency vs Time: {title}")
    if subtitle:
        axis.text(
            0.5,
            1.01,
            f"Spectrogram FFT size {actual_fft_size}; {subtitle}",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_dominant_over_time_png(
    title: str,
    frames: list[dict[str, float]],
    output_path: Path,
    y_max_hz: float,
    subtitle: str = "",
) -> None:
    plt = pyplot_module()
    times = [frame["time_seconds"] for frame in frames]
    frequencies = [frame["dominant_frequency_hz"] for frame in frames]
    figure, axis = plt.subplots(figsize=(10, 4.2), dpi=140)
    axis.plot(times, frequencies, color="#7c3aed", linewidth=1.1)
    axis.set_title(f"Dominant Frequency Over Time: {title}")
    if subtitle:
        axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Hz")
    axis.set_ylim(0, max(1.0, y_max_hz))
    axis.grid(True, color="#cbd5e1", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def save_spectrum_variability_png(
    title: str,
    frequency_bins: list[float],
    center_values: list[float],
    low_values: list[float],
    high_values: list[float],
    output_path: Path,
    max_spectrum_hz: float,
    subtitle: str = "",
) -> None:
    plt = pyplot_module()
    visible = [
        (frequency, center, low, high)
        for frequency, center, low, high in zip(frequency_bins, center_values, low_values, high_values)
        if frequency <= max_spectrum_hz
    ]
    frequencies = [item[0] for item in visible]
    center = [item[1] for item in visible]
    low = [item[2] for item in visible]
    high = [item[3] for item in visible]
    figure, axis = plt.subplots(figsize=(10, 4.2), dpi=140)
    axis.fill_between(frequencies, low, high, color="#99f6e4", alpha=0.7)
    axis.plot(frequencies, center, color="#0f766e", linewidth=1.1)
    axis.set_title(f"Segment Spectrum Variability: {title}")
    if subtitle:
        axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Mean magnitude")
    axis.set_xlim(0, max(1.0, max_spectrum_hz))
    axis.grid(True, color="#cbd5e1", linewidth=0.5)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def heat_color(value: float) -> str:
    """Map a normalized 0..1 value to a dark-blue through yellow heat color."""
    value = max(0.0, min(1.0, value))
    stops = [
        (15, 23, 42),
        (30, 64, 175),
        (20, 184, 166),
        (250, 204, 21),
    ]
    scaled = value * (len(stops) - 1)
    index = min(len(stops) - 2, int(scaled))
    fraction = scaled - index
    start = stops[index]
    end = stops[index + 1]
    red = round(start[0] + (end[0] - start[0]) * fraction)
    green = round(start[1] + (end[1] - start[1]) * fraction)
    blue = round(start[2] + (end[2] - start[2]) * fraction)
    return f"#{red:02x}{green:02x}{blue:02x}"


def spectrogram_cells_from_samples(
    samples: list[float],
    sample_rate: int,
    fft_size: int = SPECTROGRAM_FFT_SIZE,
    max_frequency_hz: float = MAX_SPECTRUM_HZ,
    band_count: int = SPECTROGRAM_BANDS,
    max_columns: int = SPECTROGRAM_COLUMNS,
) -> tuple[list[list[float]], float, float, int]:
    actual_fft_size = min(fft_size, power_of_two_at_most(len(samples)))
    if actual_fft_size < 2:
        actual_fft_size = 2
    frequency_limit = min(max_frequency_hz, sample_rate / 2)
    noverlap = actual_fft_size // 2
    frequencies, _times, spectrum = stft(
        np.asarray(samples, dtype=float),
        fs=sample_rate,
        window="hann",
        nperseg=actual_fft_size,
        noverlap=noverlap,
        nfft=actual_fft_size,
        boundary=None,
        padded=True,
    )
    magnitudes = np.abs(spectrum)
    visible_mask = (frequencies > 0) & (frequencies <= frequency_limit)
    visible_frequencies = frequencies[visible_mask]
    visible_magnitudes = magnitudes[visible_mask]
    if visible_magnitudes.size == 0:
        duration = len(samples) / sample_rate if sample_rate else 0.0
        return [], duration, frequency_limit, actual_fft_size

    column_count = visible_magnitudes.shape[1]
    column_step = max(1, math.ceil(column_count / max_columns))
    band_edges = np.linspace(0.0, frequency_limit, band_count + 1)
    cells: list[list[float]] = []

    for column_index in range(0, column_count, column_step):
        column = [0.0] * band_count
        for band_index in range(band_count):
            low = band_edges[band_index]
            high = band_edges[band_index + 1]
            if band_index == band_count - 1:
                mask = (visible_frequencies >= low) & (visible_frequencies <= high)
            else:
                mask = (visible_frequencies >= low) & (visible_frequencies < high)
            if np.any(mask):
                column[band_index] = float(np.mean(visible_magnitudes[mask, column_index]))
        cells.append(column)

    duration = len(samples) / sample_rate if sample_rate else 0.0
    return cells, duration, frequency_limit, actual_fft_size


def write_frequency_time_svg_from_samples(
    samples: list[float],
    sample_rate: int,
    title: str,
    output_path: Path,
    fft_size: int = SPECTROGRAM_FFT_SIZE,
    max_frequency_hz: float = MAX_SPECTRUM_HZ,
    subtitle: str = "",
    zoomed: bool = False,
) -> None:
    cells, duration, frequency_limit, actual_fft_size = spectrogram_cells_from_samples(
        samples,
        sample_rate,
        fft_size,
        max_frequency_hz,
    )
    width = 1000
    height = 560
    left = 112
    right = 34
    top = 72
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    column_count = max(1, len(cells))
    band_count = max(1, len(cells[0]) if cells else SPECTROGRAM_BANDS)
    cell_width = plot_width / column_count
    cell_height = plot_height / band_count
    max_magnitude = max((max(column) for column in cells), default=1.0)
    max_magnitude = max(max_magnitude, 1e-12)
    log_max = math.log10(1 + max_magnitude)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Frequency vs Time: {title}", 20),
        svg_text(
            width / 2,
            52,
            f"{'Zoomed to useful frequency range. ' if zoomed else ''}Spectrogram FFT size {actual_fft_size}, range 0-{frequency_limit:.0f} Hz{(' - ' + subtitle) if subtitle else ''}",
            12,
        ),
        f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="#0f172a"/>',
    ]

    for column_index, column in enumerate(cells):
        x = left + column_index * cell_width
        for band_index, magnitude in enumerate(column):
            y = top + plot_height - (band_index + 1) * cell_height
            normalized = math.log10(1 + magnitude) / log_max
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_width + 0.25:.2f}" height="{cell_height + 0.25:.2f}" fill="{heat_color(normalized)}"/>'
            )

    legend_width = 188
    legend_height = 58
    legend_x = width - right - legend_width - 12
    legend_y = top + 12
    bar_x = legend_x + 14
    bar_y = legend_y + 28
    bar_width = legend_width - 28
    bar_height = 12
    elements.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="{legend_width}" height="{legend_height}" '
        'fill="#f8fafc" opacity="0.92" stroke="#cbd5e1" rx="4"/>'
    )
    elements.append(svg_text(legend_x + legend_width / 2, legend_y + 18, "Color = relative magnitude", 11))
    for step in range(80):
        value = step / 79
        x = bar_x + value * bar_width
        elements.append(
            f'<rect x="{x:.2f}" y="{bar_y}" width="{bar_width / 79 + 0.5:.2f}" height="{bar_height}" fill="{heat_color(value)}"/>'
        )
    elements.append(svg_text(bar_x, legend_y + 54, "low", 10, "start"))
    elements.append(svg_text(bar_x + bar_width / 2, legend_y + 54, "log scale", 10))
    elements.append(svg_text(bar_x + bar_width, legend_y + 54, "high", 10, "end"))

    elements.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
            svg_text(width / 2, height - 12, "Time (seconds)", 13),
            svg_text(32, top + plot_height / 2, "Frequency (Hz)", 13),
        ]
    )

    for tick in range(6):
        time_value = duration * tick / 5 if duration else 0.0
        x = left + (time_value / duration) * plot_width if duration else left
        frequency = frequency_limit * tick / 5 if frequency_limit else 0.0
        y = top + plot_height - (frequency / frequency_limit) * plot_height if frequency_limit else top + plot_height
        elements.append(f'<line x1="{x:.1f}" y1="{height - bottom}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#334155"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{time_value:.1f}", 11))
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{left}" y2="{y:.1f}" stroke="#334155"/>')
        elements.append(svg_text(left - 12, y + 4, f"{frequency:.0f}", 11, "end"))

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_dominant_over_time_svg(
    title: str,
    frames: list[dict[str, float]],
    output_path: Path,
    y_max_hz: float | None = None,
    subtitle: str = "",
    zoomed: bool = False,
) -> None:
    width = 1000
    height = 420
    left = 112
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    duration = max((frame["time_seconds"] for frame in frames), default=0.0)
    max_frequency = max((frame["dominant_frequency_hz"] for frame in frames), default=1.0)
    y_max = y_max_hz or max(10.0, math.ceil(max_frequency / 100) * 100)
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Dominant Frequency Over Time: {title}", 20),
        svg_text(
            width / 2,
            52,
            f"{'Zoomed to useful frequency range. ' if zoomed else ''}Range 0-{y_max:.0f} Hz{(' - ' + subtitle) if subtitle else ''}",
            12,
        ),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Time (seconds)", 13),
        svg_text(32, top + plot_height / 2, "Hz", 13),
    ]

    for tick in range(6):
        time_value = duration * tick / 5 if duration else 0.0
        x = left + (time_value / duration) * plot_width if duration else left
        frequency = y_max * tick / 5
        y = height - bottom - (frequency / y_max) * plot_height
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#cbd5e1"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{time_value:.1f}", 11))
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{frequency:.0f}", 11, "end"))

    points = []
    for frame in frames:
        x = left + (frame["time_seconds"] / duration) * plot_width if duration else left
        y = height - bottom - (frame["dominant_frequency_hz"] / y_max) * plot_height
        points.append(f"{x:.1f},{y:.1f}")
    if points:
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#7c3aed" stroke-width="1.5"/>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_first_window_fft_csv(
    wav_path: Path,
    output_dir: Path,
    bins: list[tuple[float, float]],
    sample_rate: int,
    fft_size: int,
) -> None:
    output_path = output_dir / f"{wav_path.stem}_fft.csv"
    peak_frequency_hz, _ = dominant_frequency(bins)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_file", wav_path.name])
        writer.writerow(["sample_rate_hz", sample_rate])
        writer.writerow(["fft_size", fft_size])
        writer.writerow(["dominant_frequency_hz", f"{peak_frequency_hz:.6f}"])
        writer.writerow(["note", "first-window FFT only; dominant_frequencies.csv uses full-file average spectrum"])
        writer.writerow([])
        writer.writerow(["frequency_hz", "magnitude"])
        for frequency_hz, magnitude in bins:
            writer.writerow([f"{frequency_hz:.6f}", f"{magnitude:.12f}"])


def write_fft_bins_csv(
    wav_path: Path,
    output_dir: Path,
    bins: list[tuple[float, float]],
    sample_rate: int,
    fft_size: int,
    max_frequency_hz: float,
) -> None:
    output_path = output_dir / f"{wav_path.stem}_fft.csv"
    peak_frequency_hz, _ = dominant_frequency([item for item in bins if item[0] <= max_frequency_hz])
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_file", wav_path.name])
        writer.writerow(["sample_rate_hz", sample_rate])
        writer.writerow(["fft_size", fft_size])
        writer.writerow(["max_frequency_hz", f"{max_frequency_hz:.6f}"])
        writer.writerow(["dominant_frequency_hz", f"{peak_frequency_hz:.6f}"])
        writer.writerow([])
        writer.writerow(["frequency_hz", "magnitude"])
        for frequency_hz, magnitude in bins:
            if frequency_hz <= max_frequency_hz:
                writer.writerow([f"{frequency_hz:.6f}", f"{magnitude:.12f}"])


def write_average_fft_csv(
    wav_path: Path,
    output_dir: Path,
    analysis: dict[str, object],
) -> None:
    output_path = output_dir / f"{wav_path.stem}_average_fft.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_file", wav_path.name])
        writer.writerow(["sample_rate_hz", analysis["sample_rate_hz"]])
        writer.writerow(["fft_size", analysis["fft_size"]])
        writer.writerow(["hop_size", analysis["hop_size"]])
        writer.writerow(["analyzed_frames", analysis["number_of_frames_analyzed"]])
        writer.writerow(["min_frequency_hz", f"{float(analysis['min_frequency_hz']):.6f}"])
        max_frequency = analysis["max_frequency_hz"]
        writer.writerow(["max_frequency_hz", "" if max_frequency is None else f"{float(max_frequency):.6f}"])
        writer.writerow(["global_dominant_frequency_hz", f"{float(analysis['global_dominant_frequency_hz']):.6f}"])
        writer.writerow(["global_dominant_magnitude", f"{float(analysis['global_dominant_magnitude']):.12f}"])
        writer.writerow([])
        writer.writerow(["frequency_hz", "mean_magnitude"])
        for frequency_hz, magnitude in analysis["average_spectrum"]:  # type: ignore[index]
            writer.writerow([f"{frequency_hz:.6f}", f"{magnitude:.12f}"])


def write_dominant_over_time_csv(
    wav_path: Path,
    output_dir: Path,
    frames: list[dict[str, float]],
) -> None:
    output_path = output_dir / f"{wav_path.stem}_dominant_over_time.csv"
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["time_seconds", "dominant_frequency_hz", "dominant_magnitude"],
        )
        writer.writeheader()
        for frame in frames:
            writer.writerow(
                {
                    "time_seconds": f"{frame['time_seconds']:.6f}",
                    "dominant_frequency_hz": f"{frame['dominant_frequency_hz']:.6f}",
                    "dominant_magnitude": f"{frame['dominant_magnitude']:.12f}",
                }
            )


def write_dominant_csv(rows: list[dict[str, int | float | str]], output_path: Path) -> None:
    fieldnames = [
        "file",
        "dominant_frequency_hz",
        "dominant_magnitude",
        "dominant_method",
        "median_frame_dominant_frequency_hz",
        "mean_frame_dominant_frequency_hz",
        "std_frame_dominant_frequency_hz",
        "strongest_frame_dominant_frequency_hz",
        "strongest_frame_time_seconds",
        "duration_seconds",
        "sample_rate_hz",
        "channels",
        "fft_size",
        "hop_size",
        "frames_analyzed",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    **row,
                    "dominant_frequency_hz": f"{float(row['dominant_frequency_hz']):.6f}",
                    "dominant_magnitude": f"{float(row['dominant_magnitude']):.12f}",
                    "median_frame_dominant_frequency_hz": f"{float(row['median_frame_dominant_frequency_hz']):.6f}",
                    "mean_frame_dominant_frequency_hz": f"{float(row['mean_frame_dominant_frequency_hz']):.6f}",
                    "std_frame_dominant_frequency_hz": f"{float(row['std_frame_dominant_frequency_hz']):.6f}",
                    "strongest_frame_dominant_frequency_hz": f"{float(row['strongest_frame_dominant_frequency_hz']):.6f}",
                    "strongest_frame_time_seconds": f"{float(row['strongest_frame_time_seconds']):.6f}",
                    "duration_seconds": f"{float(row['duration_seconds']):.3f}",
                }
            )


def write_relevant_cutoffs_csv(rows: list[dict[str, int | float | str]], output_path: Path) -> None:
    fieldnames = [
        "file",
        "relevant_max_frequency_hz",
        "method",
        "minimum_hz",
        "threshold_fraction",
        "search_max_hz",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file": row["file"],
                    "relevant_max_frequency_hz": f"{float(row['relevant_max_frequency_hz']):.6f}",
                    "method": "smoothed full-file average magnitude threshold with minimum cutoff",
                    "minimum_hz": f"{RELEVANT_MIN_HZ:.6f}",
                    "threshold_fraction": f"{RELEVANT_THRESHOLD_FRACTION:.6f}",
                    "search_max_hz": f"{MAX_SPECTRUM_HZ:.6f}",
                }
            )


def write_filtered_audio_summary_csv(rows: list[dict[str, int | float | str]], output_path: Path) -> None:
    fieldnames = [
        "source_file",
        "mode",
        "kept_frequency_hz",
        "relevant_max_frequency_hz",
        "output_wav",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {
                    "source_file": row["source_file"],
                    "mode": row["mode"],
                    "kept_frequency_hz": f"{float(row['kept_frequency_hz']):.6f}",
                    "relevant_max_frequency_hz": (
                        ""
                        if row["relevant_max_frequency_hz"] == ""
                        else f"{float(row['relevant_max_frequency_hz']):.6f}"
                    ),
                    "output_wav": row["output_wav"],
                }
            )


def write_results_audit_csv(rows: list[dict[str, str | float | bool]], output_path: Path) -> None:
    fieldnames = [
        "source_file",
        "output_type",
        "output_file",
        "full_y_min_hz",
        "full_y_max_hz",
        "useful_min_frequency_hz",
        "useful_max_frequency_hz",
        "peak_frequency_hz",
        "peak_magnitude",
        "dominant_frequency_hz",
        "low_frequency_dominant",
        "zoomed_version_created",
        "warning",
        "recommendation",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_results_audit_md(rows: list[dict[str, str | float | bool]], output_path: Path) -> None:
    low_frequency_files = sorted({
        str(row["source_file"])
        for row in rows
        if row["low_frequency_dominant"]
    })
    zoomed_rows = [row for row in rows if row["zoomed_version_created"]]
    rumble_rows = [
        row for row in rows
        if "low frequency may be rumble/artifact" in str(row["warning"])
    ]
    filtered_rows = [
        row for row in rows
        if "filtered" in str(row["output_type"]) and "dominant-sine" in str(row["output_type"])
    ]

    lines = [
        "# Results Audit",
        "",
        "This audit is generated from the numerical FFT data behind each plot.",
        "",
        "## Low-Frequency Dominant Components",
    ]
    if low_frequency_files:
        for filename in low_frequency_files:
            lines.append(f"- `{filename}`")
    else:
        lines.append("- None detected.")

    lines.extend(["", "## Zoomed Plots Created"])
    if zoomed_rows:
        for row in zoomed_rows:
            lines.append(
                f"- `{row['output_file']}` zoomed to {float(row['useful_min_frequency_hz']):.1f}-{float(row['useful_max_frequency_hz']):.1f} Hz"
            )
    else:
        lines.append("- No zoomed plots were needed.")

    lines.extend(["", "## Possible Rumble Or Artifact"])
    if rumble_rows:
        seen = set()
        for row in rumble_rows:
            key = (row["source_file"], row["dominant_frequency_hz"])
            if key in seen:
                continue
            seen.add(key)
            lines.append(
                f"- `{row['source_file']}` dominant frequency {float(row['dominant_frequency_hz']):.2f} Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment."
            )
    else:
        lines.append("- None flagged below the recommended minimum frequency.")

    lines.extend(["", "## Dominant-Only Filtered Outputs"])
    if filtered_rows:
        lines.append(
            "- Dominant-only filtered WAV spectrograms are expected to look sparse, usually as a narrow horizontal band."
        )
        for row in filtered_rows:
            warning = str(row["warning"])
            if "filtered energy spread wider than expected" in warning:
                lines.append(f"- Check `{row['output_file']}`: filtered energy is wider than expected.")
    else:
        lines.append("- No filtered outputs audited.")

    lines.extend(["", "## Recommendations"])
    lines.append("- Use zoomed plots when the full y-axis hides low-frequency energy.")
    lines.append("- Treat dominant frequencies below 50 Hz with care unless low-frequency motion is expected.")
    lines.append("- See `results/results_audit.csv` for one row per audited output.")

    output_path.write_text("\n".join(lines) + "\n")


def write_dominant_svg(rows: list[dict[str, int | float | str]], output_path: Path) -> None:
    width = max(900, 115 * len(rows))
    height = 560
    left = 82
    right = 32
    top = 56
    bottom = 148
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_frequency = max(float(row["dominant_frequency_hz"]) for row in rows)
    y_max = max(10.0, math.ceil(max_frequency / 100) * 100)
    bar_gap = 18
    bar_width = max(30, (plot_width - bar_gap * (len(rows) + 1)) / len(rows))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 30, "Full-File Dominant Frequency by WAV File", 22),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(22, top + plot_height / 2, "Hz", 13),
    ]

    for tick in range(6):
        frequency = y_max * tick / 5
        y = height - bottom - (frequency / y_max) * plot_height
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{frequency:.0f}", 11, "end"))

    for index, row in enumerate(rows):
        frequency = float(row["dominant_frequency_hz"])
        x = left + bar_gap + index * (bar_width + bar_gap)
        bar_height = (frequency / y_max) * plot_height
        y = height - bottom - bar_height
        label_x = x + bar_width / 2
        elements.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#2563eb" rx="4"/>')
        elements.append(svg_text(label_x, y - 8, f"{frequency:.1f}", 11))
        elements.append(
            f'<text x="{label_x:.1f}" y="{height - bottom + 20}" font-size="11" '
            'font-family="Arial, sans-serif" text-anchor="end" '
            f'transform="rotate(-45 {label_x:.1f} {height - bottom + 20})">{escape(str(row["file"]))}</text>'
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def make_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def top_frequency_peaks(
    bins: list[tuple[float, float]],
    min_frequency_hz: float,
    max_frequency_hz: float | None,
    count: int = 5,
) -> list[dict[str, float]]:
    candidates = [
        (frequency, magnitude)
        for frequency, magnitude in bins
        if frequency >= min_frequency_hz
        and (max_frequency_hz is None or frequency <= max_frequency_hz)
    ]
    candidates.sort(key=lambda item: item[1], reverse=True)
    return [
        {"frequency_hz": frequency, "magnitude": magnitude}
        for frequency, magnitude in candidates[:count]
    ]


def is_hum_frequency(frequency_hz: float, hum_frequency_hz: float, harmonics: int, tolerance_hz: float = 3.0) -> bool:
    if hum_frequency_hz <= 0 or harmonics < 1:
        return False
    for harmonic in range(1, harmonics + 1):
        if abs(frequency_hz - hum_frequency_hz * harmonic) <= tolerance_hz:
            return True
    return False


def dominant_non_hum_frequency(
    bins: list[tuple[float, float]],
    min_frequency_hz: float,
    max_frequency_hz: float | None,
    hum_frequency_hz: float,
    harmonics: int,
) -> tuple[float, float]:
    candidates = [
        (frequency, magnitude)
        for frequency, magnitude in bins
        if frequency >= min_frequency_hz
        and (max_frequency_hz is None or frequency <= max_frequency_hz)
        and not is_hum_frequency(frequency, hum_frequency_hz, harmonics)
    ]
    return max(candidates, key=lambda item: item[1], default=(0.0, 0.0))


def segment_sample_ranges(
    sample_count: int,
    sample_rate: int,
    duration_seconds: float,
    overlap_seconds: float,
) -> list[tuple[int, int, float, float]]:
    segment_size = max(1, round(duration_seconds * sample_rate))
    overlap_size = max(0, round(overlap_seconds * sample_rate))
    hop_size = max(1, segment_size - overlap_size)
    ranges = []
    start = 0
    while start < sample_count:
        end = min(sample_count, start + segment_size)
        if end > start:
            ranges.append((start, end, start / sample_rate, end / sample_rate))
        if end == sample_count:
            break
        start += hop_size
    return ranges


def interpolate_spectrum(
    spectrum: list[tuple[float, float]],
    target_frequencies: list[float],
) -> list[float]:
    if not spectrum:
        return [0.0] * len(target_frequencies)
    source_frequencies = [frequency for frequency, _ in spectrum]
    source_magnitudes = [magnitude for _, magnitude in spectrum]
    values = []
    source_index = 0
    for target in target_frequencies:
        while source_index + 1 < len(source_frequencies) and source_frequencies[source_index + 1] < target:
            source_index += 1
        if target <= source_frequencies[0]:
            values.append(source_magnitudes[0])
        elif target >= source_frequencies[-1]:
            values.append(source_magnitudes[-1])
        else:
            left_frequency = source_frequencies[source_index]
            right_frequency = source_frequencies[source_index + 1]
            left_magnitude = source_magnitudes[source_index]
            right_magnitude = source_magnitudes[source_index + 1]
            fraction = (target - left_frequency) / (right_frequency - left_frequency)
            values.append(left_magnitude + (right_magnitude - left_magnitude) * fraction)
    return values


def percentile(sorted_values: list[float], fraction: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = fraction * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def write_segments_summary_csv(rows: list[dict[str, object]], output_path: Path) -> None:
    fieldnames = [
        "segment_index",
        "start_time_seconds",
        "end_time_seconds",
        "raw_dominant_frequency_hz",
        "dominant_non_hum_frequency_hz",
        "detected_hum_present",
        "mean_magnitude",
        "peak_magnitude",
    ]
    with output_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "segment_index": row["segment_index"],
                    "start_time_seconds": f"{float(row['start_time_seconds']):.6f}",
                    "end_time_seconds": f"{float(row['end_time_seconds']):.6f}",
                    "raw_dominant_frequency_hz": f"{float(row['raw_dominant_frequency_hz']):.6f}",
                    "dominant_non_hum_frequency_hz": f"{float(row['dominant_non_hum_frequency_hz']):.6f}",
                    "detected_hum_present": row["detected_hum_present"],
                    "mean_magnitude": f"{float(row['mean_magnitude']):.12f}",
                    "peak_magnitude": f"{float(row['peak_magnitude']):.12f}",
                }
            )


def clean_generated_outputs(path: Path) -> None:
    """Delete only generated analyzer outputs."""
    protected_names = {"", ".", "/", "Data", ".git"}
    if path.name in protected_names:
        raise ValueError(f"Refusing to clean protected path: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def dedupe_existing_results(path: Path, keep_csv: bool = False) -> dict[str, int]:
    """Remove redundant generated files according to the clean output policy."""
    removed = {"svg": 0, "csv": 0}
    if not path.exists():
        return removed
    for svg_path in path.rglob("*.svg"):
        png_peer = svg_path.with_suffix(".png")
        if png_peer.exists():
            svg_path.unlink()
            removed["svg"] += 1
    if not keep_csv:
        for csv_path in path.rglob("*.csv"):
            csv_path.unlink()
            removed["csv"] += 1
    return removed


def denoising_settings_text(
    highpass_cutoff_hz: float,
    notch_frequency_hz: float,
    notch_harmonics: int,
) -> str:
    parts = [f"high-pass {highpass_cutoff_hz:.0f} Hz"]
    if notch_frequency_hz > 0 and notch_harmonics > 0:
        parts.append(f"notch {notch_frequency_hz:.0f} Hz x {notch_harmonics} harmonics")
    else:
        parts.append("notch off")
    return ", ".join(parts)


def spectrogram_matrix_db(
    samples: list[float],
    sample_rate: int,
    fft_size: int,
    max_frequency_hz: float,
    enhanced: bool = False,
    noise_percentile: float = 20.0,
) -> tuple[list[list[float]], float, float, int]:
    cells, duration, frequency_limit, actual_fft_size = spectrogram_cells_from_samples(
        samples,
        sample_rate,
        fft_size,
        max_frequency_hz,
    )
    if not cells:
        return [[-120.0]], duration, frequency_limit, actual_fft_size
    band_count = len(cells[0])
    columns = len(cells)
    processed = [[cells[column][band] for column in range(columns)] for band in range(band_count)]
    if enhanced:
        for band_index, row in enumerate(processed):
            sorted_row = sorted(row)
            floor = percentile(sorted_row, max(0.0, min(1.0, noise_percentile / 100)))
            processed[band_index] = [max(0.0, value - floor) for value in row]
    return [
        [20 * math.log10(max(value, 1e-12)) for value in row]
        for row in processed
    ], duration, frequency_limit, actual_fft_size


def extract_pitch_contour(
    spectrogram_db: list[list[float]],
    duration: float,
    frequency_limit: float,
    min_pitch_frequency_hz: float,
    notch_frequency_hz: float,
    notch_harmonics: int,
) -> list[dict[str, float]]:
    if not spectrogram_db or not spectrogram_db[0]:
        return []
    band_count = len(spectrogram_db)
    column_count = len(spectrogram_db[0])
    contour = []
    for column_index in range(column_count):
        best_frequency = 0.0
        best_db = -1e9
        for band_index in range(band_count):
            frequency = (band_index + 0.5) * frequency_limit / band_count
            if frequency < min_pitch_frequency_hz:
                continue
            if is_hum_frequency(frequency, notch_frequency_hz, notch_harmonics):
                continue
            value = spectrogram_db[band_index][column_index]
            if value > best_db:
                best_db = value
                best_frequency = frequency
        contour.append(
            {
                "time_seconds": duration * column_index / max(1, column_count - 1),
                "dominant_frequency_hz": best_frequency,
                "dominant_magnitude": best_db,
            }
        )
    if len(contour) < 3:
        return contour
    smoothed = []
    for index, point in enumerate(contour):
        start = max(0, index - 1)
        end = min(len(contour), index + 2)
        values = [contour[item]["dominant_frequency_hz"] for item in range(start, end)]
        smoothed.append({**point, "dominant_frequency_hz": statistics.fmean(values)})
    return smoothed


def save_clean_spectrogram_png(
    samples: list[float],
    sample_rate: int,
    source_name: str,
    output_path: Path,
    title: str,
    fft_size: int,
    max_frequency_hz: float,
    dynamic_range_db: float,
    subtitle: str,
    enhanced: bool = False,
    noise_percentile: float = 20.0,
    pitch_contour: list[dict[str, float]] | None = None,
) -> tuple[list[list[float]], list[dict[str, float]], float]:
    plt = pyplot_module()
    matrix_db, duration, frequency_limit, _actual_fft_size = spectrogram_matrix_db(
        samples,
        sample_rate,
        fft_size,
        max_frequency_hz,
        enhanced,
        noise_percentile,
    )
    max_db = max((max(row) for row in matrix_db), default=0.0)
    min_db = max_db - dynamic_range_db
    figure, axis = plt.subplots(figsize=(10, 5.6), dpi=140)
    image = axis.imshow(
        matrix_db,
        origin="lower",
        aspect="auto",
        extent=[0, max(duration, 0.001), 0, frequency_limit],
        cmap="viridis",
        vmin=min_db,
        vmax=max_db,
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("Magnitude (dB)")
    if pitch_contour:
        times = [point["time_seconds"] for point in pitch_contour]
        frequencies = [point["dominant_frequency_hz"] for point in pitch_contour]
        axis.plot(times, frequencies, color="#ffffff", linewidth=1.4)
        axis.plot(times, frequencies, color="#ef4444", linewidth=0.7)
    axis.set_title(f"{title} - {source_name}")
    axis.text(0.5, 1.01, subtitle, transform=axis.transAxes, ha="center", va="bottom", fontsize=8)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return matrix_db, pitch_contour or [], frequency_limit


def save_pitch_contour_png(
    source_name: str,
    contour: list[dict[str, float]],
    output_path: Path,
    max_frequency_hz: float,
    subtitle: str,
) -> None:
    save_dominant_over_time_png(
        source_name,
        contour,
        output_path,
        max_frequency_hz,
        subtitle,
    )


def write_clean_summary_md(summary: dict[str, object], output_path: Path) -> None:
    lines = [
        f"# Audio Frequency Analysis: {summary['file']}",
        "",
        "## Default Outputs",
        "- `waveform.png`: original waveform.",
        "- `spectrogram_raw.png`: raw analysis spectrogram.",
        "- `spectrogram_denoised.png`: high-pass/notch filtered spectrogram.",
        "- `spectrogram_enhanced_pitch.png`: noise-floor suppressed spectrogram with pitch contour.",
        "- `average_spectrum_db.png`: denoised average spectrum.",
        "- `pitch_contour.png`: extracted pitch contour over time.",
        "- `audio/denoised_filtered.wav`: denoised audio that preserves the recording while reducing noise.",
        "- `audio/isolated_pitch_band.wav`: isolated audio around the selected non-hum pitch band.",
        "- `audio/isolated_pitch_contour.wav`: contour-focused isolated audio when pitch contour extraction is enabled.",
        "",
        "## Definitions",
        "- Denoised audio removes noise while preserving as much of the original recording as possible.",
        "- Isolated audio extracts a target signal and may discard other content.",
        "",
        "## Settings",
        f"- Original sample rate: {summary['original_sample_rate_hz']} Hz",
        f"- Analysis sample rate: {summary['analysis_sample_rate_hz']} Hz",
        f"- Frequency range shown: {summary['min_frequency_hz']}-{summary['max_frequency_hz_used']} Hz",
        f"- Denoising: {summary['denoising']['settings']}",  # type: ignore[index]
        "",
        "Debug outputs are intentionally verbose and are generated only when debug flags are used.",
    ]
    output_path.write_text("\n".join(lines) + "\n")


def run_clean_analysis(args: argparse.Namespace) -> None:
    if args.clean_outputs:
        clean_generated_outputs(args.results_dir)
    else:
        args.results_dir.mkdir(parents=True, exist_ok=True)
    if args.dedupe_existing_results:
        removed = dedupe_existing_results(args.results_dir, args.keep_csv)
        print(f"Deduped existing outputs: removed {removed['svg']} SVG files and {removed['csv']} CSV files")

    wav_files = sorted(args.data_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.data_dir}")

    for wav_file in wav_files:
        original_samples, original_sample_rate, channels = read_mono_wav(wav_file)
        analysis_samples, sample_rate, resampled_for_analysis = choose_analysis_audio(
            original_samples,
            original_sample_rate,
            args.analysis_sample_rate_hz,
            args.resample,
            args.resample_method,
        )
        min_frequency_hz, max_frequency_hz, warnings = clamp_frequency_settings(
            sample_rate,
            args.min_frequency_hz,
            args.max_frequency_hz,
        )
        max_frequency_used = max_frequency_hz if max_frequency_hz is not None else sample_rate / 2
        file_dir = args.results_dir / wav_file.stem
        audio_dir = file_dir / "audio"
        segment_root = file_dir / "segments"
        averages_dir = file_dir / "segment_averages"
        make_dirs(file_dir, audio_dir, segment_root, averages_dir)

        denoised_original = filter_frequency_band(
            original_samples,
            original_sample_rate,
            args.highpass_cutoff_hz,
            None,
            args.notch_frequency_hz,
            args.notch_harmonics,
        )
        denoised_analysis, _, _ = choose_analysis_audio(
            denoised_original,
            original_sample_rate,
            sample_rate,
            args.resample,
            args.resample_method,
        )
        settings = denoising_settings_text(args.highpass_cutoff_hz, args.notch_frequency_hz, args.notch_harmonics)
        subtitle = (
            f"{wav_file.name}; analysis sample rate {sample_rate} Hz; "
            f"{min_frequency_hz:.0f}-{max_frequency_used:.0f} Hz; {settings}"
        )

        save_waveform_png_from_samples(original_samples, original_sample_rate, "Original waveform", file_dir / "waveform.png")
        save_clean_spectrogram_png(analysis_samples, sample_rate, wav_file.name, file_dir / "spectrogram_raw.png", "Raw spectrogram", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, subtitle)
        save_clean_spectrogram_png(denoised_analysis, sample_rate, wav_file.name, file_dir / "spectrogram_denoised.png", "Denoised spectrogram", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, subtitle)
        enhanced_matrix, _, frequency_limit = save_clean_spectrogram_png(
            denoised_analysis,
            sample_rate,
            wav_file.name,
            file_dir / "spectrogram_enhanced_pitch.png",
            "Enhanced pitch contour",
            args.spectrogram_fft_size,
            max_frequency_used,
            args.spectrogram_dynamic_range_db,
            subtitle,
            True,
            args.noise_floor_percentile,
        )
        contour = extract_pitch_contour(
            enhanced_matrix,
            len(denoised_analysis) / sample_rate if sample_rate else 0.0,
            frequency_limit,
            args.min_pitch_frequency_hz,
            args.notch_frequency_hz,
            args.notch_harmonics,
        )
        save_clean_spectrogram_png(
            denoised_analysis,
            sample_rate,
            wav_file.name,
            file_dir / "spectrogram_enhanced_pitch.png",
            "Enhanced pitch contour",
            args.spectrogram_fft_size,
            max_frequency_used,
            args.spectrogram_dynamic_range_db,
            subtitle,
            True,
            args.noise_floor_percentile,
            contour,
        )
        save_pitch_contour_png(wav_file.name, contour, file_dir / "pitch_contour.png", max_frequency_used, subtitle)

        denoised_analysis_result = analyze_samples_over_time(
            wav_file.name,
            denoised_analysis,
            sample_rate,
            channels,
            args.fft_size,
            args.hop_size,
            min_frequency_hz,
            max_frequency_used,
        )
        denoised_spectrum = denoised_analysis_result["average_spectrum"]  # type: ignore[assignment]
        save_spectrum_png_from_bins(wav_file.name, denoised_spectrum, file_dir / "average_spectrum_db.png", max_frequency_used, True, subtitle)  # type: ignore[arg-type]

        dominant_non_hum_hz, _dominant_non_hum_mag = dominant_non_hum_frequency(
            denoised_spectrum,  # type: ignore[arg-type]
            args.min_pitch_frequency_hz,
            max_frequency_used,
            args.notch_frequency_hz,
            args.notch_harmonics,
        )
        if dominant_non_hum_hz <= 0:
            dominant_non_hum_hz = float(denoised_analysis_result["global_dominant_frequency_hz"])
        band_half_width = args.isolated_band_width_hz / 2
        isolated_low = max(args.min_pitch_frequency_hz, dominant_non_hum_hz - band_half_width)
        isolated_high = min(max_frequency_used, dominant_non_hum_hz + band_half_width)
        if isolated_high <= isolated_low:
            isolated_high = min(max_frequency_used, isolated_low + args.isolated_band_width_hz)
        isolated_band = filter_frequency_band(denoised_original, original_sample_rate, isolated_low, isolated_high, 0, 0)
        contour_band = filter_frequency_band(
            denoised_original,
            original_sample_rate,
            max(args.min_pitch_frequency_hz, dominant_non_hum_hz - args.isolated_band_width_hz),
            min(max_frequency_used, dominant_non_hum_hz + args.isolated_band_width_hz),
            0,
            0,
        )
        if args.export_filtered_audio:
            write_wav_from_mono(denoised_original, original_sample_rate, channels, audio_dir / "denoised_filtered.wav")
            write_wav_from_mono(isolated_band, original_sample_rate, channels, audio_dir / "isolated_pitch_band.wav")
            if args.extract_pitch_profile:
                write_wav_from_mono(contour_band, original_sample_rate, channels, audio_dir / "isolated_pitch_contour.wav")

        segment_spectra: list[list[tuple[float, float]]] = []
        segment_summary: list[dict[str, int | float]] = []
        ranges = segment_sample_ranges(len(denoised_analysis), sample_rate, args.segment_duration_seconds, args.segment_overlap_seconds)
        for index, (start, end, start_seconds, end_seconds) in enumerate(ranges):
            segment_dir = segment_root / f"segment_{index:03d}_{start_seconds:.2f}s_{end_seconds:.2f}s"
            make_dirs(segment_dir)
            raw_segment = analysis_samples[start:end]
            denoised_segment = denoised_analysis[start:end]
            title_suffix = f"{wav_file.name}, {start_seconds:.2f}-{end_seconds:.2f} s"
            segment_subtitle = f"{title_suffix}; analysis sample rate {sample_rate} Hz; {min_frequency_hz:.0f}-{max_frequency_used:.0f} Hz; {settings}"
            save_waveform_png_from_samples(raw_segment, sample_rate, "Original waveform", segment_dir / "waveform.png")
            save_clean_spectrogram_png(raw_segment, sample_rate, title_suffix, segment_dir / "spectrogram_raw.png", "Raw spectrogram", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, segment_subtitle)
            save_clean_spectrogram_png(denoised_segment, sample_rate, title_suffix, segment_dir / "spectrogram_denoised.png", "Denoised spectrogram", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, segment_subtitle)
            segment_matrix, _, segment_frequency_limit = save_clean_spectrogram_png(denoised_segment, sample_rate, title_suffix, segment_dir / "spectrogram_enhanced_pitch.png", "Enhanced pitch contour", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, segment_subtitle, True, args.noise_floor_percentile)
            segment_contour = extract_pitch_contour(segment_matrix, len(denoised_segment) / sample_rate if sample_rate else 0.0, segment_frequency_limit, args.min_pitch_frequency_hz, args.notch_frequency_hz, args.notch_harmonics)
            save_clean_spectrogram_png(denoised_segment, sample_rate, title_suffix, segment_dir / "spectrogram_enhanced_pitch.png", "Enhanced pitch contour", args.spectrogram_fft_size, max_frequency_used, args.spectrogram_dynamic_range_db, segment_subtitle, True, args.noise_floor_percentile, segment_contour)
            save_pitch_contour_png(title_suffix, segment_contour, segment_dir / "pitch_contour.png", max_frequency_used, segment_subtitle)
            segment_analysis = analyze_samples_over_time(title_suffix, denoised_segment, sample_rate, channels, args.fft_size, args.hop_size, min_frequency_hz, max_frequency_used)
            segment_spectrum = segment_analysis["average_spectrum"]  # type: ignore[assignment]
            segment_spectra.append(segment_spectrum)  # type: ignore[arg-type]
            save_spectrum_png_from_bins(title_suffix, segment_spectrum, segment_dir / "average_spectrum_db.png", max_frequency_used, True, segment_subtitle)  # type: ignore[arg-type]
            segment_row = {
                "segment_index": index,
                "start_time_seconds": start_seconds,
                "end_time_seconds": end_seconds,
                "dominant_non_hum_frequency_hz": dominant_non_hum_frequency(segment_spectrum, args.min_pitch_frequency_hz, max_frequency_used, args.notch_frequency_hz, args.notch_harmonics)[0],  # type: ignore[arg-type]
                "pitch_contour_points": len(segment_contour),
            }
            segment_summary.append(segment_row)
            (segment_dir / "summary.json").write_text(json.dumps(segment_row, indent=2) + "\n")

        if segment_spectra:
            target_frequencies = [frequency for frequency, _ in segment_spectra[0]]
            stacked = [interpolate_spectrum(spectrum, target_frequencies) for spectrum in segment_spectra]
            mean_values = [statistics.fmean(values) for values in zip(*stacked)]
            median_values = [statistics.median(values) for values in zip(*stacked)]
            std_values = [statistics.pstdev(values) if len(values) > 1 else 0.0 for values in zip(*stacked)]
            q25_values = [percentile(sorted(values), 0.25) for values in zip(*stacked)]
            q75_values = [percentile(sorted(values), 0.75) for values in zip(*stacked)]
            save_spectrum_png_from_bins(wav_file.name, list(zip(target_frequencies, mean_values)), averages_dir / "mean_segment_spectrum_db.png", max_frequency_used, True, "mean segment spectrum")
            save_spectrum_png_from_bins(wav_file.name, list(zip(target_frequencies, median_values)), averages_dir / "median_segment_spectrum_db.png", max_frequency_used, True, "median segment spectrum")
            save_spectrum_variability_png(wav_file.name, target_frequencies, mean_values, [max(0.0, mean - std) for mean, std in zip(mean_values, std_values)], [mean + std for mean, std in zip(mean_values, std_values)], averages_dir / "segment_spectrum_variability.png", max_frequency_used, "mean +/- 1 std")
            save_spectrum_variability_png(wav_file.name, target_frequencies, median_values, q25_values, q75_values, averages_dir / "pitch_contours_over_segments.png", max_frequency_used, "25th-75th percentile band")
            (averages_dir / "summary.json").write_text(json.dumps({"segment_count": len(segment_spectra)}, indent=2) + "\n")

        if args.export_csv:
            csv_dir = file_dir / "csv"
            make_dirs(csv_dir)
            write_average_fft_csv(wav_file, csv_dir, denoised_analysis_result)
            with (csv_dir / "segment_summary.csv").open("w", newline="") as csv_file:
                fieldnames = ["segment_index", "start_time_seconds", "end_time_seconds", "dominant_non_hum_frequency_hz", "pitch_contour_points"]
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                for row in segment_summary:
                    writer.writerow(row)
            write_dominant_over_time_csv(wav_file, csv_dir, contour)

        if args.export_debug_plots:
            debug_dir = file_dir / "debug" / "plots"
            make_dirs(debug_dir)
            save_spectrum_png_from_bins(wav_file.name, fft_bins_from_samples(analysis_samples, sample_rate, args.fft_size), debug_dir / "pre_filter_spectrum.png", max_frequency_used, True, "pre-filter spectrum")
            save_spectrum_png_from_bins(wav_file.name, denoised_spectrum, debug_dir / "post_highpass_notch_spectrum.png", max_frequency_used, True, "post high-pass/notch spectrum")  # type: ignore[arg-type]
            save_clean_spectrogram_png(analysis_samples, sample_rate, wav_file.name, debug_dir / "raw_full_range_spectrogram.png", "Raw full-range spectrogram", args.spectrogram_fft_size, min(sample_rate / 2, max_frequency_used), args.spectrogram_dynamic_range_db, subtitle)

        if args.export_debug_audio:
            debug_audio = audio_dir / "debug"
            make_dirs(debug_audio)
            highpass_only = filter_frequency_band(original_samples, original_sample_rate, args.highpass_cutoff_hz, None, 0, 0)
            notch_only = filter_frequency_band(original_samples, original_sample_rate, None, None, args.notch_frequency_hz, args.notch_harmonics)
            dominant_only = filter_to_single_frequency(original_samples, original_sample_rate, dominant_non_hum_hz)
            write_wav_from_mono(highpass_only, original_sample_rate, channels, debug_audio / "highpass_only.wav")
            write_wav_from_mono(notch_only, original_sample_rate, channels, debug_audio / "notch_only.wav")
            write_wav_from_mono(denoised_original, original_sample_rate, channels, debug_audio / "highpass_notch.wav")
            write_wav_from_mono(dominant_only, original_sample_rate, channels, debug_audio / "dominant_sine_only.wav")

        summary = {
            "file": wav_file.name,
            "original_sample_rate_hz": original_sample_rate,
            "analysis_sample_rate_hz": sample_rate,
            "resampled_for_analysis": resampled_for_analysis,
            "nyquist_frequency_hz": sample_rate / 2,
            "min_frequency_hz": min_frequency_hz,
            "max_frequency_hz_used": max_frequency_used,
            "denoising": {
                "highpass_cutoff_hz": args.highpass_cutoff_hz,
                "notch_frequency_hz": args.notch_frequency_hz,
                "notch_harmonics": args.notch_harmonics,
                "settings": settings,
            },
            "isolated_audio": {
                "dominant_non_hum_frequency_hz": dominant_non_hum_hz,
                "band_low_hz": isolated_low,
                "band_high_hz": isolated_high,
            },
            "segments": segment_summary,
            "warnings": warnings,
        }
        (file_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        write_clean_summary_md(summary, file_dir / "summary.md")

    print(f"Analyzed {len(wav_files)} WAV files")
    print(f"Wrote clean outputs to: {args.results_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WAV files and export clean denoised/isolated outputs.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--hop-size", type=int, default=None)
    parser.add_argument("--min-frequency-hz", type=float, default=RECOMMENDED_MIN_FREQUENCY_HZ)
    parser.add_argument("--max-frequency-hz", type=float, default=MAX_SPECTRUM_HZ)
    parser.add_argument("--plot-max-frequency-hz", "--max-spectrum-hz", dest="plot_max_frequency_hz", type=float, default=MAX_SPECTRUM_HZ)
    parser.add_argument("--spectrogram-fft-size", type=int, default=SPECTROGRAM_FFT_SIZE)
    parser.add_argument("--analysis-sample-rate-hz", type=int, default=DEFAULT_ANALYSIS_SAMPLE_RATE_HZ)
    parser.add_argument("--resample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resample-method", choices=["polyphase"], default="polyphase")
    parser.add_argument("--segment-duration-seconds", type=float, default=2.5)
    parser.add_argument("--segment-overlap-seconds", type=float, default=0.0)
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--extract-pitch-profile", action="store_true", default=True)
    parser.add_argument("--highpass-cutoff-hz", type=float, default=80.0)
    parser.add_argument("--notch-frequency-hz", type=float, default=50.0)
    parser.add_argument("--notch-harmonics", type=int, default=8)
    parser.add_argument("--notch-width-hz", type=float, default=3.0)
    parser.add_argument("--spectrogram-dynamic-range-db", type=float, default=80.0)
    parser.add_argument("--noise-floor-percentile", type=float, default=20.0)
    parser.add_argument("--min-pitch-frequency-hz", type=float, default=80.0)
    parser.add_argument("--isolated-band-width-hz", type=float, default=80.0)
    parser.add_argument("--export-debug-plots", action="store_true")
    parser.add_argument("--export-debug-audio", action="store_true")
    parser.add_argument("--export-filtered-audio", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-outputs", action="store_true")
    parser.add_argument("--dedupe-existing-results", action="store_true")
    parser.add_argument("--keep-csv", action="store_true")
    args = parser.parse_args()

    if args.fft_size & (args.fft_size - 1):
        raise SystemExit("--fft-size must be a power of two")
    if args.hop_size is not None and args.hop_size < 1:
        raise SystemExit("--hop-size must be at least 1")
    if args.spectrogram_fft_size & (args.spectrogram_fft_size - 1):
        raise SystemExit("--spectrogram-fft-size must be a power of two")
    if args.analysis_sample_rate_hz < 2:
        raise SystemExit("--analysis-sample-rate-hz must be at least 2")
    if args.segment_duration_seconds <= 0:
        raise SystemExit("--segment-duration-seconds must be greater than 0")
    if args.segment_overlap_seconds < 0 or args.segment_overlap_seconds >= args.segment_duration_seconds:
        raise SystemExit("--segment-overlap-seconds must be at least 0 and shorter than segment duration")
    if args.spectrogram_dynamic_range_db <= 0:
        raise SystemExit("--spectrogram-dynamic-range-db must be greater than 0")
    if not 0 <= args.noise_floor_percentile <= 100:
        raise SystemExit("--noise-floor-percentile must be between 0 and 100")
    if args.min_pitch_frequency_hz < 0:
        raise SystemExit("--min-pitch-frequency-hz must be at least 0")
    if args.isolated_band_width_hz <= 0:
        raise SystemExit("--isolated-band-width-hz must be greater than 0")

    run_clean_analysis(args)

if __name__ == "__main__":
    main()
