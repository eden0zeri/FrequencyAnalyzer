#!/usr/bin/env python3
"""Analyze WAV files with full-file sliding-window FFTs."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import wave
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_DATA_DIR = Path("Data")
DEFAULT_RESULTS_DIR = Path("results")
FFT_SIZE = 8192
WAVEFORM_POINTS = 4000
MAX_SPECTRUM_HZ = 2000.0
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
    """Radix-2 Cooley-Tukey Fast Fourier Transform."""
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
    max_bin = fft_size // 2 - 1
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
    """First-window FFT, kept for backwards-compatible raw FFT CSV output."""
    actual_fft_size = min(fft_size, power_of_two_at_most(len(samples)))
    if actual_fft_size < 2:
        return []

    block = samples[:actual_fft_size]
    if len(block) < actual_fft_size:
        block = block + [0.0] * (actual_fft_size - len(block))

    window = hann_window(actual_fft_size)
    spectrum = fft([sample * window[index] for index, sample in enumerate(block)])
    return [
        (bin_index * sample_rate / actual_fft_size, abs(spectrum[bin_index]))
        for bin_index in range(actual_fft_size // 2)
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
    duration = len(samples) / sample_rate if sample_rate else 0.0
    actual_fft_size = min(fft_size, power_of_two_at_most(len(samples)))
    if actual_fft_size < 2:
        actual_fft_size = 2
    hop_size = hop_size or actual_fft_size // 2
    hop_size = max(1, min(hop_size, actual_fft_size))

    starts = frame_starts(len(samples), actual_fft_size, hop_size)
    window = hann_window(actual_fft_size)
    bin_range = frequency_bin_range(
        sample_rate,
        actual_fft_size,
        min_frequency_hz,
        max_frequency_hz,
    )
    magnitude_sums = [0.0] * (actual_fft_size // 2)
    frames: list[dict[str, float]] = []

    for start in starts:
        block = samples[start : start + actual_fft_size]
        if len(block) < actual_fft_size:
            block = block + [0.0] * (actual_fft_size - len(block))

        spectrum = fft([sample * window[index] for index, sample in enumerate(block)])
        frame_best_frequency = 0.0
        frame_best_magnitude = 0.0

        for bin_index in range(actual_fft_size // 2):
            magnitude = abs(spectrum[bin_index])
            magnitude_sums[bin_index] += magnitude
            if bin_index in bin_range and magnitude > frame_best_magnitude:
                frame_best_frequency = bin_index * sample_rate / actual_fft_size
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
        (bin_index * sample_rate / actual_fft_size, magnitude / frames_analyzed)
        for bin_index, magnitude in enumerate(magnitude_sums)
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
        "file": wav_path.name,
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
        warnings.append("filtered dominant-only file should show a narrow horizontal band")
        useful_span = useful_max - float(useful_range["useful_min_frequency_hz"])
        if useful_span > max(200.0, dominant_frequency_hz):
            warnings.append("filtered energy spread wider than expected")
            recommendations.append("Filtered output may not be isolating a single tone cleanly.")
        else:
            recommendations.append("Sparse spectrogram is expected for dominant-only filtered audio.")

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
    hop_size = max(1, actual_fft_size // 2)
    starts = frame_starts(len(samples), actual_fft_size, hop_size)
    column_step = max(1, math.ceil(len(starts) / max_columns))
    frequency_limit = min(max_frequency_hz, sample_rate / 2)
    max_bin = max(1, min(actual_fft_size // 2 - 1, int(frequency_limit * actual_fft_size / sample_rate)))
    window = hann_window(actual_fft_size)
    cells: list[list[float]] = []

    for start in starts[::column_step]:
        block = samples[start : start + actual_fft_size]
        if len(block) < actual_fft_size:
            block = block + [0.0] * (actual_fft_size - len(block))
        spectrum = fft([sample * window[index] for index, sample in enumerate(block)])
        column = [0.0] * band_count
        counts = [0] * band_count

        for bin_index in range(1, max_bin + 1):
            band_index = min(band_count - 1, int((bin_index - 1) * band_count / max_bin))
            column[band_index] += abs(spectrum[bin_index])
            counts[band_index] += 1

        for band_index, count in enumerate(counts):
            if count:
                column[band_index] /= count
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
        if "filtered" in str(row["output_type"]) and "dominant-only" in str(row["output_type"])
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WAV files with full-file FFT windows.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument(
        "--hop-size",
        type=int,
        default=None,
        help="Sliding-window hop size in samples. Defaults to fft-size // 2.",
    )
    parser.add_argument("--min-frequency-hz", type=float, default=20.0)
    parser.add_argument("--recommended-min-frequency-hz", type=float, default=RECOMMENDED_MIN_FREQUENCY_HZ)
    parser.add_argument(
        "--max-frequency-hz",
        type=float,
        default=None,
        help="Maximum frequency considered for dominant-frequency analysis. Default: no upper limit.",
    )
    parser.add_argument(
        "--plot-max-frequency-hz",
        "--max-spectrum-hz",
        dest="plot_max_frequency_hz",
        type=float,
        default=MAX_SPECTRUM_HZ,
        help="Full-range y-axis maximum for spectrum and spectrogram plots.",
    )
    parser.add_argument("--spectrogram-fft-size", type=int, default=SPECTROGRAM_FFT_SIZE)
    parser.add_argument("--auto-zoom", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--audit-results", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    if args.fft_size & (args.fft_size - 1):
        raise SystemExit("--fft-size must be a power of two")
    if args.hop_size is not None and args.hop_size < 1:
        raise SystemExit("--hop-size must be at least 1")
    if args.spectrogram_fft_size & (args.spectrogram_fft_size - 1):
        raise SystemExit("--spectrogram-fft-size must be a power of two")

    wav_files = sorted(args.data_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.data_dir}")

    fft_dir = args.results_dir / "fft"
    fft_average_dir = args.results_dir / "fft_average"
    fft_relevant_dir = args.results_dir / "fft_relevant"
    waveform_dir = args.results_dir / "waveforms"
    spectrum_dir = args.results_dir / "spectra"
    spectrum_zoomed_dir = args.results_dir / "spectra_zoomed"
    spectrum_relevant_dir = args.results_dir / "spectra_relevant"
    spectrum_relevant_zoomed_dir = args.results_dir / "spectra_relevant_zoomed"
    frequency_time_dir = args.results_dir / "frequency_time"
    frequency_time_zoomed_dir = args.results_dir / "frequency_time_zoomed"
    frequency_time_relevant_dir = args.results_dir / "frequency_time_relevant"
    frequency_time_relevant_zoomed_dir = args.results_dir / "frequency_time_relevant_zoomed"
    dominant_over_time_dir = args.results_dir / "dominant_over_time"
    dominant_over_time_svg_dir = args.results_dir / "dominant_over_time_svg"
    dominant_over_time_zoomed_svg_dir = args.results_dir / "dominant_over_time_zoomed_svg"
    filtered_full_dir = args.results_dir / "filtered_full"
    filtered_full_wav_dir = filtered_full_dir / "wav"
    filtered_full_waveform_dir = filtered_full_dir / "waveforms"
    filtered_full_spectrum_dir = filtered_full_dir / "spectra"
    filtered_full_frequency_time_dir = filtered_full_dir / "frequency_time"
    filtered_relevant_dir = args.results_dir / "filtered_relevant"
    filtered_relevant_wav_dir = filtered_relevant_dir / "wav"
    filtered_relevant_waveform_dir = filtered_relevant_dir / "waveforms"
    filtered_relevant_spectrum_dir = filtered_relevant_dir / "spectra"
    filtered_relevant_frequency_time_dir = filtered_relevant_dir / "frequency_time"
    make_dirs(
        args.results_dir,
        fft_dir,
        fft_average_dir,
        fft_relevant_dir,
        waveform_dir,
        spectrum_dir,
        spectrum_zoomed_dir,
        spectrum_relevant_dir,
        spectrum_relevant_zoomed_dir,
        frequency_time_dir,
        frequency_time_zoomed_dir,
        frequency_time_relevant_dir,
        frequency_time_relevant_zoomed_dir,
        dominant_over_time_dir,
        dominant_over_time_svg_dir,
        dominant_over_time_zoomed_svg_dir,
        filtered_full_dir,
        filtered_full_wav_dir,
        filtered_full_waveform_dir,
        filtered_full_spectrum_dir,
        filtered_full_frequency_time_dir,
        filtered_relevant_dir,
        filtered_relevant_wav_dir,
        filtered_relevant_waveform_dir,
        filtered_relevant_spectrum_dir,
        filtered_relevant_frequency_time_dir,
    )

    dominant_rows: list[dict[str, int | float | str]] = []
    cutoff_rows: list[dict[str, int | float | str]] = []
    filtered_audio_rows: list[dict[str, int | float | str]] = []
    audit_rows: list[dict[str, str | float | bool]] = []

    for wav_file in wav_files:
        analysis = analyze_dominant_frequency_over_time(
            wav_file,
            args.fft_size,
            args.hop_size,
            args.min_frequency_hz,
            args.max_frequency_hz,
        )
        samples = analysis["samples"]  # type: ignore[assignment]
        sample_rate = int(analysis["sample_rate_hz"])
        channels = int(analysis["channels"])
        average_spectrum = analysis["average_spectrum"]  # type: ignore[assignment]
        frames = analysis["frames"]  # type: ignore[assignment]
        actual_fft_size = int(analysis["fft_size"])

        first_window_bins = fft_bins_from_samples(samples, sample_rate, args.fft_size)  # type: ignore[arg-type]
        write_first_window_fft_csv(
            wav_file,
            fft_dir,
            first_window_bins,
            sample_rate,
            actual_fft_size,
        )
        write_average_fft_csv(wav_file, fft_average_dir, analysis)
        write_dominant_over_time_csv(wav_file, dominant_over_time_dir, frames)  # type: ignore[arg-type]

        global_frequency = float(analysis["global_dominant_frequency_hz"])
        global_magnitude = float(analysis["global_dominant_magnitude"])
        relevant_max_frequency_hz = relevant_frequency_cutoff(
            average_spectrum,  # type: ignore[arg-type]
            args.plot_max_frequency_hz,
            RELEVANT_MIN_HZ,
            RELEVANT_THRESHOLD_FRACTION,
        )
        useful_range = detect_useful_frequency_range(
            average_spectrum,  # type: ignore[arg-type]
            args.min_frequency_hz,
            args.max_frequency_hz,
            hard_plot_ceiling_hz=args.plot_max_frequency_hz,
        )
        relevant_bins = [
            item for item in average_spectrum  # type: ignore[union-attr]
            if item[0] <= relevant_max_frequency_hz
        ]
        relevant_peak_frequency_hz, _ = dominant_frequency(relevant_bins)
        relevant_useful_range = detect_useful_frequency_range(
            relevant_bins,
            args.min_frequency_hz,
            relevant_max_frequency_hz,
            hard_plot_ceiling_hz=relevant_max_frequency_hz,
        )
        dominant_time_svg = dominant_over_time_svg_dir / f"{wav_file.stem}_dominant_over_time.svg"
        write_dominant_over_time_svg(
            wav_file.name,
            frames,  # type: ignore[arg-type]
            dominant_time_svg,
            y_max_hz=args.plot_max_frequency_hz,
            subtitle="original full range",
        )
        dominant_time_zoom_created = False
        if args.auto_zoom and zoom_needed(float(useful_range["useful_max_frequency_hz"]), args.plot_max_frequency_hz):
            dominant_time_zoom_created = True
            low = float(useful_range["useful_min_frequency_hz"])
            high = float(useful_range["useful_max_frequency_hz"])
            write_dominant_over_time_svg(
                wav_file.name,
                frames,  # type: ignore[arg-type]
                dominant_over_time_zoomed_svg_dir / f"{wav_file.stem}_dominant_over_time_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                y_max_hz=high,
                subtitle="original",
                zoomed=True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "dominant frequency over time",
                dominant_time_svg,
                0.0,
                args.plot_max_frequency_hz,
                useful_range,
                global_frequency,
                args.recommended_min_frequency_hz,
                dominant_time_zoom_created,
            )
        )
        write_fft_bins_csv(
            wav_file,
            fft_relevant_dir,
            average_spectrum,  # type: ignore[arg-type]
            sample_rate,
            actual_fft_size,
            relevant_max_frequency_hz,
        )

        dominant_rows.append(
            {
                "file": wav_file.name,
                "dominant_frequency_hz": global_frequency,
                "dominant_magnitude": global_magnitude,
                "dominant_method": "mean spectrum across full file",
                "median_frame_dominant_frequency_hz": analysis["median_frame_dominant_frequency_hz"],
                "mean_frame_dominant_frequency_hz": analysis["mean_frame_dominant_frequency_hz"],
                "std_frame_dominant_frequency_hz": analysis["std_frame_dominant_frequency_hz"],
                "strongest_frame_dominant_frequency_hz": analysis["strongest_frame_dominant_frequency_hz"],
                "strongest_frame_time_seconds": analysis["strongest_frame_time_seconds"],
                "duration_seconds": analysis["duration_seconds"],
                "sample_rate_hz": sample_rate,
                "channels": channels,
                "fft_size": actual_fft_size,
                "hop_size": analysis["hop_size"],
                "frames_analyzed": analysis["number_of_frames_analyzed"],
            }
        )
        cutoff_rows.append(
            {
                "file": wav_file.name,
                "relevant_max_frequency_hz": relevant_max_frequency_hz,
            }
        )

        write_waveform_svg_from_samples(
            samples,  # type: ignore[arg-type]
            sample_rate,
            wav_file.name,
            waveform_dir / f"{wav_file.stem}_waveform.svg",
        )
        write_spectrum_svg_from_bins(
            wav_file.name,
            average_spectrum,  # type: ignore[arg-type]
            actual_fft_size,
            spectrum_dir / f"{wav_file.stem}_fft_spectrum.svg",
            args.plot_max_frequency_hz,
            "original full range",
        )
        spectrum_zoom_created = False
        if args.auto_zoom and zoom_needed(float(useful_range["useful_max_frequency_hz"]), args.plot_max_frequency_hz):
            spectrum_zoom_created = True
            low = float(useful_range["useful_min_frequency_hz"])
            high = float(useful_range["useful_max_frequency_hz"])
            write_spectrum_svg_from_bins(
                wav_file.name,
                average_spectrum,  # type: ignore[arg-type]
                actual_fft_size,
                spectrum_zoomed_dir / f"{wav_file.stem}_fft_spectrum_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                high,
                "original",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "average spectrum",
                spectrum_dir / f"{wav_file.stem}_fft_spectrum.svg",
                0.0,
                args.plot_max_frequency_hz,
                useful_range,
                global_frequency,
                args.recommended_min_frequency_hz,
                spectrum_zoom_created,
            )
        )
        write_spectrum_svg_from_bins(
            wav_file.name,
            average_spectrum,  # type: ignore[arg-type]
            actual_fft_size,
            spectrum_relevant_dir / f"{wav_file.stem}_fft_spectrum_relevant.svg",
            relevant_max_frequency_hz,
            "original relevant range",
        )
        relevant_spectrum_zoom_created = False
        if args.auto_zoom and zoom_needed(float(relevant_useful_range["useful_max_frequency_hz"]), relevant_max_frequency_hz):
            relevant_spectrum_zoom_created = True
            low = float(relevant_useful_range["useful_min_frequency_hz"])
            high = float(relevant_useful_range["useful_max_frequency_hz"])
            write_spectrum_svg_from_bins(
                wav_file.name,
                relevant_bins,
                actual_fft_size,
                spectrum_relevant_zoomed_dir / f"{wav_file.stem}_fft_spectrum_relevant_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                high,
                "original relevant",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "relevant average spectrum",
                spectrum_relevant_dir / f"{wav_file.stem}_fft_spectrum_relevant.svg",
                0.0,
                relevant_max_frequency_hz,
                relevant_useful_range,
                relevant_peak_frequency_hz,
                args.recommended_min_frequency_hz,
                relevant_spectrum_zoom_created,
            )
        )
        write_frequency_time_svg_from_samples(
            samples,  # type: ignore[arg-type]
            sample_rate,
            wav_file.name,
            frequency_time_dir / f"{wav_file.stem}_frequency_time.svg",
            args.spectrogram_fft_size,
            args.plot_max_frequency_hz,
            "original full range",
        )
        frequency_time_zoom_created = False
        if args.auto_zoom and zoom_needed(float(useful_range["useful_max_frequency_hz"]), args.plot_max_frequency_hz):
            frequency_time_zoom_created = True
            low = float(useful_range["useful_min_frequency_hz"])
            high = float(useful_range["useful_max_frequency_hz"])
            write_frequency_time_svg_from_samples(
                samples,  # type: ignore[arg-type]
                sample_rate,
                wav_file.name,
                frequency_time_zoomed_dir / f"{wav_file.stem}_frequency_time_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                args.spectrogram_fft_size,
                high,
                "original",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "frequency time spectrogram",
                frequency_time_dir / f"{wav_file.stem}_frequency_time.svg",
                0.0,
                args.plot_max_frequency_hz,
                useful_range,
                global_frequency,
                args.recommended_min_frequency_hz,
                frequency_time_zoom_created,
            )
        )
        write_frequency_time_svg_from_samples(
            samples,  # type: ignore[arg-type]
            sample_rate,
            wav_file.name,
            frequency_time_relevant_dir / f"{wav_file.stem}_frequency_time_relevant.svg",
            args.spectrogram_fft_size,
            relevant_max_frequency_hz,
            "original relevant range",
        )
        relevant_frequency_time_zoom_created = False
        if args.auto_zoom and zoom_needed(float(relevant_useful_range["useful_max_frequency_hz"]), relevant_max_frequency_hz):
            relevant_frequency_time_zoom_created = True
            low = float(relevant_useful_range["useful_min_frequency_hz"])
            high = float(relevant_useful_range["useful_max_frequency_hz"])
            write_frequency_time_svg_from_samples(
                samples,  # type: ignore[arg-type]
                sample_rate,
                wav_file.name,
                frequency_time_relevant_zoomed_dir / f"{wav_file.stem}_frequency_time_relevant_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                args.spectrogram_fft_size,
                high,
                "original relevant",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "relevant frequency time spectrogram",
                frequency_time_relevant_dir / f"{wav_file.stem}_frequency_time_relevant.svg",
                0.0,
                relevant_max_frequency_hz,
                relevant_useful_range,
                relevant_peak_frequency_hz,
                args.recommended_min_frequency_hz,
                relevant_frequency_time_zoom_created,
            )
        )

        filtered_full_samples = filter_to_single_frequency(
            samples,  # type: ignore[arg-type]
            sample_rate,
            global_frequency,
        )
        filtered_relevant_samples = filter_to_single_frequency(
            samples,  # type: ignore[arg-type]
            sample_rate,
            relevant_peak_frequency_hz,
        )
        filtered_full_wav = filtered_full_wav_dir / f"{wav_file.stem}_full_dominant_only.wav"
        filtered_relevant_wav = filtered_relevant_wav_dir / f"{wav_file.stem}_relevant_dominant_only.wav"
        write_wav_from_mono(filtered_full_samples, sample_rate, channels, filtered_full_wav)
        write_wav_from_mono(filtered_relevant_samples, sample_rate, channels, filtered_relevant_wav)
        filtered_audio_rows.extend(
            [
                {
                    "source_file": wav_file.name,
                    "mode": "full",
                    "kept_frequency_hz": global_frequency,
                    "relevant_max_frequency_hz": "",
                    "output_wav": str(filtered_full_wav),
                },
                {
                    "source_file": wav_file.name,
                    "mode": "relevant",
                    "kept_frequency_hz": relevant_peak_frequency_hz,
                    "relevant_max_frequency_hz": relevant_max_frequency_hz,
                    "output_wav": str(filtered_relevant_wav),
                },
            ]
        )

        filtered_full_analysis = analyze_dominant_frequency_over_time(
            filtered_full_wav,
            args.fft_size,
            args.hop_size,
            args.min_frequency_hz,
            args.max_frequency_hz,
        )
        filtered_relevant_analysis = analyze_dominant_frequency_over_time(
            filtered_relevant_wav,
            args.fft_size,
            args.hop_size,
            args.min_frequency_hz,
            args.max_frequency_hz,
        )
        filtered_full_useful_range = detect_useful_frequency_range(
            filtered_full_analysis["average_spectrum"],  # type: ignore[arg-type]
            args.min_frequency_hz,
            args.max_frequency_hz,
            hard_plot_ceiling_hz=args.plot_max_frequency_hz,
        )
        filtered_relevant_useful_range = detect_useful_frequency_range(
            filtered_relevant_analysis["average_spectrum"],  # type: ignore[arg-type]
            args.min_frequency_hz,
            relevant_max_frequency_hz,
            hard_plot_ceiling_hz=relevant_max_frequency_hz,
        )
        write_waveform_svg_from_samples(
            filtered_full_samples,
            sample_rate,
            filtered_full_wav.name,
            filtered_full_waveform_dir / f"{wav_file.stem}_full_dominant_only_waveform.svg",
        )
        write_spectrum_svg_from_bins(
            filtered_full_wav.name,
            filtered_full_analysis["average_spectrum"],  # type: ignore[arg-type]
            int(filtered_full_analysis["fft_size"]),
            filtered_full_spectrum_dir / f"{wav_file.stem}_full_dominant_only_spectrum.svg",
            args.plot_max_frequency_hz,
            "full dominant-only",
        )
        filtered_full_spectrum_zoom_created = False
        if args.auto_zoom and zoom_needed(float(filtered_full_useful_range["useful_max_frequency_hz"]), args.plot_max_frequency_hz):
            filtered_full_spectrum_zoom_created = True
            low = float(filtered_full_useful_range["useful_min_frequency_hz"])
            high = float(filtered_full_useful_range["useful_max_frequency_hz"])
            write_spectrum_svg_from_bins(
                filtered_full_wav.name,
                filtered_full_analysis["average_spectrum"],  # type: ignore[arg-type]
                int(filtered_full_analysis["fft_size"]),
                filtered_full_spectrum_dir / f"{wav_file.stem}_full_dominant_only_spectrum_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                high,
                "full dominant-only",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "filtered full dominant-only spectrum",
                filtered_full_spectrum_dir / f"{wav_file.stem}_full_dominant_only_spectrum.svg",
                0.0,
                args.plot_max_frequency_hz,
                filtered_full_useful_range,
                float(filtered_full_analysis["global_dominant_frequency_hz"]),
                args.recommended_min_frequency_hz,
                filtered_full_spectrum_zoom_created,
                True,
            )
        )
        write_frequency_time_svg_from_samples(
            filtered_full_samples,
            sample_rate,
            filtered_full_wav.name,
            filtered_full_frequency_time_dir / f"{wav_file.stem}_full_dominant_only_frequency_time.svg",
            args.spectrogram_fft_size,
            args.plot_max_frequency_hz,
            "full dominant-only",
        )
        filtered_full_frequency_time_zoom_created = False
        if args.auto_zoom and zoom_needed(float(filtered_full_useful_range["useful_max_frequency_hz"]), args.plot_max_frequency_hz):
            filtered_full_frequency_time_zoom_created = True
            low = float(filtered_full_useful_range["useful_min_frequency_hz"])
            high = float(filtered_full_useful_range["useful_max_frequency_hz"])
            write_frequency_time_svg_from_samples(
                filtered_full_samples,
                sample_rate,
                filtered_full_wav.name,
                filtered_full_frequency_time_dir / f"{wav_file.stem}_full_dominant_only_frequency_time_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                args.spectrogram_fft_size,
                high,
                "full dominant-only",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "filtered full dominant-only frequency time spectrogram",
                filtered_full_frequency_time_dir / f"{wav_file.stem}_full_dominant_only_frequency_time.svg",
                0.0,
                args.plot_max_frequency_hz,
                filtered_full_useful_range,
                float(filtered_full_analysis["global_dominant_frequency_hz"]),
                args.recommended_min_frequency_hz,
                filtered_full_frequency_time_zoom_created,
                True,
            )
        )
        write_waveform_svg_from_samples(
            filtered_relevant_samples,
            sample_rate,
            filtered_relevant_wav.name,
            filtered_relevant_waveform_dir / f"{wav_file.stem}_relevant_dominant_only_waveform.svg",
        )
        write_spectrum_svg_from_bins(
            filtered_relevant_wav.name,
            filtered_relevant_analysis["average_spectrum"],  # type: ignore[arg-type]
            int(filtered_relevant_analysis["fft_size"]),
            filtered_relevant_spectrum_dir / f"{wav_file.stem}_relevant_dominant_only_spectrum.svg",
            relevant_max_frequency_hz,
            "relevant dominant-only",
        )
        filtered_relevant_spectrum_zoom_created = False
        if args.auto_zoom and zoom_needed(float(filtered_relevant_useful_range["useful_max_frequency_hz"]), relevant_max_frequency_hz):
            filtered_relevant_spectrum_zoom_created = True
            low = float(filtered_relevant_useful_range["useful_min_frequency_hz"])
            high = float(filtered_relevant_useful_range["useful_max_frequency_hz"])
            write_spectrum_svg_from_bins(
                filtered_relevant_wav.name,
                filtered_relevant_analysis["average_spectrum"],  # type: ignore[arg-type]
                int(filtered_relevant_analysis["fft_size"]),
                filtered_relevant_spectrum_dir / f"{wav_file.stem}_relevant_dominant_only_spectrum_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                high,
                "relevant dominant-only",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "filtered relevant dominant-only spectrum",
                filtered_relevant_spectrum_dir / f"{wav_file.stem}_relevant_dominant_only_spectrum.svg",
                0.0,
                relevant_max_frequency_hz,
                filtered_relevant_useful_range,
                float(filtered_relevant_analysis["global_dominant_frequency_hz"]),
                args.recommended_min_frequency_hz,
                filtered_relevant_spectrum_zoom_created,
                True,
            )
        )
        write_frequency_time_svg_from_samples(
            filtered_relevant_samples,
            sample_rate,
            filtered_relevant_wav.name,
            filtered_relevant_frequency_time_dir / f"{wav_file.stem}_relevant_dominant_only_frequency_time.svg",
            args.spectrogram_fft_size,
            relevant_max_frequency_hz,
            "relevant dominant-only",
        )
        filtered_relevant_frequency_time_zoom_created = False
        if args.auto_zoom and zoom_needed(float(filtered_relevant_useful_range["useful_max_frequency_hz"]), relevant_max_frequency_hz):
            filtered_relevant_frequency_time_zoom_created = True
            low = float(filtered_relevant_useful_range["useful_min_frequency_hz"])
            high = float(filtered_relevant_useful_range["useful_max_frequency_hz"])
            write_frequency_time_svg_from_samples(
                filtered_relevant_samples,
                sample_rate,
                filtered_relevant_wav.name,
                filtered_relevant_frequency_time_dir / f"{wav_file.stem}_relevant_dominant_only_frequency_time_zoomed_{hz_for_filename(low)}_{hz_for_filename(high)}Hz.svg",
                args.spectrogram_fft_size,
                high,
                "relevant dominant-only",
                True,
            )
        audit_rows.append(
            audit_row(
                wav_file.name,
                "filtered relevant dominant-only frequency time spectrogram",
                filtered_relevant_frequency_time_dir / f"{wav_file.stem}_relevant_dominant_only_frequency_time.svg",
                0.0,
                relevant_max_frequency_hz,
                filtered_relevant_useful_range,
                float(filtered_relevant_analysis["global_dominant_frequency_hz"]),
                args.recommended_min_frequency_hz,
                filtered_relevant_frequency_time_zoom_created,
                True,
            )
        )

    write_dominant_csv(dominant_rows, args.results_dir / "dominant_frequencies.csv")
    write_dominant_svg(dominant_rows, args.results_dir / "dominant_frequencies.svg")
    write_relevant_cutoffs_csv(cutoff_rows, args.results_dir / "relevant_frequency_cutoffs.csv")
    write_filtered_audio_summary_csv(filtered_audio_rows, args.results_dir / "filtered_audio_summary.csv")
    if args.audit_results:
        write_results_audit_csv(audit_rows, args.results_dir / "results_audit.csv")
        write_results_audit_md(audit_rows, args.results_dir / "results_audit.md")

    print(f"Analyzed {len(wav_files)} WAV files")
    print(f"Wrote first-window FFT CSV files to: {fft_dir}")
    print(f"Wrote full-file average FFT CSV files to: {fft_average_dir}")
    print(f"Wrote dominant-over-time CSV files to: {dominant_over_time_dir}")
    print(f"Wrote dominant-over-time plots to: {dominant_over_time_svg_dir}")
    print(f"Wrote relevant FFT CSV files to: {fft_relevant_dir}")
    print(f"Wrote waveform plots to: {waveform_dir}")
    print(f"Wrote average FFT spectrum plots to: {spectrum_dir}")
    print(f"Wrote relevant average FFT spectrum plots to: {spectrum_relevant_dir}")
    print(f"Wrote frequency-time plots to: {frequency_time_dir}")
    print(f"Wrote relevant frequency-time plots to: {frequency_time_relevant_dir}")
    print(f"Wrote full-spectrum dominant-only audio and plots to: {filtered_full_dir}")
    print(f"Wrote relevant-spectrum dominant-only audio and plots to: {filtered_relevant_dir}")
    if args.audit_results:
        print(f"Wrote results audit to: {args.results_dir / 'results_audit.csv'}")
        print(f"Wrote human-readable audit to: {args.results_dir / 'results_audit.md'}")
    print(f"Wrote dominant frequency summary to: {args.results_dir}")


if __name__ == "__main__":
    main()
