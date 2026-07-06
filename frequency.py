#!/usr/bin/env python3
"""Analyze WAV files with FFT and save CSV/SVG results."""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path
from xml.sax.saxutils import escape


DEFAULT_DATA_DIR = Path("Data")
DEFAULT_RESULTS_DIR = Path("results")
FFT_SIZE = 8192
WAVEFORM_POINTS = 4000
MAX_SPECTRUM_HZ = 2000.0
SPECTROGRAM_FFT_SIZE = 2048
SPECTROGRAM_BANDS = 180
SPECTROGRAM_COLUMNS = 320
RELEVANT_MIN_HZ = 400.0
RELEVANT_THRESHOLD_FRACTION = 0.10
SMOOTHING_BINS = 9
FILTER_BLOCK_SIZE = 4096


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


def hann_window(size: int) -> list[float]:
    return [0.5 - 0.5 * math.cos(2 * math.pi * index / (size - 1)) for index in range(size)]


def wav_metadata(path: Path) -> dict[str, int | float]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        return {
            "sample_rate_hz": sample_rate,
            "sample_width_bytes": wav_file.getsampwidth(),
            "channels": wav_file.getnchannels(),
            "frame_count": frame_count,
            "duration_seconds": frame_count / sample_rate if sample_rate else 0.0,
        }


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


def fft_wav_file(
    path: Path,
    fft_size: int = FFT_SIZE,
) -> tuple[int, int, list[tuple[float, float]]]:
    """Read the first FFT-sized chunk of a WAV file and return frequency/magnitude bins."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()

        if frame_count < fft_size:
            fft_size = 1 << (frame_count.bit_length() - 1)
        if fft_size < 2:
            raise ValueError(f"{path} is too short for FFT")

        raw = wav_file.readframes(fft_size)

    samples = decode_pcm_frames(raw, sample_width, channels)
    window = hann_window(fft_size)
    spectrum = fft([sample * window[index] for index, sample in enumerate(samples)])

    bins: list[tuple[float, float]] = []
    for bin_index in range(fft_size // 2):
        frequency_hz = bin_index * sample_rate / fft_size
        magnitude = abs(spectrum[bin_index])
        bins.append((frequency_hz, magnitude))

    return sample_rate, fft_size, bins


def waveform_trace(
    path: Path,
    point_count: int = WAVEFORM_POINTS,
) -> tuple[list[tuple[float, float]], float]:
    """Return evenly spaced amplitude points for drawing a classic waveform."""
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        duration = frame_count / sample_rate if sample_rate else 0.0
        bucket_size = max(1, math.ceil(frame_count / point_count))
        points: list[tuple[float, float]] = []
        frames_read = 0

        while frames_read < frame_count:
            frames_to_read = min(bucket_size, frame_count - frames_read)
            raw = wav_file.readframes(frames_to_read)
            samples = decode_pcm_frames(raw, sample_width, channels)

            if samples:
                midpoint = len(samples) // 2
                time_seconds = (frames_read + midpoint) / sample_rate
                points.append((time_seconds, samples[midpoint]))

            frames_read += frames_to_read

    return points, duration


def dominant_frequency(bins: list[tuple[float, float]]) -> tuple[float, float]:
    """Return the strongest non-DC FFT bin as frequency and magnitude."""
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
    """Estimate where the useful spectrum has died off."""
    visible_bins = [(frequency, magnitude) for frequency, magnitude in bins[1:] if frequency <= max_search_hz]
    if not visible_bins:
        return minimum_hz

    frequencies = [frequency for frequency, _ in visible_bins]
    smoothed = smoothed_magnitudes([magnitude for _, magnitude in visible_bins])
    peak_magnitude = max(smoothed)
    threshold = peak_magnitude * threshold_fraction
    cutoff = max(
        (frequency for frequency, magnitude in zip(frequencies, smoothed) if magnitude >= threshold),
        default=minimum_hz,
    )

    return min(max(cutoff, minimum_hz), max_search_hz)


def filter_to_single_frequency(
    samples: list[float],
    sample_rate: int,
    target_frequency_hz: float,
    block_size: int = FILTER_BLOCK_SIZE,
) -> list[float]:
    """Reconstruct audio using only one target frequency with a moving amplitude estimate."""
    if not samples or target_frequency_hz <= 0:
        return [0.0] * len(samples)

    block_size = min(block_size, 1 << (len(samples).bit_length() - 1))
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
            value = (
                cos_amplitude * math.cos(angle)
                + sin_amplitude * math.sin(angle)
            )
            output[output_index] += value * weight
            weights[output_index] += weight

        if start + block_size >= len(samples):
            break

    for index, weight in enumerate(weights):
        if weight:
            output[index] /= weight

    return output


def write_single_frequency_wav(
    wav_path: Path,
    output_path: Path,
    target_frequency_hz: float,
) -> None:
    samples, sample_rate, channels = read_mono_wav(wav_path)
    filtered = filter_to_single_frequency(samples, sample_rate, target_frequency_hz)
    write_wav_from_mono(filtered, sample_rate, channels, output_path)


def write_fft_csv(
    wav_path: Path,
    output_dir: Path,
    fft_size: int = FFT_SIZE,
    max_frequency_hz: float | None = None,
) -> tuple[float, float]:
    sample_rate, actual_fft_size, bins = fft_wav_file(wav_path, fft_size)
    output_path = output_dir / f"{wav_path.stem}_fft.csv"
    output_bins = [item for item in bins if max_frequency_hz is None or item[0] <= max_frequency_hz]
    peak_frequency_hz, peak_magnitude = dominant_frequency(output_bins)

    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["source_file", wav_path.name])
        writer.writerow(["sample_rate_hz", sample_rate])
        writer.writerow(["fft_size", actual_fft_size])
        if max_frequency_hz is not None:
            writer.writerow(["max_frequency_hz", f"{max_frequency_hz:.6f}"])
        writer.writerow(["dominant_frequency_hz", f"{peak_frequency_hz:.6f}"])
        writer.writerow([])
        writer.writerow(["frequency_hz", "magnitude"])

        for frequency_hz, magnitude in output_bins:
            writer.writerow([f"{frequency_hz:.6f}", f"{magnitude:.12f}"])

    return peak_frequency_hz, peak_magnitude


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
                    "method": "smoothed magnitude threshold with minimum cutoff",
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


def write_dominant_csv(rows: list[dict[str, int | float | str]], output_path: Path) -> None:
    fieldnames = [
        "file",
        "dominant_frequency_hz",
        "dominant_magnitude",
        "duration_seconds",
        "sample_rate_hz",
        "channels",
        "fft_size",
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
                    "duration_seconds": f"{float(row['duration_seconds']):.3f}",
                }
            )


def svg_text(x: float, y: float, text: str, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" '
        f'font-family="Arial, sans-serif" text-anchor="{anchor}">{escape(text)}</text>'
    )


def write_waveform_svg(wav_path: Path, output_path: Path) -> None:
    points, duration = waveform_trace(wav_path)
    width = 1000
    height = 420
    left = 78
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    center_y = top + plot_height / 2
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"Waveform: {wav_path.name}", 20),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Time (seconds)", 13),
        svg_text(22, top + plot_height / 2, "Amplitude", 13),
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

    elements.append(
        f'<line x1="{left}" y1="{center_y:.1f}" x2="{width - right}" y2="{center_y:.1f}" stroke="#64748b" stroke-width="1"/>'
    )
    if waveform_points:
        elements.append(
            f'<polyline points="{" ".join(waveform_points)}" fill="none" stroke="#2563eb" stroke-width="1.25"/>'
        )

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def write_spectrum_svg(
    wav_path: Path,
    output_path: Path,
    fft_size: int = FFT_SIZE,
    max_spectrum_hz: float = MAX_SPECTRUM_HZ,
) -> None:
    sample_rate, actual_fft_size, bins = fft_wav_file(wav_path, fft_size)
    frequency_limit = min(max_spectrum_hz, sample_rate / 2)
    visible_bins = [item for item in bins if item[0] <= frequency_limit]
    max_magnitude = max((magnitude for _, magnitude in visible_bins), default=1.0)
    max_magnitude = max(max_magnitude, 1e-12)
    peak_frequency_hz, _ = dominant_frequency(visible_bins)
    width = 1000
    height = 420
    left = 78
    right = 34
    top = 58
    bottom = 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        svg_text(width / 2, 32, f"FFT Spectrum: {wav_path.name}", 20),
        svg_text(width / 2, 52, f"FFT size {actual_fft_size}, peak {peak_frequency_hz:.2f} Hz", 12),
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#334155"/>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#334155"/>',
        svg_text(width / 2, height - 12, "Frequency (Hz)", 13),
        svg_text(22, top + plot_height / 2, "Magnitude", 13),
    ]

    for tick in range(6):
        frequency = frequency_limit * tick / 5
        x = left + (frequency / frequency_limit) * plot_width if frequency_limit else left
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{height - bottom + 6}" stroke="#cbd5e1"/>')
        elements.append(svg_text(x, height - bottom + 24, f"{frequency:.0f}", 11))

        magnitude = max_magnitude * tick / 5
        y = top + plot_height - (magnitude / max_magnitude) * plot_height
        elements.append(f'<line x1="{left - 6}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" stroke="#cbd5e1"/>')
        elements.append(svg_text(left - 12, y + 4, f"{magnitude:.1f}", 11, "end"))

    points = []
    for frequency, magnitude in visible_bins:
        x = left + (frequency / frequency_limit) * plot_width if frequency_limit else left
        y = top + plot_height - (magnitude / max_magnitude) * plot_height
        points.append(f"{x:.1f},{y:.1f}")

    if points:
        elements.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="#0f766e" stroke-width="1.5"/>')

    elements.append("</svg>")
    output_path.write_text("\n".join(elements))


def spectrogram_cells(
    wav_path: Path,
    fft_size: int = SPECTROGRAM_FFT_SIZE,
    max_frequency_hz: float = MAX_SPECTRUM_HZ,
    band_count: int = SPECTROGRAM_BANDS,
    max_columns: int = SPECTROGRAM_COLUMNS,
) -> tuple[list[list[float]], float, float, int]:
    """Return time/frequency magnitude cells for a spectrogram."""
    with wave.open(str(wav_path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        channels = wav_file.getnchannels()
        frame_count = wav_file.getnframes()
        duration = frame_count / sample_rate if sample_rate else 0.0
        actual_fft_size = min(fft_size, 1 << (frame_count.bit_length() - 1))

        if actual_fft_size < 2:
            raise ValueError(f"{wav_path} is too short for spectrogram")

        hop_size = actual_fft_size // 2
        window_count = max(1, 1 + (frame_count - actual_fft_size) // hop_size)
        column_step = max(1, math.ceil(window_count / max_columns))
        frequency_limit = min(max_frequency_hz, sample_rate / 2)
        max_bin = max(1, min(actual_fft_size // 2 - 1, int(frequency_limit * actual_fft_size / sample_rate)))
        window = hann_window(actual_fft_size)
        cells: list[list[float]] = []

        for window_index in range(0, window_count, column_step):
            start_frame = window_index * hop_size
            wav_file.setpos(start_frame)
            raw = wav_file.readframes(actual_fft_size)
            samples = decode_pcm_frames(raw, sample_width, channels)
            if len(samples) < actual_fft_size:
                samples.extend([0.0] * (actual_fft_size - len(samples)))

            spectrum = fft([sample * window[index] for index, sample in enumerate(samples)])
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

    return cells, duration, frequency_limit, actual_fft_size


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


def write_frequency_time_svg(
    wav_path: Path,
    output_path: Path,
    fft_size: int = SPECTROGRAM_FFT_SIZE,
    max_frequency_hz: float = MAX_SPECTRUM_HZ,
) -> None:
    cells, duration, frequency_limit, actual_fft_size = spectrogram_cells(
        wav_path,
        fft_size,
        max_frequency_hz,
    )
    width = 1000
    height = 560
    left = 78
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
        svg_text(width / 2, 32, f"Frequency vs Time: {wav_path.name}", 20),
        svg_text(width / 2, 52, f"Spectrogram FFT size {actual_fft_size}", 12),
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
            svg_text(22, top + plot_height / 2, "Frequency (Hz)", 13),
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
        svg_text(width / 2, 30, "Dominant Frequency by WAV File", 22),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze WAV files with FFT.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--fft-size", type=int, default=FFT_SIZE)
    parser.add_argument("--max-spectrum-hz", type=float, default=MAX_SPECTRUM_HZ)
    parser.add_argument("--spectrogram-fft-size", type=int, default=SPECTROGRAM_FFT_SIZE)
    args = parser.parse_args()

    if args.fft_size & (args.fft_size - 1):
        raise SystemExit("--fft-size must be a power of two")
    if args.spectrogram_fft_size & (args.spectrogram_fft_size - 1):
        raise SystemExit("--spectrogram-fft-size must be a power of two")

    wav_files = sorted(args.data_dir.glob("*.wav"))
    if not wav_files:
        raise SystemExit(f"No WAV files found in {args.data_dir}")

    fft_dir = args.results_dir / "fft"
    fft_relevant_dir = args.results_dir / "fft_relevant"
    waveform_dir = args.results_dir / "waveforms"
    spectrum_dir = args.results_dir / "spectra"
    spectrum_relevant_dir = args.results_dir / "spectra_relevant"
    frequency_time_dir = args.results_dir / "frequency_time"
    frequency_time_relevant_dir = args.results_dir / "frequency_time_relevant"
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
    for output_dir in (
        args.results_dir,
        fft_dir,
        fft_relevant_dir,
        waveform_dir,
        spectrum_dir,
        spectrum_relevant_dir,
        frequency_time_dir,
        frequency_time_relevant_dir,
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
    ):
        output_dir.mkdir(parents=True, exist_ok=True)

    dominant_rows: list[dict[str, int | float | str]] = []
    cutoff_rows: list[dict[str, int | float | str]] = []
    filtered_audio_rows: list[dict[str, int | float | str]] = []
    for wav_file in wav_files:
        peak_frequency_hz, peak_magnitude = write_fft_csv(wav_file, fft_dir, args.fft_size)
        _, _, bins = fft_wav_file(wav_file, args.fft_size)
        relevant_max_frequency_hz = relevant_frequency_cutoff(
            bins,
            args.max_spectrum_hz,
            RELEVANT_MIN_HZ,
            RELEVANT_THRESHOLD_FRACTION,
        )
        write_fft_csv(
            wav_file,
            fft_relevant_dir,
            args.fft_size,
            relevant_max_frequency_hz,
        )
        metadata = wav_metadata(wav_file)
        dominant_rows.append(
            {
                "file": wav_file.name,
                "dominant_frequency_hz": peak_frequency_hz,
                "dominant_magnitude": peak_magnitude,
                "duration_seconds": metadata["duration_seconds"],
                "sample_rate_hz": metadata["sample_rate_hz"],
                "channels": metadata["channels"],
                "fft_size": args.fft_size,
            }
        )
        cutoff_rows.append(
            {
                "file": wav_file.name,
                "relevant_max_frequency_hz": relevant_max_frequency_hz,
            }
        )

        relevant_bins = [
            item for item in bins
            if item[0] <= relevant_max_frequency_hz
        ]
        relevant_peak_frequency_hz, _ = dominant_frequency(relevant_bins)
        filtered_full_wav = filtered_full_wav_dir / f"{wav_file.stem}_full_dominant_only.wav"
        filtered_relevant_wav = filtered_relevant_wav_dir / f"{wav_file.stem}_relevant_dominant_only.wav"
        write_single_frequency_wav(wav_file, filtered_full_wav, peak_frequency_hz)
        write_single_frequency_wav(wav_file, filtered_relevant_wav, relevant_peak_frequency_hz)
        filtered_audio_rows.extend(
            [
                {
                    "source_file": wav_file.name,
                    "mode": "full",
                    "kept_frequency_hz": peak_frequency_hz,
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

        write_waveform_svg(wav_file, waveform_dir / f"{wav_file.stem}_waveform.svg")
        write_spectrum_svg(
            wav_file,
            spectrum_dir / f"{wav_file.stem}_fft_spectrum.svg",
            args.fft_size,
            args.max_spectrum_hz,
        )
        write_spectrum_svg(
            wav_file,
            spectrum_relevant_dir / f"{wav_file.stem}_fft_spectrum_relevant.svg",
            args.fft_size,
            relevant_max_frequency_hz,
        )
        write_frequency_time_svg(
            wav_file,
            frequency_time_dir / f"{wav_file.stem}_frequency_time.svg",
            args.spectrogram_fft_size,
            args.max_spectrum_hz,
        )
        write_frequency_time_svg(
            wav_file,
            frequency_time_relevant_dir / f"{wav_file.stem}_frequency_time_relevant.svg",
            args.spectrogram_fft_size,
            relevant_max_frequency_hz,
        )
        write_waveform_svg(
            filtered_full_wav,
            filtered_full_waveform_dir / f"{wav_file.stem}_full_dominant_only_waveform.svg",
        )
        write_spectrum_svg(
            filtered_full_wav,
            filtered_full_spectrum_dir / f"{wav_file.stem}_full_dominant_only_spectrum.svg",
            args.fft_size,
            args.max_spectrum_hz,
        )
        write_frequency_time_svg(
            filtered_full_wav,
            filtered_full_frequency_time_dir / f"{wav_file.stem}_full_dominant_only_frequency_time.svg",
            args.spectrogram_fft_size,
            args.max_spectrum_hz,
        )
        write_waveform_svg(
            filtered_relevant_wav,
            filtered_relevant_waveform_dir / f"{wav_file.stem}_relevant_dominant_only_waveform.svg",
        )
        write_spectrum_svg(
            filtered_relevant_wav,
            filtered_relevant_spectrum_dir / f"{wav_file.stem}_relevant_dominant_only_spectrum.svg",
            args.fft_size,
            relevant_max_frequency_hz,
        )
        write_frequency_time_svg(
            filtered_relevant_wav,
            filtered_relevant_frequency_time_dir / f"{wav_file.stem}_relevant_dominant_only_frequency_time.svg",
            args.spectrogram_fft_size,
            relevant_max_frequency_hz,
        )

    write_dominant_csv(dominant_rows, args.results_dir / "dominant_frequencies.csv")
    write_dominant_svg(dominant_rows, args.results_dir / "dominant_frequencies.svg")
    write_relevant_cutoffs_csv(cutoff_rows, args.results_dir / "relevant_frequency_cutoffs.csv")
    write_filtered_audio_summary_csv(filtered_audio_rows, args.results_dir / "filtered_audio_summary.csv")

    print(f"Analyzed {len(wav_files)} WAV files")
    print(f"Wrote FFT CSV files to: {fft_dir}")
    print(f"Wrote relevant FFT CSV files to: {fft_relevant_dir}")
    print(f"Wrote waveform plots to: {waveform_dir}")
    print(f"Wrote FFT spectrum plots to: {spectrum_dir}")
    print(f"Wrote relevant FFT spectrum plots to: {spectrum_relevant_dir}")
    print(f"Wrote frequency-time plots to: {frequency_time_dir}")
    print(f"Wrote relevant frequency-time plots to: {frequency_time_relevant_dir}")
    print(f"Wrote full-spectrum dominant-only audio and plots to: {filtered_full_dir}")
    print(f"Wrote relevant-spectrum dominant-only audio and plots to: {filtered_relevant_dir}")
    print(f"Wrote dominant frequency summary to: {args.results_dir}")


if __name__ == "__main__":
    main()
