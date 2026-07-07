#!/usr/bin/env python3
"""Streamlit UI for frequency.py segment and resampling analysis."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

import frequency


DATA_DIR = Path("Data")
RESULTS_DIR = Path("outputs/ui")


def render_svg(path: Path, height: int = 440) -> None:
    if path.exists():
        components.html(path.read_text(), height=height, scrolling=True)
    else:
        st.info(f"Not generated: {path.name}")


def render_plot(path: Path, height: int = 440) -> None:
    if path.suffix.lower() == ".png" and path.exists():
        st.image(str(path), use_column_width=True)
    elif path.exists():
        render_svg(path, height)
    else:
        st.info(f"Not generated: {path.name}")


def wav_bytes(samples: list[float], sample_rate: int, channels: int) -> bytes:
    with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
        frequency.write_wav_from_mono(samples, sample_rate, channels, Path(temp_file.name))
        return Path(temp_file.name).read_bytes()


def run_analysis(source_wav: Path, output_dir: Path, options: dict[str, object]) -> None:
    staged_input_dir = output_dir / "_input"
    staged_input_dir.mkdir(parents=True, exist_ok=True)
    staged_wav = staged_input_dir / source_wav.name
    staged_wav.write_bytes(source_wav.read_bytes())
    command = [
        sys.executable,
        "frequency.py",
        "--data-dir",
        str(staged_input_dir),
        "--results-dir",
        str(output_dir),
        "--clean-outputs",
        "--analysis-sample-rate-hz",
        str(options["analysis_sample_rate_hz"]),
        "--segment-duration-seconds",
        str(options["segment_duration_seconds"]),
        "--segment-overlap-seconds",
        str(options["segment_overlap_seconds"]),
        "--max-frequency-hz",
        str(options["max_frequency_hz"]),
        "--min-frequency-hz",
        str(options["min_frequency_hz"]),
        "--notch-frequency-hz",
        str(options["notch_frequency_hz"]),
        "--notch-harmonics",
        str(options["notch_harmonics"]),
        "--highpass-cutoff-hz",
        str(options["highpass_cutoff_hz"]),
        "--noise-floor-percentile",
        str(options["noise_floor_percentile"]),
        "--spectrogram-dynamic-range-db",
        str(options["spectrogram_dynamic_range_db"]),
        "--min-pitch-frequency-hz",
        str(options["min_pitch_frequency_hz"]),
    ]
    if options["export_csv"]:
        command.append("--export-csv")
    if options["export_debug_plots"]:
        command.append("--export-debug-plots")
    if options["export_debug_audio"]:
        command.append("--export-debug-audio")
    if not options["resample"]:
        command.append("--no-resample")

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.stdout:
        st.code(completed.stdout)
    if completed.returncode != 0:
        st.error(completed.stderr or "Analysis failed.")
        st.stop()


def main() -> None:
    st.set_page_config(page_title="Frequency Analyzer", layout="wide")
    st.title("Frequency Analyzer")

    wav_files = sorted(DATA_DIR.glob("*.wav"))
    uploaded = st.sidebar.file_uploader("Upload WAV", type=["wav"])
    selected_name = st.sidebar.selectbox(
        "Select WAV",
        [path.name for path in wav_files],
        disabled=uploaded is not None or not wav_files,
    )

    resample = st.sidebar.checkbox("Resample for analysis", value=True)
    analysis_sample_rate_hz = st.sidebar.number_input("Analysis sample rate Hz", min_value=2, value=5000, step=100)
    nyquist_frequency_hz = analysis_sample_rate_hz / 2
    st.sidebar.warning(
        f"At {analysis_sample_rate_hz} Hz sample rate, maximum analyzable frequency is {nyquist_frequency_hz:.0f} Hz."
    )
    min_frequency_hz = st.sidebar.number_input("Minimum frequency Hz", min_value=0.0, value=50.0, step=10.0)
    max_frequency_hz = st.sidebar.number_input("Maximum frequency Hz", min_value=1.0, value=2000.0, step=100.0)
    if max_frequency_hz > nyquist_frequency_hz:
        st.sidebar.warning("Maximum frequency is above Nyquist and will be clamped by the analyzer.")

    segment_duration_seconds = st.sidebar.slider("Segment duration seconds", 0.25, 10.0, 2.5, 0.25)
    segment_overlap_seconds = st.sidebar.slider(
        "Segment overlap seconds",
        0.0,
        max(0.0, segment_duration_seconds - 0.25),
        0.0,
        0.25,
    )
    show_segment_average_spectra = st.sidebar.checkbox("Show segment average spectra", value=True)
    highpass_cutoff_hz = st.sidebar.number_input("High-pass cutoff Hz", min_value=0.0, value=80.0, step=10.0)
    notch_frequency_hz = st.sidebar.number_input("Hum frequency Hz", min_value=0.0, value=50.0, step=10.0)
    notch_harmonics = st.sidebar.number_input("Hum harmonics", min_value=0, value=8, step=1)
    min_pitch_frequency_hz = st.sidebar.number_input("Minimum pitch Hz", min_value=0.0, value=80.0, step=10.0)
    noise_floor_percentile = st.sidebar.slider("Noise percentile", 0.0, 100.0, 20.0, 5.0)
    spectrogram_dynamic_range_db = st.sidebar.slider("Dynamic range dB", 20.0, 120.0, 80.0, 5.0)
    advanced_debug = st.sidebar.checkbox("Advanced/debug outputs", value=False)
    export_csv = st.sidebar.checkbox("Export CSV", value=False)

    st.caption(
        "Full-file analysis summarizes the whole recording. Segment analysis shows local pitch profiles. "
        "Segment-average plots average the numerical segment spectra, not image files."
    )

    if uploaded is None and not wav_files:
        st.info("Upload a WAV or add WAV files to Data/.")
        return

    if uploaded is not None:
        upload_dir = RESULTS_DIR / "_uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        source_wav = upload_dir / uploaded.name
        source_wav.write_bytes(uploaded.getvalue())
    else:
        source_wav = DATA_DIR / selected_name

    output_dir = RESULTS_DIR / source_wav.stem
    if st.button("Analyze", type="primary"):
        run_analysis(
            source_wav,
            output_dir,
            {
                "resample": resample,
                "analysis_sample_rate_hz": analysis_sample_rate_hz,
                "segment_duration_seconds": segment_duration_seconds,
                "segment_overlap_seconds": segment_overlap_seconds,
                "min_frequency_hz": min_frequency_hz,
                "max_frequency_hz": max_frequency_hz,
                "notch_frequency_hz": notch_frequency_hz,
                "notch_harmonics": notch_harmonics,
                "highpass_cutoff_hz": highpass_cutoff_hz,
                "noise_floor_percentile": noise_floor_percentile,
                "spectrogram_dynamic_range_db": spectrogram_dynamic_range_db,
                "min_pitch_frequency_hz": min_pitch_frequency_hz,
                "export_debug_plots": advanced_debug,
                "export_debug_audio": advanced_debug,
                "export_csv": export_csv,
            },
        )
        st.session_state["last_source_wav"] = str(source_wav)
        st.session_state["last_output_root"] = str(output_dir / source_wav.stem)

    output_root_value = st.session_state.get("last_output_root")
    source_wav_value = st.session_state.get("last_source_wav")
    if not output_root_value:
        return

    output_root = Path(output_root_value)
    source_wav = Path(source_wav_value)
    summary_path = output_root / "summary.json"
    if not summary_path.exists():
        st.info("Run analysis to view segment results.")
        return

    summary = json.loads(summary_path.read_text())
    segment_rows = summary.get("segments", [])
    st.subheader("Full-file analysis")
    st.json(summary)

    st.subheader("Segment summary")
    st.dataframe(segment_rows, use_container_width=True)

    if show_segment_average_spectra:
        st.subheader("Segment-averaged spectrum")
        average_dir = output_root / "segment_averages"
        render_plot(average_dir / "mean_segment_spectrum_db.png")
        render_plot(average_dir / "segment_spectrum_variability.png")
        render_plot(average_dir / "pitch_contours_over_segments.png")

    if segment_rows:
        segment_index = st.slider("Explore segment", 0, len(segment_rows) - 1, 0)
        selected_segment = segment_rows[segment_index]
        segment_dir_name = (
            f"segment_{segment_index:03d}_"
            f"{float(selected_segment['start_time_seconds']):.2f}s_"
            f"{float(selected_segment['end_time_seconds']):.2f}s"
        )
        segment_dir = output_root / "segments" / segment_dir_name
        st.metric("Dominant non-hum frequency Hz", f"{float(selected_segment['dominant_non_hum_frequency_hz']):.2f}")
        render_plot(segment_dir / "waveform.png")
        render_plot(segment_dir / "spectrogram_raw.png", 580)
        render_plot(segment_dir / "spectrogram_denoised.png", 580)
        render_plot(segment_dir / "spectrogram_enhanced_pitch.png", 580)
        render_plot(segment_dir / "average_spectrum_db.png")
        render_plot(segment_dir / "pitch_contour.png")
        audio_dir = output_root / "audio"
        for audio_name in ("denoised_filtered.wav", "isolated_pitch_band.wav", "isolated_pitch_contour.wav"):
            audio_path = audio_dir / audio_name
            if audio_path.exists():
                st.audio(audio_path.read_bytes(), format="audio/wav")

        original_samples, original_rate, channels = frequency.read_mono_wav(source_wav)
        start = round(float(selected_segment["start_time_seconds"]) * original_rate)
        end = round(float(selected_segment["end_time_seconds"]) * original_rate)
        segment_samples = original_samples[start:end]
        st.caption("Original segment audio")
        st.audio(wav_bytes(segment_samples, original_rate, channels), format="audio/wav")


if __name__ == "__main__":
    main()
