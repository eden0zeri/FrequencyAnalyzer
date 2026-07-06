# Frequency Analyzer

This project analyzes WAV recordings in `Data/` using a pure-Python Fast Fourier Transform.

The input recordings are read as PCM audio, converted to normalized mono samples, and analyzed with overlapping Hann-windowed FFT frames across the whole file. For the TASCAM files currently in `Data/`, the sample rate is 48,000 Hz, meaning 48,000 samples per second per channel.

## Dominant Frequency

`dominant_frequency_hz` is the frequency bin with the largest average FFT magnitude across the full WAV file.

In earlier versions, the dominant frequency came from only the first FFT window. That could be misleading when a recording changed over time. The current script analyzes overlapping windows from the beginning to the end of each WAV, averages the magnitude of each frequency bin, and chooses the bin with the highest mean magnitude.

Magnitude is what determines which frequency is dominant. FFT output values are complex numbers, and the script uses their magnitude to measure how strongly each frequency appears.

## Run

```bash
python3 frequency.py
```

Useful options:

```bash
python3 frequency.py --fft-size 8192 --hop-size 4096
python3 frequency.py --min-frequency-hz 20
python3 frequency.py --max-frequency-hz 2000
python3 frequency.py --no-auto-zoom
python3 frequency.py --no-audit-results
```

Defaults:

- `--fft-size 8192`
- `--hop-size fft_size // 2`
- `--min-frequency-hz 20.0`
- `--recommended-min-frequency-hz 50.0`
- `--max-frequency-hz` has no upper limit unless provided
- `--plot-max-frequency-hz 2000.0` controls the full-range plot y-axis, not the full-file dominant-frequency search
- auto zoom and result auditing are enabled by default

## Outputs

- `results/dominant_frequencies.csv`: full-file dominant frequency summary.
- `results/dominant_frequencies.svg`: bar chart of full-file dominant frequencies.
- `results/dominant_over_time/`: per-frame dominant frequency CSVs.
- `results/dominant_over_time_svg/`: dominant frequency vs time plots.
- `results/dominant_over_time_zoomed_svg/`: zoomed dominant frequency vs time plots.
- `results/fft/`: first-window FFT CSVs, kept for comparison.
- `results/fft_average/`: full-file average FFT CSVs.
- `results/fft_relevant/`: full-file average FFT data truncated to the relevant frequency cutoff.
- `results/waveforms/`: waveform plots.
- `results/spectra/`: full-file average spectrum plots.
- `results/spectra_zoomed/`: zoomed full-file average spectrum plots.
- `results/spectra_relevant/`: relevant-frequency spectrum plots.
- `results/spectra_relevant_zoomed/`: zoomed relevant-frequency spectrum plots.
- `results/frequency_time/`: spectrograms.
- `results/frequency_time_zoomed/`: zoomed spectrograms.
- `results/frequency_time_relevant/`: relevant-frequency spectrograms.
- `results/frequency_time_relevant_zoomed/`: zoomed relevant-frequency spectrograms.
- `results/relevant_frequency_cutoffs.csv`: cutoff chosen for each file.
- `results/filtered_full/`: WAVs and plots reconstructed from the full-file dominant frequency only.
- `results/filtered_relevant/`: WAVs and plots reconstructed from the dominant frequency inside the relevant cutoff only.
- `results/filtered_audio_summary.csv`: frequencies used for filtered WAV generation.
- `results/results_audit.csv`: machine-readable plot and result audit.
- `results/results_audit.md`: human-readable audit summary.

## Interpreting the results

A mostly dark spectrogram is not necessarily wrong. It often means the useful signal energy is concentrated in a much smaller frequency band than the full plotted y-axis. When that happens, use the matching zoomed plot. The script automatically detects useful frequency ranges and creates zoomed spectra or spectrograms when the default axis would hide the visible band.

Dominant frequency is the frequency with the largest spectral magnitude in the selected search range. Magnitude, not the real or imaginary part alone, determines which frequency is dominant.

Very low dominant frequencies, especially below 50 Hz, may be rumble, wind, handling noise, electrical or mechanical vibration, or real signal depending on the experiment. The program does not silently remove those frequencies; it reports them and flags them in the audit.

Filtered dominant-only WAV files should usually show a narrow horizontal band in their spectrograms. That sparse appearance is expected because those files intentionally keep only one dominant frequency.

Read `results/results_audit.md` first when deciding which plots need attention. Use `results/results_audit.csv` when you want the detailed per-output audit rows.

## Notes

The filtered WAV files are intentionally extreme: they keep only one dominant frequency. They are useful for comparison and inspection, but they are not meant to sound like natural denoised audio.

The script does not require NumPy, SciPy, Matplotlib, or any other third-party package.
