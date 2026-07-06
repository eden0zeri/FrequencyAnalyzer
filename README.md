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
```

Defaults:

- `--fft-size 8192`
- `--hop-size fft_size // 2`
- `--min-frequency-hz 20.0`
- `--max-frequency-hz` has no upper limit unless provided
- `--max-spectrum-hz 2000.0` controls the displayed/trimmed spectrum range, not the full-file dominant-frequency search

## Outputs

- `results/dominant_frequencies.csv`: full-file dominant frequency summary.
- `results/dominant_frequencies.svg`: bar chart of full-file dominant frequencies.
- `results/dominant_over_time/`: per-frame dominant frequency CSVs.
- `results/dominant_over_time_svg/`: dominant frequency vs time plots.
- `results/fft/`: first-window FFT CSVs, kept for comparison.
- `results/fft_average/`: full-file average FFT CSVs.
- `results/fft_relevant/`: full-file average FFT data truncated to the relevant frequency cutoff.
- `results/waveforms/`: waveform plots.
- `results/spectra/`: full-file average spectrum plots.
- `results/spectra_relevant/`: relevant-frequency spectrum plots.
- `results/frequency_time/`: spectrograms.
- `results/frequency_time_relevant/`: relevant-frequency spectrograms.
- `results/relevant_frequency_cutoffs.csv`: cutoff chosen for each file.
- `results/filtered_full/`: WAVs and plots reconstructed from the full-file dominant frequency only.
- `results/filtered_relevant/`: WAVs and plots reconstructed from the dominant frequency inside the relevant cutoff only.
- `results/filtered_audio_summary.csv`: frequencies used for filtered WAV generation.

## Notes

The filtered WAV files are intentionally extreme: they keep only one dominant frequency. They are useful for comparison and inspection, but they are not meant to sound like natural denoised audio.

The script does not require NumPy, SciPy, Matplotlib, or any other third-party package.
