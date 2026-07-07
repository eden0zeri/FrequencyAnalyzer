# Frequency Analyzer

Analyze WAV recordings from `Data/` and write clean, interpretable frequency-analysis outputs to `outputs/`.

The analyzer preserves the original WAV files. Resampling, filtering, denoising, pitch extraction, plots, and derived WAVs are written as generated outputs.

## Quick Run

```bash
python3 frequency.py --clean-outputs
```

Useful options:

```bash
python3 frequency.py --analysis-sample-rate-hz 1500 --max-frequency-hz 500 --plot-max-frequency-hz 500
python3 frequency.py --highpass-cutoff-hz 80 --notch-frequency-hz 50 --notch-harmonics 8
python3 frequency.py --segment-duration-seconds 2.5 --segment-overlap-seconds 0
python3 frequency.py --export-debug-plots --export-debug-audio
python3 frequency.py --export-csv
```

Optional Streamlit UI:

```bash
streamlit run streamlit_app.py
```

## Denoised vs Isolated Audio

Use these terms consistently:

- Denoised audio removes unwanted noise while preserving as much of the original recording as possible.
- Isolated audio extracts a target signal and may discard other parts of the original sound.

Default audio outputs:

- `audio/denoised_filtered.wav`: broad denoising with high-pass and notch filters.
- `audio/isolated_pitch_band.wav`: band-limited extraction around the selected/detected non-hum pitch.
- `audio/isolated_pitch_contour.wav`: contour-focused isolation when pitch contour extraction is enabled.

`denoised_filtered.wav` should sound closest to the original recording. The isolated files are for hearing a target pitch region and may sound less natural.

## Clean Default Outputs

For each WAV, the default output folder is:

```text
outputs/<file_stem>/
  waveform.png
  spectrogram_raw.png
  spectrogram_denoised.png
  spectrogram_enhanced_pitch.png
  average_spectrum_db.png
  pitch_contour.png
  audio/
    denoised_filtered.wav
    isolated_pitch_band.wav
    isolated_pitch_contour.wav
  segments/
    segment_000_0.00s_2.50s/
      waveform.png
      spectrogram_raw.png
      spectrogram_denoised.png
      spectrogram_enhanced_pitch.png
      average_spectrum_db.png
      pitch_contour.png
      summary.json
  segment_averages/
    mean_segment_spectrum_db.png
    median_segment_spectrum_db.png
    segment_spectrum_variability.png
    pitch_contours_over_segments.png
    summary.json
  summary.json
  summary.md
```

Normal users should inspect these files first. The default output avoids duplicate full/relevant/zoomed plot families, duplicate SVG/PNG pairs, and CSV clutter.

## Filtration

This program uses several kinds of filters to isolate interesting pitch contours from low-frequency noise, electrical hum, and steady background energy.

In this project, `dominant frequency` generally means the frequency bin that is strongest after averaging across analysis windows. That combines amplitude with persistence. A loud, steady noise can dominate even if it is not the sound we care about, because it appears in many windows and contributes consistently to the average spectrum. A brief target sound can be more interesting while still losing the full-file average contest to long-lasting hum or rumble.

That is why the analyzer separates raw, denoised, enhanced, and isolated outputs instead of trusting the raw dominant frequency alone.

### High-pass filter

- Default: `--highpass-cutoff-hz 80`
- Purpose: reduce low rumble, handling noise, wind, vibration, and very low hum.
- Type: digital Butterworth IIR high-pass filter.
- Implementation: `scipy.signal.butter(..., btype="highpass", output="sos")`, applied with second-order-section filtering.

A high-pass filter keeps frequencies above the cutoff and attenuates frequencies below it. Butterworth filters are used because they have a smooth, maximally flat passband, so they avoid ripple in the retained frequency range.

### Notch filter

- Default: `--notch-frequency-hz 50 --notch-harmonics 8`
- Purpose: reduce fixed electrical/mechanical hum at `50 Hz` or `60 Hz` and harmonics such as `100`, `150`, `200 Hz`.
- Type: narrow second-order IIR notch filter.
- Implementation: `scipy.signal.iirnotch(...)`.

A notch filter is a very narrow band-stop filter. It removes a tight frequency stripe while leaving nearby frequencies mostly intact. This matters because hum is often steady and can dominate average spectra even when it is not perceptually or scientifically interesting.

### Band-pass filter

- Used for: `audio/isolated_pitch_band.wav` and `audio/isolated_pitch_contour.wav`.
- Purpose: extract a target pitch region while discarding lower and higher frequencies.
- Type: digital Butterworth IIR band-pass filter.
- Implementation: `scipy.signal.butter(..., btype="bandpass", output="sos")`.

Band-pass filtering is more aggressive than denoising. It is an isolation tool, not a preservation tool. It can make a target frequency range easier to hear, but it may remove important surrounding sound.

### Resampling anti-alias filter

- Used when `--analysis-sample-rate-hz` is lower than the original WAV sample rate.
- Purpose: safely reduce the analysis sample rate without folding high frequencies into lower frequencies.
- Type: polyphase FIR anti-aliasing resampling filter.
- Implementation: `scipy.signal.resample_poly`.

This is not simple sample dropping. The anti-alias filter removes frequencies that cannot be represented at the lower sample rate before resampling.

### Spectral noise-floor suppression

- Used for: `spectrogram_enhanced_pitch.png` and pitch-contour extraction.
- Purpose: reveal time-varying pitch contours that are hidden by steady background noise.
- Type: STFT-domain magnitude enhancement, not a normal time-domain audio filter.
- Implementation: compute an STFT, estimate a per-frequency noise floor using a percentile across time, subtract that floor, clip negative values to zero, convert to dB, then overlay a smoothed pitch contour.

This enhanced view is for visualization and pitch extraction. It does not replace the raw spectrogram. The raw spectrogram is still generated so you can check what was actually recorded.

### FFT and STFT analysis

Average spectra use NumPy `numpy.fft.rfft` and `numpy.fft.rfftfreq`, which are appropriate for real-valued audio. Spectrograms use `scipy.signal.stft`, which splits audio into short overlapping windows and computes a spectrum for each time window.

The key tradeoff is time resolution versus frequency resolution. Longer windows separate nearby frequencies better, while shorter windows show faster changes more clearly.

### Why filtering can hide real signal

Every filter is a choice. If a real signal overlaps the noise band, filtering can reduce or remove it. For example, a high-pass filter at `80 Hz` will suppress real content below `80 Hz`, and a notch at `100 Hz` can reduce real signal near `100 Hz`.

For that reason, normal outputs include:

- `spectrogram_raw.png`: what the recording contains before denoising.
- `spectrogram_denoised.png`: what remains after high-pass and notch filtering.
- `spectrogram_enhanced_pitch.png`: a visualization-focused view for pitch contours.

Compare these before deciding whether a filter setting is too aggressive.

## Debug Outputs

Debug outputs are available but off by default.

Use:

```bash
python3 frequency.py --export-debug-plots --export-debug-audio
```

Debug plots may include pre-filter spectra, post-filter spectra, full-range raw spectrograms, and intermediate diagnostic plots. Debug audio may include:

```text
audio/debug/highpass_only.wav
audio/debug/notch_only.wav
audio/debug/highpass_notch.wav
audio/debug/dominant_sine_only.wav
```

Debug files are intentionally verbose and are only for checking whether each filter is doing what it should.

## CSV Policy

CSV files are not generated by default.

Use:

```bash
python3 frequency.py --export-csv
```

This writes focused CSV files such as:

- `csv/average_spectrum.csv`
- `csv/segment_summary.csv`
- `csv/pitch_peaks_over_time.csv`

## Cleanup

Use `--clean-outputs` to delete and regenerate the selected output directory. It only targets generated output folders.

Use `--dedupe-existing-results` to remove redundant generated files according to the clean output policy. SVG files are removed when a PNG equivalent exists. CSVs are removed unless `--keep-csv` is passed.

## Resampling

Resampling is enabled by default and uses `scipy.signal.resample_poly`, which applies anti-alias filtering. The Nyquist limit is half the analysis sample rate. For example, `1500 Hz` analysis sample rate can represent frequencies up to `750 Hz`.

Resampling affects analysis and plots only; original input WAVs are not modified.

## Dependencies

The analyzer uses NumPy `numpy.fft.rfft` / `numpy.fft.rfftfreq` for real-valued spectra and SciPy `scipy.signal.stft` for spectrograms. Resampling and filtering also use SciPy signal tools. PNG plotting requires Matplotlib.
