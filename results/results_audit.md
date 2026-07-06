# Results Audit

This audit is generated from the numerical FFT data behind each plot.

## Low-Frequency Dominant Components
- `260630_0016.wav`
- `260630_0017.wav`
- `260630_0019.wav`
- `260630_0020.wav`

## Zoomed Plots Created
- `results/dominant_over_time_svg/260630_0016_dominant_over_time.svg` zoomed to 0.0-287.1 Hz
- `results/spectra/260630_0016_fft_spectrum.svg` zoomed to 0.0-287.1 Hz
- `results/frequency_time/260630_0016_frequency_time.svg` zoomed to 0.0-287.1 Hz
- `results/filtered_full/spectra/260630_0016_full_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/frequency_time/260630_0016_full_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/spectra/260630_0016_relevant_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/frequency_time/260630_0016_relevant_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/dominant_over_time_svg/260630_0017_dominant_over_time.svg` zoomed to 0.0-142.2 Hz
- `results/spectra/260630_0017_fft_spectrum.svg` zoomed to 0.0-142.2 Hz
- `results/spectra_relevant/260630_0017_fft_spectrum_relevant.svg` zoomed to 0.0-142.2 Hz
- `results/frequency_time/260630_0017_frequency_time.svg` zoomed to 0.0-142.2 Hz
- `results/frequency_time_relevant/260630_0017_frequency_time_relevant.svg` zoomed to 0.0-142.2 Hz
- `results/filtered_full/spectra/260630_0017_full_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/frequency_time/260630_0017_full_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/spectra/260630_0017_relevant_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/frequency_time/260630_0017_relevant_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/spectra/260630_0018_full_dominant_only_spectrum.svg` zoomed to 0.0-190.6 Hz
- `results/filtered_full/frequency_time/260630_0018_full_dominant_only_frequency_time.svg` zoomed to 0.0-190.6 Hz
- `results/filtered_relevant/spectra/260630_0018_relevant_dominant_only_spectrum.svg` zoomed to 0.0-190.6 Hz
- `results/filtered_relevant/frequency_time/260630_0018_relevant_dominant_only_frequency_time.svg` zoomed to 0.0-190.6 Hz
- `results/filtered_full/spectra/260630_0019_full_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/frequency_time/260630_0019_full_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/spectra/260630_0019_relevant_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/frequency_time/260630_0019_relevant_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/spectra/260630_0020_full_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_full/frequency_time/260630_0020_full_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/spectra/260630_0020_relevant_dominant_only_spectrum.svg` zoomed to 0.0-100.0 Hz
- `results/filtered_relevant/frequency_time/260630_0020_relevant_dominant_only_frequency_time.svg` zoomed to 0.0-100.0 Hz

## Possible Rumble Or Artifact
- `260630_0016.wav` dominant frequency 41.02 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0017.wav` dominant frequency 41.02 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0019.wav` dominant frequency 23.44 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0019.wav` dominant frequency 17.58 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0020.wav` dominant frequency 46.88 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0020.wav` dominant frequency 5.86 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.
- `260630_0020.wav` dominant frequency 23.44 Hz may be rumble, wind, handling noise, electrical/mechanical vibration, or real signal depending on the experiment.

## Dominant-Only Filtered Outputs
- Dominant-only filtered WAV spectrograms are expected to look sparse, usually as a narrow horizontal band.

## Recommendations
- Use zoomed plots when the full y-axis hides low-frequency energy.
- Treat dominant frequencies below 50 Hz with care unless low-frequency motion is expected.
- See `results/results_audit.csv` for one row per audited output.
