**Big picture summary**
  frequency.py does:
  - Readys each .wav file from Data
  - Converts setero audio to mono normalized samples
  - Runs an FFT on the first chunk of each WAV
  - Finds the strongest frequency bin, called the "domninant frequency"
  - Saces CSV files with frequency/magnitude data
  - Saves SVG plots:
    - waveform oveer time
    - FFT spectrum
    - frequency vs time spectrogram
    - dominant frequency histogram
  - Creates new WAV files that keep only the dominant frequency
  - Writes summary CSV files

freqency.py uses its own FFT using pure Python, using a radix-2 Cooley-Tukey FFT, which reursively 
splits the input into even-indexed and odd-indexed samples, computes smaller FFTs, and combines
them with complex phase factors. Input length must be a power of 2.
