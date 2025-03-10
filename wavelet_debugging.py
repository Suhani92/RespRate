# Step 1: Baseline Drift Correction
phase_signal_corrected = detrend(phase_signal, type='linear')

# Step 2: Clutter Suppression using Wavelet Denoising
wavelet = 'sym8'  # Using Symlet 8 wavelet
level = 4  # Reduced number of decomposition levels

# Perform wavelet decomposition
coeffs = pywt.wavedec(phase_signal_corrected, wavelet, level=level)

# Level-dependent thresholding
thresholds = [np.std(c) * np.sqrt(2 * np.log(len(c))) for c in coeffs]
coeffs_thresh = [coeffs[0]] + [pywt.threshold(c, t, mode='soft') for c, t in zip(coeffs[1:], thresholds[1:])]

# Reconstruct the denoised signal
phase_signal_denoised = pywt.waverec(coeffs_thresh, wavelet)

# Step 3: Wavelet Transformation for Respiratory Signal Extraction
# Reconstruct the respiratory signal using approximation coefficients
resp_signal = pywt.waverec([coeffs_thresh[0], None, None, None], wavelet)

# Step 4: Smooth the Reconstructed Signal (with less aggressive smoothing)
resp_signal_smoothed = savgol_filter(resp_signal, window_length=7, polyorder=3)

# Step 5: Calculate Respiration Rate Using FFT
fft_values = np.fft.fft(resp_signal_smoothed)
fft_freqs = np.fft.fftfreq(len(resp_signal_smoothed), d=1/fs)

# Focus on the positive frequency range
positive_freqs = fft_freqs[fft_freqs >= 0]
positive_magnitude = np.abs(fft_values[fft_freqs >= 0])

# Broaden the respiratory frequency range for initial detection
respiratory_mask = (positive_freqs >= 0.1) & (positive_freqs <= 0.5)  # Adjust range
respiratory_freqs = positive_freqs[respiratory_mask]
respiratory_magnitude = positive_magnitude[respiratory_mask]

if len(respiratory_magnitude) > 0:
    dominant_freq = respiratory_freqs[np.argmax(respiratory_magnitude)]
    breathing_rate_bpm = dominant_freq * 60  # Convert Hz to BPM
    print(f"Estimated Average Respiratory Rate: {breathing_rate_bpm:.2f} BPM")
else:
    print("No dominant frequency detected in the respiratory range.")

# Step 6: Plot Results
plt.figure(figsize=(12, 10))

# Original Phase Signal
plt.subplot(5, 1, 1)
plt.plot(phase_signal, label="Original Phase Signal")
plt.title("Original Phase Signal")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Corrected Phase Signal (Baseline Drift Removed)
plt.subplot(5, 1, 2)
plt.plot(phase_signal_corrected, label="Corrected Phase Signal", color="orange")
plt.title("Corrected Phase Signal (Baseline Drift Removed)")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Denoised Phase Signal (Clutter Suppressed)
plt.subplot(5, 1, 3)
plt.plot(phase_signal_denoised, label="Denoised Phase Signal", color="green")
plt.title("Denoised Phase Signal (Clutter Suppressed)")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Reconstructed Respiratory Signal
plt.subplot(5, 1, 4)
plt.plot(resp_signal_smoothed, label="Reconstructed Respiratory Signal", color="blue")
plt.title("Reconstructed Respiratory Signal")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude")
plt.grid()

# FFT Magnitude Spectrum
plt.subplot(5, 1, 5)
plt.plot(positive_freqs, positive_magnitude, label="FFT Magnitude", color="purple")
if len(respiratory_magnitude) > 0:
    plt.axvline(x=dominant_freq, color="red", linestyle="--", label=f"Dominant Frequency: {dominant_freq:.2f} Hz")
plt.title("FFT Magnitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()