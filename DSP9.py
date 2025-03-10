# this code performs wavelet transformation on the phase signal using debauchie's db-4 wavelet and does four level decomposition
# and then reconstructs the signal and then performs fft to calculate the breathe per minute
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pywt
from scipy.signal import detrend

# File path
file_path = r"C:\Users\GOPAL\Downloads\amritoutdoor (2)\amritoutdoor\Sitting\amrit03sittingL.h5"
# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)
    imag_part = np.array(frame["imag"], dtype=np.float64)

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (antennas, range bins, sweeps)

# Parameters
fs = 10  # Sampling frequency (Hz)

# Extract phase signal
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)
range_indices = np.arange(max(0, peak_range_index[0] - 5), min(IQ_data.shape[1], peak_range_index[0] + 5) + 1)
filtered_data = IQ_data[:, range_indices, :]
tau_iq = 0.5
alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

phase_signal = np.zeros(filtered_data.shape[2])
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phase_signal[s] = np.angle(z)

# Step 1: Baseline Drift Correction
phase_signal_corrected = detrend(phase_signal, type='linear')  # Linear detrending

# Step 2: Wavelet Decomposition
wavelet = 'db4'
coeffs = pywt.wavedec(phase_signal_corrected, wavelet, level=4)

# Extract approximation and detail coefficients
cA4, cD4, cD3, cD2, cD1 = coeffs

# Step 3: Reconstruct Respiratory Signal Using Approximation Coefficients
resp_signal = pywt.waverec([cA4, None, None, None, None], wavelet)

# Step 4: Calculate Respiration Rate Using FFT
fft_values = np.fft.fft(resp_signal)
fft_freqs = np.fft.fftfreq(len(resp_signal), d=1/fs)

# Focus on the positive frequency range
positive_freqs = fft_freqs[fft_freqs >= 0]
positive_magnitude = np.abs(fft_values[fft_freqs >= 0])

# Broaden the respiratory frequency range for initial detection
respiratory_mask = (positive_freqs >= 0.04) & (positive_freqs <= 0.6)
respiratory_freqs = positive_freqs[respiratory_mask]
respiratory_magnitude = positive_magnitude[respiratory_mask]

if len(respiratory_magnitude) > 0:
    dominant_freq = respiratory_freqs[np.argmax(respiratory_magnitude)]
    breathing_rate_bpm = dominant_freq * 60  # Convert Hz to BPM
    print(f"Estimated Average Respiratory Rate: {breathing_rate_bpm:.2f} BPM")
else:
    print("No dominant frequency detected in the respiratory range.")

# Step 5: Plot Results
plt.figure(figsize=(12, 10))

# Original Phase Signal
plt.subplot(5, 1, 1)
plt.plot(phase_signal, label="Original Phase Signal")
plt.title("Original Phase Signal")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Corrected Phase Signal
plt.subplot(5, 1, 2)
plt.plot(phase_signal_corrected, label="Corrected Phase Signal", color="orange")
plt.title("Corrected Phase Signal (Baseline Drift Removed)")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Approximation Coefficients
plt.subplot(5, 1, 3)
plt.plot(cA4, label="Approximation Coefficients (Level 4)", color="green")
plt.title("Approximation Coefficients (Level 4)")
plt.xlabel("Index")
plt.ylabel("Amplitude")
plt.grid()

# Reconstructed Respiratory Signal
plt.subplot(5, 1, 4)
plt.plot(resp_signal, label="Reconstructed Respiratory Signal", color="blue")
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
