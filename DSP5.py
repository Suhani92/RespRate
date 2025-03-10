# # this code plots the phase signal vs frames plot then applies baseline drift on it and then applies 
# # wavelet transformation on it using the db-4 wavelet and then performs fft on it to estimate the bpm

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
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Parameters
fs = 10  # Sampling frequency (Hz)

# Compute the phase signal (already calculated in your code)
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)
range_indices = np.arange(max(0, peak_range_index[0] - 5), min(IQ_data.shape[1], peak_range_index[0] + 5) + 1)
downsampled_data = IQ_data[:, range_indices, :]
tau_iq = 0.5
alpha_iq = np.exp(-2 / (tau_iq * fs))
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]
phi = np.zeros(filtered_data.shape[2])
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = np.angle(z)

# Step 1: Baseline Drift Correction (Detrending)
phi_corrected = detrend(phi, type='linear')

# Step 2: Wavelet Decomposition
wavelet = 'db4'  # Daubechies wavelet
coeffs = pywt.wavedec(phi_corrected, wavelet, level=5)

# Reconstruct relevant components (approximate and detailed coefficients)
# For respiration: Keep levels corresponding to 0.1–0.5 Hz
coeffs_filtered = [np.zeros_like(c) for c in coeffs]
coeffs_filtered[2] = coeffs[2]  # Example: Keep the 3rd level (adjust based on frequency range)

# Reconstruct the respiratory signal
resp_signal = pywt.waverec(coeffs_filtered, wavelet)

# Step 3: Perform FFT to Compute Breathing Rate
def compute_breathing_rate_fft(signal, fs, lowcut=0.1, highcut=0.35):
    """
    Compute the breathing rate using FFT in the specified frequency range.
    """
    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1/fs)  # Frequency bins
    fft_values = np.fft.fft(signal)  # FFT of the signal

    # Mask for the respiration frequency range (0.1–0.35 Hz)
    respiration_mask = (frequencies >= lowcut) & (frequencies <= highcut)

    # Find the dominant frequency in the respiration range
    dominant_frequency = frequencies[respiration_mask][np.argmax(np.abs(fft_values[respiration_mask]))]

    # Convert frequency to breaths per minute (BPM)
    breathing_rate_bpm = dominant_frequency * 60
    return breathing_rate_bpm, frequencies, fft_values

# Compute breathing rate using FFT
breathing_rate_bpm, frequencies, fft_values = compute_breathing_rate_fft(resp_signal, fs)

print(f"Estimated Breathing Rate (FFT): {breathing_rate_bpm:.2f} BPM")

# Plot results
plt.figure(figsize=(12, 10))

# Original Phase Signal
plt.subplot(3, 1, 1)
plt.plot(phi, label="Original Phase Signal")
plt.title("Original Phase Signal")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Corrected Phase Signal
plt.subplot(3, 1, 2)
plt.plot(phi_corrected, label="Corrected Phase Signal", color="orange")
plt.title("Baseline-Corrected Phase Signal")
plt.xlabel("Frame Index")
plt.ylabel("Phase (radians)")
plt.grid()

# Reconstructed Respiratory Signal
plt.subplot(3, 1, 3)
plt.plot(resp_signal, label="Reconstructed Respiratory Signal", color="green")
plt.title("Reconstructed Respiratory Signal (Wavelet)")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude")
plt.grid()

# Plot FFT Spectrum
plt.figure(figsize=(12, 6))
respiration_mask = (frequencies >= 0.1) & (frequencies <= 0.35)
plt.plot(frequencies[respiration_mask], np.abs(fft_values[respiration_mask]), label="FFT Spectrum", color="blue")
plt.axvline(x=breathing_rate_bpm / 60, color='red', linestyle='--', label=f"Dominant Frequency: {breathing_rate_bpm / 60:.2f} Hz")
plt.title("FFT Spectrum of Respiratory Signal (0.1–0.4 Hz)")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()