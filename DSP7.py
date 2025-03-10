import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# File path
file_path = r"C:\Users\GOPAL\Downloads\amritoutdoor (2)\amritoutdoor\Sitting\amrit03sittingL.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  # Extract real part
    imag_part = np.array(frame["imag"], dtype=np.float64)  # Extract imaginary part

# Combine real and imaginary parts into complex IQ data
IQ_data = real_part + 1j * imag_part  # Shape: (1794, 32, 40)

# Transpose data to match MATLAB's order: (antennas x range bins x sweeps)
IQ_data = IQ_data.transpose(2, 1, 0)  # Shape: (40, 32, 1794)

# Parameters
fs = 50 # Sweep rate (Hz)
tau_iq = 0.5  # Time constant for low-pass filter (seconds)
f_low = 0.1  # High-pass filter cutoff frequency (Hz)

# Downsampling
D = 100  # Downsampling factor
downsampled_data = IQ_data[:, ::D, :]  # Shape: (40, downsampled ranges, 1794)

# Temporal low-pass filter parameters
alpha_iq = np.exp(-2 / (tau_iq * fs))  # Low-pass filter coefficient

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Phase unwrapping and high-pass filtering parameters
alpha_phi = np.exp(-2 * f_low / fs)  # High-pass filter coefficient

# Initialize phase values
phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep

# Calculate phase for each sweep
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

# Bandpass Filter Design for Breathing
def bandpass_filter(data, lowcut, highcut, fs, order=20):
    nyquist = 0.05 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = butter(order, [low, high], btype="band")  # Design Butterworth bandpass filter
    y = filtfilt(b, a, data)  # Apply the filter using filtfilt for zero-phase filtering
    return y

# Apply the bandpass filter (breathing frequency range: 0.1 Hz to 3 Hz)
lowcut = 0.1  # Lower cutoff frequency (Hz)
highcut = 0.3  # Upper cutoff frequency (Hz)
phi_bandpassed = bandpass_filter(phi, lowcut, highcut, fs, order=5)

# Apply Moving Average Filter
def moving_average(data, window_size):
    """Applies a simple moving average filter."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Define the window size (Adjust based on the expected breathing period, e.g., 1-5 seconds)
window_size = int(fs * 0.35)  # 0.5 seconds window
phi_smoothed = moving_average(phi_bandpassed, window_size)

# FFT Analysis
n_fft = len(phi_smoothed)
fft_result = np.fft.fft(phi_smoothed, n=n_fft)
fft_freqs = np.fft.fftfreq(n_fft, d=1/fs)  # Frequency axis

# Only consider positive frequencies
positive_freqs = fft_freqs[:n_fft // 2]
positive_fft = np.abs(fft_result[:n_fft // 2])

# Find dominant frequency (breathing rate)
dominant_freq_idx = np.argmax(positive_fft)
breathing_freq = positive_freqs[dominant_freq_idx]  # Dominant frequency (Hz)
breathing_rate = breathing_freq * 60  # Convert to breaths per minute (BPM)

# Plot the phase and FFT in a single window
plt.figure(figsize=(15, 10))

# Plot the phase signals
plt.subplot(2, 1, 1)
plt.plot(range(len(phi)), phi, label="Original Phase", linewidth=0.8, alpha=0.5)
plt.plot(range(len(phi_bandpassed)), phi_bandpassed, label="Bandpassed Phase", linewidth=1.5, color='orange')
plt.plot(range(len(phi_smoothed)), phi_smoothed, label="Smoothed Phase (Moving Average)", linewidth=2, color='green')
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Phase (radians)')
plt.title('Phase vs. Frames (With Bandpass and Moving Average Filters)')
plt.grid(True)
plt.legend()

# Plot the FFT spectrum
plt.subplot(2, 1, 2)
plt.plot(positive_freqs, positive_fft, label="FFT Spectrum", color='purple')
plt.axvline(x=breathing_freq, color='red', linestyle='--', label=f'Dominant Frequency: {breathing_freq:.2f} Hz')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Print breathing rate
print(f"Breathing Frequency: {breathing_freq:.2f} Hz")
print(f"Breathing Rate: {breathing_rate:.2f} BPM")
