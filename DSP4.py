# this code applies median filter on the phase signal vs time plot obtained after applying variation trend method 
# then it applies baseline drift and computes the breath rate 
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import medfilt, butter, sosfilt

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
fs = 20  # Sweep rate (Hz)
range_spacing = 0.5e-3  # Range spacing (m)
D = 100  # Downsampling factor
tau_iq = 0.5  # Time constant for low-pass filter (seconds)
f_low = 0.1  # High-pass filter cutoff frequency (Hz)

# Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude (across all sweeps)
mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# Select the range indices based on the peak range bin
range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

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

# Apply median filtering to remove baseline drift
kernel_size = int(fs * 2) + 1  # Window size: 2 seconds (must be odd)
baseline = medfilt(phi, kernel_size=kernel_size)

# Remove baseline drift
phi_corrected = phi - baseline

# Bandpass filter function
def butter_bandpass(lowcut, highcut, fs, order=6):
    nyq = 0.50 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    return sos

def bandpass_filter(data, lowcut, highcut, fs, order=6):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    filtered_data = sosfilt(sos, data)
    return filtered_data

# Apply bandpass filter
phi_bandpassed = bandpass_filter(
    phi_corrected, 
    lowcut=0.1,  # Lower respiratory frequency
    highcut=0.5,  # Upper respiratory frequency
    fs=fs
)

# FFT for Breath Rate Estimation
N = len(phi_bandpassed)
T = 1.0 / fs  # sampling interval
yf = np.fft.fft(phi_bandpassed)
xf = np.fft.fftfreq(N, T)[:N//2]

# Compute magnitude spectrum
magnitude_spectrum = 2.0/N * np.abs(yf[0:N//2])

# Focus on respiratory frequency range
respiratory_mask = (xf >= 0.1) & (xf <= 0.5)
respiratory_freqs = xf[respiratory_mask]
respiratory_magnitudes = magnitude_spectrum[respiratory_mask]

# Find the peak frequency in the respiratory range
peak_respiratory_index = np.argmax(respiratory_magnitudes)
breath_rate = respiratory_freqs[peak_respiratory_index] * 60  # Convert to breaths per minute

# Plotting results
plt.figure(figsize=(15, 12))

# Original phase signal
plt.subplot(4, 1, 1)
plt.plot(phi, label="Original Phase Signal")
plt.title("Original Phase Signal")
plt.xlabel("Frame Index (sweeps)")
plt.ylabel("Phase (radians)")
plt.grid()

# Estimated baseline
plt.subplot(4, 1, 2)
plt.plot(baseline, label="Estimated Baseline (Median Filter)", color="orange")
plt.title("Estimated Baseline")
plt.xlabel("Frame Index (sweeps)")
plt.ylabel("Phase (radians)")
plt.grid()

# Corrected phase signal
plt.subplot(4, 1, 3)
plt.plot(phi_corrected, label="Corrected Phase Signal", color="green")
plt.title("Corrected Phase Signal (After Baseline Removal)")
plt.xlabel("Frame Index (sweeps)")
plt.ylabel("Phase (radians)")
plt.grid()

# Magnitude Spectrum with Respiratory Range
plt.subplot(4, 1, 4)
plt.plot(xf, magnitude_spectrum)
plt.title(f'Magnitude Spectrum (Estimated Breath Rate: {breath_rate:.2f} breaths/min)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 1)  # Limit x-axis to 1 Hz
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Estimated Breath Rate: {breath_rate:.2f} breaths/minute")
