# this code applies the bandpass filter to the vt phase signal plto and then performs fft for hifghest peak detection
# and gthenfurther calculation of breath per minute
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
fs = 10  # Sweep rate (Hz)
tau_iq = 0.5  # Time constant for low-pass filter (seconds)
f_low = 0.1  # High-pass filter cutoff frequency (Hz)

# Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude (across all sweeps)
mean_magnitude = np.mean(magnitude_data, axis=2)  # Mean over sweeps
peak_range_index = np.argmax(mean_magnitude, axis=1)  # Index for each antenna

# Select range indices
range_start_bin = max(0, peak_range_index[0] - 5)  # Adjust as needed
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
D = 100  # Downsampling factor
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

# Bandpass Filter Design for Breathing
def bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist  # Normalize low cutoff frequency
    high = highcut / nyquist  # Normalize high cutoff frequency
    b, a = butter(order, [low, high], btype="band")  # Design Butterworth bandpass filter
    y = filtfilt(b, a, data)  # Apply the filter using filtfilt for zero-phase filtering
    return y

# Apply the bandpass filter (breathing frequency range: 0.1 Hz to 0.4 Hz)
lowcut = 0.1  
highcut = 0.4  
phi_bandpassed = bandpass_filter(phi, lowcut, highcut, fs, order=5)

# Normalize the bandpassed signal
phi_bandpassed_normalized = (phi_bandpassed - np.min(phi_bandpassed)) / (np.max(phi_bandpassed) - np.min(phi_bandpassed))

# Perform FFT on the bandpassed signal
n = len(phi_bandpassed_normalized)  
fft_values = np.fft.fft(phi_bandpassed_normalized)  
frequencies = np.fft.fftfreq(n, d=1/fs)  

# Focus on the frequency range of interest (0.1–0.4 Hz)
respiratory_mask = (frequencies >= lowcut) & (frequencies <= highcut)
resp_freqs = frequencies[respiratory_mask]
resp_fft_values = fft_values[respiratory_mask]

# Find the dominant frequency in the respiratory range
if len(resp_freqs) > 0:
    peak_index = np.argmax(np.abs(resp_fft_values))  
    peak_freq_hz = resp_freqs[peak_index]  
    breaths_per_minute = peak_freq_hz * 60  
else:
    breaths_per_minute = None

# Plotting Bandpassed Signal
plt.figure(figsize=(12,6))
plt.plot(phi_bandpassed_normalized, label='Bandpassed Signal', color='green')
plt.title('Bandpassed Signal after Filtering')
plt.xlabel('Sample Index')
plt.ylabel('Normalized Amplitude')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plotting FFT Spectrum in Respiratory Range
plt.figure(figsize=(12,6))
plt.plot(resp_freqs, np.abs(resp_fft_values), label="FFT Magnitude", color='blue')
if breaths_per_minute is not None:
    plt.axvline(x=peak_freq_hz, color='red', linestyle='--', label=f"Dominant Frequency: {peak_freq_hz:.2f} Hz")
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Spectrum (0.1–0.4 Hz)')
plt.xlim(0.1, 0.4)   # Limit x-axis to respiratory range of interest
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Print results
if breaths_per_minute is not None:
    print(f"Dominant Frequency: {peak_freq_hz:.2f} Hz")
    print(f"Estimated Breaths Per Minute (BPM): {breaths_per_minute:.2f}")
else:
    print("No valid frequency detected in the specified range.")
