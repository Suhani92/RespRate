
import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import fft, fftfreq
from PyEMD import EEMD

# File path
file_path = r"/home/sarthak/breathing_files/amritoutdoor/Laying/amrit01laying.h5"

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

# Clutter suppression: Subtract the mean phase
mean_phi = np.mean(phi)  # Compute the mean phase
phi_clutter_suppressed = phi - mean_phi  # Subtract the mean to remove clutter

# Bandpass filter parameters
lowcut = 0.1  # Low cutoff frequency (Hz)
highcut = 0.5  # High cutoff frequency (Hz)
order = 4  # Filter order

# Design the bandpass filter
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Apply the bandpass filter
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)  # Zero-phase filtering
    return y

# Apply the bandpass filter to the clutter-suppressed phase signal
phi_filtered = bandpass_filter(phi_clutter_suppressed, lowcut, highcut, fs, order)

# Normalize the filtered phase signal
max_abs_phi = np.max(np.abs(phi_filtered))  # Maximum absolute value
phi_normalized = phi_filtered / max_abs_phi  # Normalize to [-1, 1]

# Plot the final normalized phase signal
plt.figure(figsize=(12, 6))
plt.plot(range(len(phi_normalized)), phi_normalized, linewidth=1.5)
plt.xticks(np.arange(0, len(phi_normalized), step=100))  # Set x-axis ticks at intervals of 100
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Normalized Phase')
plt.title('Normalized Phase vs. Frames (0.1–0.4 Hz Bandpass Filtered)')
plt.grid(True)
plt.tight_layout()  # Ensure no clipping
plt.show()

# Function to apply EEMD and extract the respiration signal
def extract_respiration_eemd(signal):
    # Initialize EEMD
    eemd = EEMD()
    
    # Decompose the signal into IMFs
    imfs = eemd(signal)
    
    # Respiration is typically in the low-frequency IMFs (e.g., IMF 3–5)
    respiration_signal = np.sum(imfs[2:5], axis=0)  # Sum IMFs 3–5
    
    return respiration_signal, imfs

# Apply EEMD to the normalized phase signal
respiration_signal, imfs = extract_respiration_eemd(phi_normalized)

# Plot the IMFs (optional, for visualization)
plt.figure(figsize=(12, 8))
for i, imf in enumerate(imfs):
    plt.subplot(len(imfs), 1, i+1)
    plt.plot(imf, linewidth=1.5)
    plt.title(f'IMF {i+1}')
    plt.grid(True)
plt.tight_layout()
plt.show()

# Plot the reconstructed respiration signal
plt.figure(figsize=(12, 6))
plt.plot(range(len(respiration_signal)), respiration_signal, linewidth=1.5, label='Reconstructed Respiration Signal')
plt.xticks(np.arange(0, len(respiration_signal), step=100))  # Set x-axis ticks at intervals of 100
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Amplitude')
plt.title('Reconstructed Respiration Signal (EEMD)')
plt.grid(True)
plt.legend()
plt.tight_layout()  # Ensure no clipping
plt.show()

# Function to apply a moving average filter
def moving_average_filter(signal, window_size):
    # Create a window for averaging
    window = np.ones(window_size) / window_size
    # Apply convolution to compute the moving average
    smoothed_signal = np.convolve(signal, window, mode='same')
    return smoothed_signal

# Apply moving average filter to the reconstructed respiration signal
window_size = 20  # Adjust the window size as needed
respiration_signal_smoothed = moving_average_filter(respiration_signal, window_size)

# Plot the smoothed respiration signal
plt.figure(figsize=(12, 6))
plt.plot(range(len(respiration_signal_smoothed)), respiration_signal_smoothed, linewidth=1.5, color='red', label='Smoothed Respiration Signal')
plt.xticks(np.arange(0, len(respiration_signal_smoothed), step=100))  # Set x-axis ticks at intervals of 100
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Amplitude')
plt.title('Smoothed Respiration Signal (Moving Average Filter)')
plt.grid(True)
plt.legend()
plt.tight_layout()  # Ensure no clipping
plt.show()

# Custom peak detection function
def custom_peak_detection(signal):
    """
    Detect peaks in a signal where dy/dx = 0 and y > 0.
    """
    # Calculate the first derivative of the signal
    dy_dx = np.gradient(signal)
    
    # Find points where the derivative changes from positive to negative (local maxima)
    peaks = []
    for i in range(1, len(dy_dx)):
        if dy_dx[i - 1] > 0 and dy_dx[i] < 0 and signal[i] > 0:
            peaks.append(i)
    
    return np.array(peaks)

# Time-Domain Analysis (Custom Peak Detection)
def estimate_breath_rate_time_domain(signal, fs):
    # Use custom peak detection
    peaks = custom_peak_detection(signal)
    
    # Calculate the total duration of the signal
    total_duration = len(signal) / fs  # Total duration in seconds
    
    # Calculate the breath rate (breaths per minute)
    if total_duration > 0:
        breath_rate = (len(peaks) / total_duration) * 60  # Breaths per minute
    else:
        breath_rate = 0  # No signal duration
    
    return breath_rate, peaks

# Estimate breath rate using custom peak detection
breath_rate_time_domain, peaks = estimate_breath_rate_time_domain(respiration_signal_smoothed, fs)

# Print the number of peaks detected and the calculated breath rate
print(f"Number of Peaks Detected: {len(peaks)}")
print(f"Total Duration (T): {len(respiration_signal_smoothed) / fs:.2f} seconds")
print(f"Breath Rate (Time Domain - Custom Peak Detection): {breath_rate_time_domain:.2f} bpm")

# Plot the smoothed respiration signal with detected peaks (Custom Peak Detection)
plt.figure(figsize=(12, 6))
plt.plot(range(len(respiration_signal_smoothed)), respiration_signal_smoothed, label='Smoothed Respiration Signal', linewidth=1.5)
plt.plot(peaks, respiration_signal_smoothed[peaks], 'ro', label='Detected Peaks (Custom)')
plt.xticks(np.arange(0, len(respiration_signal_smoothed), step=100))  # Set x-axis ticks at intervals of 100
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Amplitude')
plt.title('Smoothed Respiration Signal with Detected Peaks (Custom Peak Detection)')
plt.legend()
plt.grid(True)
plt.tight_layout()  # Ensure no clipping
plt.show()