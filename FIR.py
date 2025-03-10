import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt

# File path
file_path = r"C:\acconeerData\breath01sparseiq.h5"

# Load data
with h5py.File(file_path, "r") as f:
    frame = f["sessions/session_0/group_0/entry_0/result/frame"]
    real_part = np.array(frame["real"], dtype=np.float64)  
    imag_part = np.array(frame["imag"], dtype=np.float64)  

# Combine into complex IQ data
IQ_data = real_part + 1j * imag_part  
IQ_data = IQ_data.transpose(2, 1, 0)  

# Parameters
fs = 100  # Sampling frequency
range_spacing = 0.5e-3  
D = 100  
tau_iq = 0.04  
f_low, f_high = 0.2, 3.0  # Bandpass cutoff frequencies

# Compute magnitude and find strongest range bin
magnitude_data = np.abs(IQ_data)
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)

range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  

# Low-pass filtering
alpha_iq = np.exp(-2 / (tau_iq * fs))  
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# Compute phase
phi = np.zeros(filtered_data.shape[2])
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = np.angle(z)

# FIR Bandpass Filter
num_taps = 101  
fir_coeff = firwin(num_taps, [f_low, f_high], pass_zero=False, fs=fs)
filtered_phi_fir = filtfilt(fir_coeff, 1, phi)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(phi, label='Original Phase', alpha=0.6)
plt.plot(filtered_phi_fir, label='FIR Filtered Phase', linewidth=2, color='r')
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Phase (radians)')
plt.title('Phase vs. Frames (FIR Filter)')
plt.legend()
plt.grid(True)
plt.show()
