import numpy as np
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# File path
file_path = r"C:\Users\GOPAL\Downloads\amritoutdoor (2)\amritoutdoor\Sitting\amrit03sittingL.h5"

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

# IIR Bandpass Filter
order = 5cd 
b, a = butter(order, [f_low, f_high], btype='bandpass', fs=fs)
filtered_phi_iir = filtfilt(b, a, phi)

# Plot
plt.figure(figsize=(12, 6))
plt.plot(phi, label='Original Phase', alpha=0.5)
plt.plot(filtered_phi_iir, label='IIR Filtered Phase', linewidth=2, color='g')
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Phase (radians)')
plt.title('Phase vs. Frames (IIR Filter)')
plt.legend()
plt.grid(True)
plt.show()
