import numpy as np
import h5py
import matplotlib.pyplot as plt
from vmdpy import VMD  # Import VMD function

# File path
file_path = r"C:\acconeerData\amritoutdoor\Laying\amrit03laying.h5"

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
fs = 100  # Sweep rate (Hz)
range_spacing = 0.5e-3  # Range spacing (m)
D = 100  # Downsampling factor
tau_iq = 0.04  # Time constant for low-pass filter (seconds)
f_low = 0.2 # High-pass filter cutoff frequency (Hz)

# Compute magnitude of IQ data
magnitude_data = np.abs(IQ_data)

# Find the range bin with the highest peak magnitude
mean_magnitude = np.mean(magnitude_data, axis=2)
peak_range_index = np.argmax(mean_magnitude, axis=1)

# Select the range indices around the peak
range_start_bin = max(0, peak_range_index[0] - 5)
range_end_bin = min(IQ_data.shape[1], peak_range_index[0] + 5)
range_indices = np.arange(range_start_bin, range_end_bin + 1)

# Downsampling
downsampled_data = IQ_data[:, range_indices[::D], :]  # Shape: (40, downsampled ranges, 1794)

# Low-pass filter parameters
alpha_iq = np.exp(-2 / (tau_iq * fs))

# Initialize filtered data
filtered_data = np.zeros_like(downsampled_data)
filtered_data[:, :, 0] = downsampled_data[:, :, 0]

# Apply temporal low-pass filter
for s in range(1, downsampled_data.shape[2]):
    filtered_data[:, :, s] = alpha_iq * filtered_data[:, :, s - 1] + \
                             (1 - alpha_iq) * downsampled_data[:, :, s]

# High-pass filter coefficient
alpha_phi = np.exp(-2 * f_low / fs)

# Compute phase
phi = np.zeros(filtered_data.shape[2])  # Phase for each sweep
for s in range(1, filtered_data.shape[2]):
    z = np.sum(filtered_data[:, :, s] * np.conj(filtered_data[:, :, s - 1]))
    phi[s] = alpha_phi * phi[s - 1] + np.angle(z)

# Plot original phase signal
plt.figure(figsize=(12, 6))
plt.plot(range(len(phi)), phi, linewidth=1.5)
plt.xticks(np.arange(0, len(phi), step=100))
plt.xlabel('Frame Index (sweeps)')
plt.ylabel('Phase (radians)')
plt.title('Original Phase vs. Frames')
plt.grid(True)
plt.tight_layout()
plt.show()

### APPLY VARIATIONAL MODE DECOMPOSITION (VMD) ###
# Define VMD parameters
alpha = 2000         # Moderate bandwidth constraint
tau = 0              # Noise-tolerance (no noise)
K = 4                # Number of modes (can be adjusted)
DC = 0               # No DC component
init = 1             # Initialize frequencies
tol = 1e-6           # Convergence tolerance

# Perform VMD
u, u_hat, omega = VMD(phi, alpha, tau, K, DC, init, tol)

# Plot all decomposed modes
plt.figure(figsize=(12, 8))
for i in range(K):
    plt.subplot(K, 1, i + 1)
    plt.plot(u[i, :], linewidth=1.5)
    plt.ylabel(f"IMF {i+1}")
    plt.grid(True)

plt.suptitle("All VMD Intrinsic Mode Functions (IMFs)")
plt.tight_layout()
plt.show()

# Select IMF2
imf2 = u[1, :]

# Plot IMF2 separately
plt.figure(figsize=(12, 6))
plt.plot(imf2, 'g', linewidth=1.5)
plt.xlabel("Frame Index")
plt.ylabel("Amplitude")
plt.title("IMF2 (from VMD)")
plt.grid(True)
plt.show()

# Plot original phase signal with IMF2
plt.figure(figsize=(12, 6))
plt.plot(phi, 'k--', linewidth=1.5, label="Original Phase Signal")
plt.plot(imf2, 'g', linewidth=1.5, label="IMF2")
plt.xlabel("Frame Index")
plt.ylabel("Amplitude")
plt.title("Original Phase Signal and IMF2 (VMD)")
plt.legend()
plt.grid(True)
plt.show()
