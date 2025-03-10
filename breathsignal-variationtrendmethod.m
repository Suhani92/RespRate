% Load the data
data = h5read("C:\Users\GOPAL\Downloads\breathing_10.h5", ...
              "/sessions/session_0/group_0/entry_0/result/frame");

% Extract real and imaginary parts and convert to double
realPart = double(data.real); % [40×32×1794 int16]
imagPart = double(data.imag);

% Combine real and imaginary parts into complex IQ data
IQ_data = complex(realPart, imagPart); % [40×32×1794]

% Parameters
fs = 100; % Sweep rate (Hz)
range_spacing = 0.5e-3; % Range spacing (m)
D = 100; % Downsampling factor
tau_iq = 0.5; % Time constant for low-pass filter (seconds)
f_low = 0.1; % High-pass filter cutoff frequency (Hz)

% Compute the magnitude of IQ data (sweeps x range bins)
magnitude_data = abs(IQ_data); % Magnitude = sqrt(real^2 + imag^2)

% Find the range bin with the highest peak magnitude (across all sweeps)
[max_magnitude, peak_range_index] = max(mean(magnitude_data, 3)); % Mean over sweeps

% Define the range start and end based on the peak range index
range_start_bin = peak_range_index - 5; % Adjust to capture the region around the peak
range_end_bin = peak_range_index + 5;   % Adjust as needed

% Check bounds to avoid exceeding the range indices
range_start_bin = max(1, range_start_bin);
range_end_bin = min(size(IQ_data, 2), range_end_bin);

% Define the range values corresponding to the selected range indices
range_start = (range_start_bin - 1) * range_spacing; % Convert index to meters
range_end = (range_end_bin - 1) * range_spacing;     % Convert index to meters

% Determine the range indices corresponding to the peak magnitude region
range_indices = find((0:size(IQ_data, 2) - 1) * range_spacing >= range_start & ...
                     (0:size(IQ_data, 2) - 1) * range_spacing <= range_end);

% Downsampling
downsampled_data = IQ_data(:, range_indices(1:D:end), :); % x_D[s,d]
num_downsampled_ranges = size(downsampled_data, 2);

% Temporal low-pass filter parameters
alpha_iq = exp(-2 / (tau_iq * fs)); % Low-pass filter coefficient

% Initialize filtered data
filtered_data = zeros(size(downsampled_data)); % Initialize with same size
filtered_data(:, :, 1) = downsampled_data(:, :, 1); % First frame unfiltered

% Apply temporal low-pass filter
for s = 2:size(downsampled_data, 3)
    filtered_data(:, :, s) = alpha_iq * filtered_data(:, :, s - 1) + ...
                             (1 - alpha_iq) * downsampled_data(:, :, s);
end

% Phase unwrapping and high-pass filtering parameters
alpha_phi = exp(-2 * f_low / fs); % High-pass filter coefficient

% Initialize phase values
phi = zeros(1, size(filtered_data, 3)); % Phase for each sweep

% Calculate phase for each sweep
for s = 2:size(filtered_data, 3)
    % Sum over all range indices after downsampling
    z = sum(filtered_data(:, :, s) .* conj(filtered_data(:, :, s - 1)), 'all');
    % Compute phase difference and apply high-pass filter
    phi(s) = alpha_phi * phi(s - 1) + angle(z);
end

% Plot the phase vs. frames
figure;
plot(1:length(phi), phi, 'LineWidth', 1.5);
xlabel('Frame Index (sweeps)');
ylabel('Phase (radians)');
title('Phase vs. Frames');
grid on;



