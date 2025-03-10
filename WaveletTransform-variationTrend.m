% Load the data
data = h5read("C:\acconeerData\breath1sparseIQ.h5", ...
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

% Decomposition using MODWT (Nondecimated Wavelet Transform)
numberOfLevels = 6; % Number of levels for wavelet decomposition
wt = modwt(phi, "sym6", numberOfLevels); % Using Daubechies wavelet ("db1")

% Energy by level for the decomposition
energyByLevel = 100 * sum((wt.^2), 2) / sum(wt.^2, "all"); % Calculate energy percentage by level

% Plot the original phase waveform
figure;
subplot(2,1,1); % First subplot for phase waveform
plot(1:length(phi), phi, 'LineWidth', 1.5);
xlabel('Frame Index (sweeps)');
ylabel('Phase (radians)');
title('Phase vs. Frames');
grid on;

% Plot the wavelet decomposition coefficients for each level
subplot(2,1,2); % Second subplot for wavelet decomposition
hold on;
for level = 1:numberOfLevels
    plot(wt(level, :), 'DisplayName', ['Level ' num2str(level)], 'LineWidth', 1.5);
end
xlabel('Frame Index (sweeps)');
ylabel('Coefficient Value');
title('Wavelet Decomposition Coefficients');
legend('show');
grid on;

% Save the wavelet decomposition output and energy calculation to a .mat file
output_file = 'C:\acconeerData\breath_wavelet_decomposition.mat'; % Path to save the .mat file

% Save variables to .mat file
save(output_file, 'wt', 'energyByLevel', 'phi', 'fs', 'range_start', 'range_end', 'range_indices', 'D', 'tau_iq', 'f_low');

% Display confirmation message
disp(['Wavelet decomposition data saved to ', output_file]);

% 
% % Load the data
% data = h5read("C:\acconeerData\heartrate2sparseIQ.h5", ...
%               "/sessions/session_0/group_0/entry_0/result/frame");
% 
% % Extract real and imaginary parts and convert to double
% realPart = double(data.real); % [40×32×1794 int16]
% imagPart = double(data.imag);
% 
% % Combine real and imaginary parts into complex IQ data
% IQ_data = complex(realPart, imagPart); % [40×32×1794]
% 
% % Parameters
% fs = 100; % Sweep rate (Hz)
% range_spacing = 0.5e-3; % Range spacing (m)
% D = 100; % Downsampling factor
% tau_iq = 0.5; % Time constant for low-pass filter (seconds)
% f_low = 0.1; % High-pass filter cutoff frequency (Hz)
% 
% % Compute the magnitude of IQ data (sweeps x range bins)
% magnitude_data = abs(IQ_data); % Magnitude = sqrt(real^2 + imag^2)
% 
% % Find the range bin with the highest peak magnitude (across all sweeps)
% [max_magnitude, peak_range_index] = max(mean(magnitude_data, 3)); % Mean over sweeps
% 
% % Define the range start and end based on the peak range index
% range_start_bin = peak_range_index - 5; % Adjust to capture the region around the peak
% range_end_bin = peak_range_index + 5;   % Adjust as needed
% 
% % Check bounds to avoid exceeding the range indices
% range_start_bin = max(1, range_start_bin);
% range_end_bin = min(size(IQ_data, 2), range_end_bin);
% 
% % Define the range values corresponding to the selected range indices
% range_start = (range_start_bin - 1) * range_spacing; % Convert index to meters
% range_end = (range_end_bin - 1) * range_spacing;     % Convert index to meters
% 
% % Determine the range indices corresponding to the peak magnitude region
% range_indices = find((0:size(IQ_data, 2) - 1) * range_spacing >= range_start & ...
%                      (0:size(IQ_data, 2) - 1) * range_spacing <= range_end);
% 
% % Downsampling
% downsampled_data = IQ_data(:, range_indices(1:D:end), :); % x_D[s,d]
% num_downsampled_ranges = size(downsampled_data, 2);
% 
% % Temporal low-pass filter parameters
% alpha_iq = exp(-2 / (tau_iq * fs)); % Low-pass filter coefficient
% 
% % Initialize filtered data
% filtered_data = zeros(size(downsampled_data)); % Initialize with same size
% filtered_data(:, :, 1) = downsampled_data(:, :, 1); % First frame unfiltered
% 
% % Apply temporal low-pass filter
% for s = 2:size(downsampled_data, 3)
%     filtered_data(:, :, s) = alpha_iq * filtered_data(:, :, s - 1) + ...
%                              (1 - alpha_iq) * downsampled_data(:, :, s);
% end
% 
% % Phase unwrapping and high-pass filtering parameters
% alpha_phi = exp(-2 * f_low / fs); % High-pass filter coefficient
% 
% % Initialize phase values
% phi = zeros(1, size(filtered_data, 3)); % Phase for each sweep
% 
% % Calculate phase for each sweep
% for s = 2:size(filtered_data, 3)
%     % Sum over all range indices after downsampling
%     z = sum(filtered_data(:, :, s) .* conj(filtered_data(:, :, s - 1)), 'all');
%     % Compute phase difference and apply high-pass filter
%     phi(s) = alpha_phi * phi(s - 1) + angle(z);
% end
% 
% % Decomposition using MODWT (Nondecimated Wavelet Transform)
% numberOfLevels = 6; % Number of levels for wavelet decomposition
% wt = modwt(phi, "sym6", numberOfLevels); % Using Symlet wavelet ("sym6")
% 
% % Energy by level for the decomposition
% energyByLevel = 100 * sum((wt.^2), 2) / sum(wt.^2, "all"); % Calculate energy percentage by level
% 
% % Create a single figure to plot all levels
% figure;
% 
% % Plot the original phase waveform in the first subplot
% subplot(numberOfLevels + 1, 1, 1); % First subplot for phase waveform
% plot(1:length(phi), phi, 'LineWidth', 1.5);
% xlabel('Frame Index (sweeps)');
% ylabel('Phase (radians)');
% title('Phase vs. Frames');
% grid on;
% 
% % Plot the wavelet decomposition coefficients for each level in subsequent subplots
% for level = 1:numberOfLevels
%     subplot(numberOfLevels + 1, 1, level + 1); % Next subplots for decomposition levels
%     plot(wt(level, :), 'LineWidth', 1.5);
%     xlabel('Frame Index (sweeps)');
%     ylabel(['Coefficient Value (Level ' num2str(level) ')']);
%     title(['Wavelet Decomposition - Level ' num2str(level)]);
%     grid on;
% end
% 
% % Save the wavelet decomposition output and energy calculation to a .mat file
% output_file = 'C:\acconeerData\breath_wavelet_decomposition.mat'; % Path to save the .mat file
% 
% % Save variables to .mat file
% save(output_file, 'wt', 'energyByLevel', 'phi', 'fs', 'range_start', 'range_end', 'range_indices', 'D', 'tau_iq', 'f_low');
% 
% % Display confirmation message
% disp(['Wavelet decomposition data saved to ', output_file]);
