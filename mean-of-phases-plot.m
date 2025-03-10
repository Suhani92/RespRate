% File path of the HDF5 file
h5FilePath = "C:\acconeerData\breath1sparseIQ.h5";

% Dataset path within the HDF5 file
datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";

% Load the data
data = h5read(h5FilePath, datasetPath);

% Extract the real (I) and imaginary (Q) components
I = double(data.real); % Real part
Q = double(data.imag); % Imaginary part

% Calculate the phase using arc tangent demodulation
phase = atan2(Q, I); % Phase in radians

% Average the phase for each frame (optional, to smooth data)
averagePhase = squeeze(mean(phase, [1, 2])); % Average over points in each frame

% Convert phase from radians to degrees (optional)
averagePhaseDeg = rad2deg(averagePhase);

% Get the number of frames
numFrames = size(phase, 3);

% Create the frame numbers
frameNumbers = 1:numFrames;

% Plot the average phase for each frame
figure;
plot(frameNumbers, averagePhaseDeg, 'LineWidth', 1.5);
grid on;
title('Phase Across Frames (Arc Tangent Demodulation)');
xlabel('Frames');
ylabel('Average Phase (Degrees)');



% Unwrapped phase data 
% % File path of the HDF5 file
% h5FilePath = "C:\acconeerData\breath01sparseiq.h5";
% 
% % Dataset path within the HDF5 file
% datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";
% 
% % Load the data
% data = h5read(h5FilePath, datasetPath);
% 
% % Extract the real (I) and imaginary (Q) components
% I = double(data.real); % Real part
% Q = double(data.imag); % Imaginary part
% 
% % Calculate the phase using arc tangent demodulation
% phase = atan2(Q, I); % Phase in radians
% 
% % Average the phase for each frame
% averagePhase = squeeze(mean(phase, [1, 2])); % Average over points in each frame
% 
% % Unwrap the phase to remove discontinuities
% unwrappedPhase = unwrap(averagePhase); % Unwrap in radians
% 
% % Convert both wrapped and unwrapped phases to degrees
% averagePhaseDeg = rad2deg(averagePhase); % Wrapped phase
% unwrappedPhaseDeg = rad2deg(unwrappedPhase); % Unwrapped phase
% 
% % Get the number of frames
% numFrames = size(phase, 3);
% 
% % Create the frame numbers
% frameNumbers = 1:numFrames;
% 
% % Plot both wrapped and unwrapped phases for comparison
% figure;
% plot(frameNumbers, averagePhaseDeg, 'LineWidth', 1.5, 'DisplayName', 'Wrapped Phase');
% hold on;
% plot(frameNumbers, unwrappedPhaseDeg, 'LineWidth', 1.5, 'DisplayName', 'Unwrapped Phase');
% grid on;
% title('Phase Across Frames (Wrapped vs Unwrapped)');
% xlabel('Frames');
% ylabel('Phase (Degrees)');
% legend('Location', 'best');
% hold off;


% % Does an FFT on the phased data 
% % File path of the HDF5 file
% h5FilePath = "C:\acconeerData\breath1sparseIQ.h5";
% 
% % Dataset path within the HDF5 file
% datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";
% 
% % Load the data
% data = h5read(h5FilePath, datasetPath);
% 
% % Extract the real (I) and imaginary (Q) components
% I = double(data.real); % Real part
% Q = double(data.imag); % Imaginary part
% 
% % Calculate the phase using arc tangent demodulation
% phase = atan2(Q, I); % Phase in radians
% 
% % Average the phase for each frame
% averagePhase = squeeze(mean(phase, [1, 2])); % Average over points in each frame
% 
% % Unwrap the phase to remove discontinuities
% unwrappedPhase = unwrap(averagePhase); % Unwrapped phase in radians
% 
% % Sampling frequency (estimate or define based on the frame rate of the radar)
% fs = 20; % Example: 20 Hz frame rate (adjust based on your radar setup)
% 
% % Perform FFT on the unwrapped phase
% N = length(unwrappedPhase); % Number of samples
% fftResult = fft(unwrappedPhase); % FFT computation
% frequencies = (0:N-1) * (fs / N); % Frequency axis (Hz)
% 
% % Magnitude of FFT (scaled)
% magnitudeFFT = abs(fftResult) / N;
% 
% % Only take the first half of the spectrum (positive frequencies)
% halfIdx = 1:floor(N/2);
% frequencies = frequencies(halfIdx); % Positive frequencies
% magnitudeFFT = 2 * magnitudeFFT(halfIdx); % Scale for one-sided FFT
% 
% % Plot the unwrapped phase
% figure;
% plot(1:N, rad2deg(unwrappedPhase), 'LineWidth', 1.5);
% grid on;
% title('Unwrapped Phase Across Frames');
% xlabel('Frames');
% ylabel('Unwrapped Phase (Degrees)');
% 
% % Plot the FFT result
% figure;
% plot(frequencies, magnitudeFFT, 'LineWidth', 1.5);
% grid on;
% title('FFT of Unwrapped Phase');
% xlabel('Frequency (Hz)');
% ylabel('Magnitude');
