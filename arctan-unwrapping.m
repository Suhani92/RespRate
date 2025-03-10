% File path of the HDF5 file
h5FilePath = "C:\acconeerData\breath01sparseiq.h5";

% Dataset path within the HDF5 file
datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";

% Load the data
data = h5read(h5FilePath, datasetPath);

% Extract the real (I) and imaginary (Q) components
I = double(data.real); % Real part
Q = double(data.imag); % Imaginary part

% Calculate the magnitude and phase
magnitude = sqrt(I.^2 + Q.^2); % Magnitude
phase = atan2(Q, I);           % Phase in radians

% Initialize array to store phase at the peak for each frame
numFrames = size(magnitude, 3); % Number of frames
peakPhases = zeros(1, numFrames);

% Loop through each frame to find the peak phase
for frame = 1:numFrames
    % Extract the magnitude and phase for the current frame
    currentMagnitude = magnitude(:, :, frame);
    currentPhase = phase(:, :, frame);
    
    % Find the index of the maximum magnitude
    [~, peakIndex] = max(currentMagnitude(:)); 
    
    % Convert linear index to subscripts
    [row, col] = ind2sub(size(currentMagnitude), peakIndex);
    
    % Get the phase at the peak
    peakPhases(frame) = currentPhase(row, col);
end

% Unwrap the phase to remove discontinuities
unwrappedPhase = unwrap(peakPhases);

% Convert unwrapped phase from radians to degrees (optional)
unwrappedPhaseDeg = rad2deg(unwrappedPhase);

% Create the frame numbers
frameNumbers = 1:numFrames;

% Plot the unwrapped phase at the peak for each frame
figure;
plot(frameNumbers, unwrappedPhaseDeg, 'LineWidth', 1.5);
grid on;
title('Unwrapped Phase at the Peak of Each Frame');
xlabel('Frame Number');
ylabel('Unwrapped Peak Phase (Degrees)');
