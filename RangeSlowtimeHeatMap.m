% Load the data
data = h5read("C:\acconeerData\breath01sparseiq.h5", ...
              "/sessions/session_0/group_0/entry_0/result/frame");

% Convert real and imaginary parts to double for complex operations
realPart = double(data.real);
imagPart = double(data.imag);

% Combine real and imaginary parts into a complex array
complexData = realPart + 1i * imagPart;

% Check the size of the complex data
disp(size(complexData)); % Should be [number of points, sweeps per frame, number of frames]

% Get the number of frames and range bins
[numRangeBins, numSweeps, numFrames] = size(complexData);

% Define step length and range start/end
stepLength = 0.02; % 20 mm in meters
rangeStart = 0.2;  % Starting at 0.2 meters
rangeEnd = 1.0;    % Ending at 1 meter

% Calculate the range values for each range bin (0.2m to 1m)
rangeBins = linspace(rangeStart, rangeEnd, numRangeBins);

% Initialize matrix to store magnitude for each frame
magnitudeData = zeros(numRangeBins, numFrames);

% Calculate the magnitude for each frame
for frame = 1:numFrames
    singleFrame = complexData(:, :, frame); % Extract a single frame
    magnitudeData(:, frame) = abs(singleFrame(:, 1)); % Take magnitude of the first sweep (or average across sweeps)
end

% Plot the magnitude for each frame with range values on the y-axis
figure;
imagesc(1:numFrames, rangeBins, magnitudeData); % x = frames, y = range bins (distance in meters)
colorbar;
title('Range slow time matrix');
xlabel('Frame Number');
ylabel('Range (meters)');
axis tight;
colormap('jet'); % Optionally choose a color map


