% This code plots the real time range profile with timestamp and distance 
% Load the data
data = h5read("C:\Users\GOPAL\Downloads\breathing_9.h5", ...
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

% Create timestamps for each frame (assuming constant frame interval)
frameInterval = 0.05; % 50 ms (adjust based on actual data rate)
timeStamps = (0:numFrames-1) * frameInterval; % Time in seconds

% Initialize figure for animation
figure;
hPlot = plot(rangeBins, zeros(1, numRangeBins), 'LineWidth', 2); % Placeholder line
title('Real time Range Profile');
xlabel('Range (meters)');
ylabel('Amplitude');
grid on;
axis([rangeStart rangeEnd 0 1.2 * max(abs(complexData(:)))]); % Adjust axis based on expected amplitude

% Add text annotation for timestamp and max amplitude location
hText = text(rangeStart + 0.1, 1.1 * max(abs(complexData(:))), '', ...
             'FontSize', 12, 'FontWeight', 'bold', 'Color', 'black');

% Animation loop
for frame = 1:numFrames
    % Extract the frame and compute the amplitude for all range bins
    singleFrame = complexData(:, :, frame);
    magnitudeProfile = abs(singleFrame(:, 1)); % Take magnitude of the first sweep (or average across sweeps)
    
    % Find the index and value of the maximum amplitude
    [maxAmplitude, maxIndex] = max(magnitudeProfile);
    maxRange = rangeBins(maxIndex); % Convert index to range in meters
    
    % Update the plot data
    set(hPlot, 'YData', magnitudeProfile);
    
    % Update text annotation with timestamp and max amplitude location
    set(hText, 'String', sprintf('Time: %.2f s, Object at @ %.2f m', timeStamps(frame), maxRange));
    
    drawnow;
    
    % Pause for a short duration for better visualization
    pause(0.05); % Adjust this value to control animation speed
end
