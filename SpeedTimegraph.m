% This code pltos the speed time graph of the saved data 

% % Load the data
data = h5read("C:\acconeerData\velocity08speeddetector.h5", ...
              "/sessions/session_0/group_0/entry_0/result/frame");

% Convert real and imaginary parts to double for complex operations
realPart = double(data.real);
imagPart = double(data.imag);

% Combine real and imaginary parts into a complex array
complexData = realPart + 1i * imagPart;

% Check the size of the complex data
disp(size(complexData)); % Should be [numRangeBins, numSweeps, numFrames]

% Get the number of frames and range bins
[numRangeBins, numSweeps, numFrames] = size(complexData);

% Calculate the amplitude (magnitude) for each range bin and frame
% Amplitude is the magnitude of the complex data
amplitudeData = abs(complexData); % Size: [numRangeBins, numSweeps, numFrames]

% Aggregate amplitude over range bins and sweeps for each frame
amplitudePerFrame = squeeze(mean(mean(amplitudeData, 1), 2)); % Size: [1, numFrames]

% Plot the amplitude over frames
figure;
plot(1:numFrames, amplitudePerFrame, 'b-', 'LineWidth', 1.5); % Frames on x-axis, amplitude on y-axis
title('Amplitude of I and Q Components Across Frames');
xlabel('Frame Number');
ylabel('Amplitude');
grid on;




