
% Load the data
data = h5read("C:\acconeerData/maammoving01sparseiq.h5", ...
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

% Define step length and range start/end
stepLength = 0.02; % 20 mm in meters
rangeStart = 0.2;  % Starting at 0.2 meters
rangeEnd = 1.0;    % Ending at 1 meter

% Calculate the range values for each range bin (0.2m to 1m)
rangeBins = linspace(rangeStart, rangeEnd, numRangeBins);

% Initialize matrix to store the Doppler spectrum for each range bin
dopplerMatrix = zeros(numRangeBins, numFrames);

% Apply FFT across frames for each range bin
for rangeBin = 1:numRangeBins
    % Extract the complex signal for the current range bin across all frames
    rangeSignal = squeeze(complexData(rangeBin, 1, :)); % Use the first sweep
    % Perform FFT and take the magnitude
    dopplerSpectrum = fftshift(abs(fft(rangeSignal, numFrames)));
    % Store the Doppler spectrum
    dopplerMatrix(rangeBin, :) = dopplerSpectrum;
end

% Generate the Doppler frequency axis
dopplerBins = linspace(-numFrames/2, numFrames/2 - 1, numFrames);

% Plot the Range-Doppler Heatmap with Doppler on X-axis and Range on Y-axis
figure;
imagesc(dopplerBins, rangeBins, dopplerMatrix); % Set Y-axis range from 0.2 to 1 meter
colorbar;
title('Range-Doppler Heatmap');
xlabel('Doppler (Hz)');  % Doppler on X-axis
ylabel('Range (meters)');  % Range on Y-axis (from 0.2 to 1 meter)
axis tight;
colormap('jet'); % Use the 'jet' colormap for better visualization


