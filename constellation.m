% This code code plots the constellation graph , plots the I and Q

% File path of the HDF5 file
h5FilePath = "C:\acconeerData\breath01sparseiq.h5";

% Dataset path within the HDF5 file
datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";

% Load the data
data = h5read(h5FilePath, datasetPath);

% Extract the real (I) and imaginary (Q) components
I = double(data.real); % Real part
Q = double(data.imag); % Imaginary part

% Combine I and Q to form the complex signal
complexSignal = I + 1i * Q;

% Select a specific frame for visualization (e.g., frame 1)
frameNumber = 1; % Specify the frame number
singleFrame = complexSignal(:, :, frameNumber);

% Reshape the frame data into a 1D vector for plotting
I_flat = real(singleFrame(:));
Q_flat = imag(singleFrame(:));

% Plot the constellation graph
figure;
scatter(I_flat, Q_flat, 5, 'filled'); % Scatter plot with points of size 10
grid on;
title(['Constellation Diagram for Frame ', num2str(frameNumber)]);
xlabel('In-Phase (I)');
ylabel('Quadrature (Q)');
axis equal; % Equal scaling for both axes
