% File path of the HDF5 file
h5FilePath = "C:\acconeerData\breath01sparseiq.h5";

% Dataset path within the HDF5 file
datasetPath = "/sessions/session_0/group_0/entry_0/result/frame";

% Load the data
data = h5read(h5FilePath, datasetPath);

% Extract the real (I) and imaginary (Q) components
I = data.real; % Real part
Q = data.imag; % Imaginary part

% Get the number of frames and reshape data
numFrames = size(I, 3);
numPoints = size(I, 1) * size(I, 2); % Number of data points per frame

% Flatten I and Q into 2D matrices
I_flat = reshape(I, [], numFrames); % Rows = Points per frame, Columns = Frames
Q_flat = reshape(Q, [], numFrames);

% Create frame numbers
frameNumbers = repelem(1:numFrames, numPoints)'; % Repeat frame numbers for each point

% Combine Frame Numbers, I, and Q into a single table
combinedData = [frameNumbers, reshape(I_flat, [], 1), reshape(Q_flat, [], 1)];

% Define headers
headers = ["Frame_Number", "I_Value", "Q_Value"];

% Convert data to table for saving with headers
T = array2table(combinedData, 'VariableNames', headers);

% Specify the desired file path for saving the data
csvFilePath = "C:\Users\GOPAL\Documents\IQ_breath01sparseiq.csv";

% Save the table to CSV
writetable(T, csvFilePath);

disp(['Data successfully saved to: ', csvFilePath]);
