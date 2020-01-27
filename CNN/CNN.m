%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 4 second model
Table = readtable('E:\Documenti\Datasets\UrbanSound8K\metadata\UrbanSound8K.csv');
folders = fullfile('E:\Documenti\Datasets\8Kprocessed\to4sec\');
hornfolders = fullfile('E:\Documenti\Datasets\Downloaded_sample\car_horn\4sec\');
hornADS = audioDatastore(hornfolders,'LabelSource','foldernames', 'FileExtension','.wav','IncludeSubfolders',true);
FullADS = audioDatastore(folders,'LabelSource','foldernames', 'FileExtension','.wav','IncludeSubfolders',true);

%% Labeling
Tnew = table(Table.slice_file_name, Table.class, 'RowNames',Table.slice_file_name);
adslen = length(FullADS.Files);
for i = 1:adslen
    [~,name,ext] = fileparts(char(FullADS.Files(i)));
    FullADS.Labels(i) = Tnew({strcat(name,ext)},:).Var2;
end
countEachLabel(FullADS)
exlen = length(hornADS.Files);
ExtendedADS = FullADS;
for i = adslen+1:adslen+exlen
   ExtendedADS.Files(i) = hornADS.Files(i-adslen);
   ExtendedADS.Labels(i) = hornADS.Labels(i-adslen);
end
countEachLabel(ExtendedADS)
[train_set, test_set] = splitEachLabel(ExtendedADS,0.80,'randomized'); %Divide dataset in two randomly, 80perc used to train
display(countEachLabel(train_set))
display(countEachLabel(test_set))
%% Param extraction
[data,info] = read(train_set);
data = data./max(data,[],'all');
fs = info.SampleRate;
sound(data,fs)
reset(train_set)
[~, chn] = size(data);
if chn ~= 1
    dataMidSide = [sum(data,2),data(:,1)-data(:,2)];
else
    dataMidSide = [data, data];
end
segmentLength  = 1;
segmentOverlap = 0.5;
[dataBufferedMid,~] = buffer(dataMidSide(:,1),round(segmentLength*fs),round(segmentOverlap*fs),'nodelay');
[dataBufferedSide,~] = buffer(dataMidSide(:,2),round(segmentLength*fs),round(segmentOverlap*fs),'nodelay');
dataBuffered = zeros(size(dataBufferedMid,1),size(dataBufferedMid,2)+size(dataBufferedSide,2));
dataBuffered(:,1:2:end) = dataBufferedMid;
dataBuffered(:,2:2:end) = dataBufferedSide;

windowLength   = 1024;
samplesPerHop  = 512;
samplesOverlap = windowLength - samplesPerHop;
fftLength      = 2*windowLength;
numBands       = 128;
spec = melSpectrogram(dataBuffered,fs, ...
    'WindowLength',windowLength, ...
    'OverlapLength',samplesOverlap, ...
    'FFTLength',fftLength, ...
    'NumBands',numBands);
spec = log10(spec+eps);
X = reshape(spec,size(spec,1),size(spec,2),size(data,2),[]);

for channel = 1:2:11
    figure
    melSpectrogram(dataBuffered(:,channel),fs, ...
        'WindowLength',windowLength, ...
        'OverlapLength',samplesOverlap, ...
        'FFTLength',fftLength, ...
        'NumBands',numBands);
    title(sprintf('Segment %d',ceil(channel/2)))
end
%start paralleling
pp = parpool('IdleTimeout',inf);
%% Feature extraction
train_set_tall = tall(train_set);
xTrain = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    train_set_tall, ...
    'UniformOutput',false);
xTrain = gather(xTrain);
xTrain = cat(4,xTrain{:});

test_set_tall = tall(test_set);
xTest = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    test_set_tall, ...
    'UniformOutput',false);
xTest = gather(xTest);
xTest = cat(4,xTest{:});
numSegmentsPer4seconds = size(dataBuffered,2)/2;
yTrain = repmat(train_set.Labels,1,numSegmentsPer4seconds)';
yTrain = yTrain(:);

%% Define layers
imgSize = [size(xTrain,1),size(xTrain,2),size(xTrain,3)];
numF = 32;
layers = [ ...
    imageInputLayer(imgSize)

    batchNormalizationLayer

    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,8*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,8*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(ceil(imgSize(1:2)/8))

    dropoutLayer(0.5)

    fullyConnectedLayer(11)
    softmaxLayer
    classificationLayer];

%% Define model
miniBatchSize = 128;
tuneme = 256;
lr = 0.05*miniBatchSize/tuneme;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',lr, ...
    'MiniBatchSize',miniBatchSize, ...
    'Momentum',0.9, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',2, ...
    'LearnRateDropFactor',0.2);

%% Training
trainedNet = trainNetwork(xTrain,yTrain,layers,options);

%% Evaluation
cnnResponsesPerSegment = predict(trainedNet,xTest);
classes = trainedNet.Layers(end).Classes;
numFiles = numel(test_set.Files);

counter = 1;
cnnResponses = zeros(numFiles,numel(classes));
for channel = 1:numFiles
    cnnResponses(channel,:) = sum(cnnResponsesPerSegment(counter:counter+numSegmentsPer4seconds-1,:),1)/numSegmentsPer4seconds;
    counter = counter + numSegmentsPer4seconds;
end
[~,classIdx] = max(cnnResponses,[],2);
cnnPredictedLabels = classes(classIdx);
figure
cm = confusionchart(test_set.Labels,cnnPredictedLabels,'title','Test Accuracy - CNN');
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';

fprintf('Average accuracy of 4s CNN = %0.2f\n',mean(test_set.Labels==cnnPredictedLabels)*100)
save('4sec-trained-model-ext', 'cm', 'cnnPredictedLabels', 'trainedNet', 'cnnResponses', 'options', 'layers');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% One second model
clear;
Table = readtable('E:\Documenti\Datasets\UrbanSound8K\metadata\UrbanSound8K.csv');
folders = fullfile('E:\Documenti\Datasets\8Kprocessed\to1sec\');
hornfolders = fullfile('E:\Documenti\Datasets\Downloaded_sample\car_horn\1sec\');
hornADS = audioDatastore(hornfolders,'LabelSource','foldernames', 'FileExtension','.wav','IncludeSubfolders',true);
FullADS = audioDatastore(folders,'LabelSource','foldernames', 'FileExtension','.wav','IncludeSubfolders',true);

%%Labeling
Tnew = table(Table.slice_file_name, Table.class, 'RowNames',Table.slice_file_name);
adslen = length(FullADS.Files);
for i = 1:adslen
    [~,name,ext] = fileparts(char(FullADS.Files(i)));
    FullADS.Labels(i) = Tnew({strcat(name,ext)},:).Var2;
end
countEachLabel(FullADS)
exlen = length(hornADS.Files);
ExtendedADS = FullADS;
for i = adslen+1:adslen+exlen
    disp(i);
   ExtendedADS.Files(i) = hornADS.Files(i-adslen);
   ExtendedADS.Labels(i) = hornADS.Labels(i-adslen);
end
countEachLabel(ExtendedADS)
%% Split data
[train_set, test_set] = splitEachLabel(ExtendedADS,0.80,'randomized'); %Divide dataset in two randomly, 80perc used to train
display(countEachLabel(train_set))
display(countEachLabel(test_set))
%% Param extraction
[data,info] = read(train_set);
data = data./max(data,[],'all');
fs = info.SampleRate;
sound(data,fs)
reset(train_set)
[~, chn] = size(data);
if chn ~= 1
    dataMidSide = [sum(data,2),data(:,1)-data(:,2)];
else
    dataMidSide = [data, data];
end
segmentLength  = 0.25;
segmentOverlap = 0.125;
[dataBufferedMid,~] = buffer(dataMidSide(:,1),round(segmentLength*fs),round(segmentOverlap*fs),'nodelay');
[dataBufferedSide,~] = buffer(dataMidSide(:,2),round(segmentLength*fs),round(segmentOverlap*fs),'nodelay');
dataBuffered = zeros(size(dataBufferedMid,1),size(dataBufferedMid,2)+size(dataBufferedSide,2));
dataBuffered(:,1:2:end) = dataBufferedMid;
dataBuffered(:,2:2:end) = dataBufferedSide;

windowLength   = 256;
samplesPerHop  = 128;
samplesOverlap = windowLength - samplesPerHop;
fftLength      = 2*windowLength;
numBands       = 64;
spec = melSpectrogram(dataBuffered,fs, ...
    'WindowLength',windowLength, ...
    'OverlapLength',samplesOverlap, ...
    'FFTLength',fftLength, ...
    'NumBands',numBands);
spec = log10(spec+eps);
X = reshape(spec,size(spec,1),size(spec,2),size(data,2),[]);

for channel = 1:2:11
    figure
    melSpectrogram(dataBuffered(:,channel),fs, ...
        'WindowLength',windowLength, ...
        'OverlapLength',samplesOverlap, ...
        'FFTLength',fftLength, ...
        'NumBands',numBands);
    title(sprintf('Segment %d',ceil(channel/2)))
end
%% Feature extraction
train_set_tall = tall(train_set);
xTrain = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    train_set_tall, ...
    'UniformOutput',false);
xTrain = gather(xTrain);
xTrain = cat(4,xTrain{:});

test_set_tall = tall(test_set);
xTest = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    test_set_tall, ...
    'UniformOutput',false);
xTest = gather(xTest);
xTest = cat(4,xTest{:});
numSegmentsPer1seconds = size(dataBuffered,2)/2;
yTrain = repmat(train_set.Labels,1,numSegmentsPer1seconds)';
yTrain = yTrain(:);

%% Define layers
imgSize = [size(xTrain,1),size(xTrain,2),size(xTrain,3)];
numF = 32;
layers = [ ...
    imageInputLayer(imgSize)

    batchNormalizationLayer

    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(3,'Stride',2,'Padding','same')

    convolution2dLayer(3,8*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    convolution2dLayer(3,8*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer

    averagePooling2dLayer(ceil(imgSize(1:2)/8))

    dropoutLayer(0.5)

    fullyConnectedLayer(12)
    softmaxLayer
    classificationLayer];
%% Define model
miniBatchSize = 128;
tuneme = 256;
lr = 0.05*miniBatchSize/tuneme;
options = trainingOptions('sgdm', ...
    'InitialLearnRate',lr, ...
    'MiniBatchSize',miniBatchSize, ...
    'Momentum',0.9, ...
    'L2Regularization',0.005, ...
    'MaxEpochs',10, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',2, ...
    'LearnRateDropFactor',0.2);
%% Training
trainedNet1sec = trainNetwork(xTrain,yTrain,layers,options);

%% Evaluation
cnnResponsesPerSegment = predict(trainedNet1sec,xTest);
classes = trainedNet1sec.Layers(end).Classes;
numFiles = numel(test_set.Files);

counter = 1;
cnnResponses1 = zeros(numFiles,numel(classes));
for channel = 1:numFiles
    cnnResponses1(channel,:) = sum(cnnResponsesPerSegment(counter:counter+numSegmentsPer1seconds-1,:),1)/numSegmentsPer1seconds;
    counter = counter + numSegmentsPer1seconds;
end
[~,classIdx] = max(cnnResponses1,[],2);
cnnPredictedLabels1 = classes(classIdx);
figure
cm1 = confusionchart(test_set.Labels,cnnPredictedLabels1,'title','Test Accuracy - CNN 1 sec');
cm1.ColumnSummary = 'column-normalized';
cm1.RowSummary = 'row-normalized';

fprintf('Average accuracy of 1s CNN = %0.2f\n',mean(test_set.Labels==cnnPredictedLabels1)*100)
save('1sec-trained-model-ext', 'cm1', 'cnnPredictedLabels1', 'trainedNet1sec', 'cnnResponses1');

%% Stop Paralleling
delete(pp)