folders = fullfile('E:\Documenti\Datasets\Downloaded_sample\Processed\4sec\');
TestADS = audioDatastore(folders, 'FileExtension','.wav', 'LabelSource','foldernames', 'IncludeSubfolders', true);
% pp = parpool('IdleTimeout',inf);

%% Param
segmentLength  = 1;
segmentOverlap = 0.5;
windowLength   = 2048;
samplesPerHop  = 1024;
fftLength      = 2*windowLength;
numBands       = 128;
fs = 44100;
classes = trainedNet.Layers(end).Classes;
numSegmentsPer4seconds = 7;

%% Prediction
predict_tall = tall(TestADS);
zTest = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    predict_tall, ...
    'UniformOutput',false);
zTest = gather(zTest);
zTest = cat(4,zTest{:});

TestResponsesPerSegment = predict(trainedNet,zTest);
numFiles = numel(TestADS.Files);

counter = 1;
TestResponses = zeros(numFiles,numel(classes));
for channel = 1:numFiles
    TestResponses(channel,:) = sum(TestResponsesPerSegment(counter:counter+numSegmentsPer4seconds-1,:),1)/numSegmentsPer4seconds;
    counter = counter + numSegmentsPer4seconds;
end
[~,classIdx] = max(TestResponses,[],2);
TestPredictedLabels = classes(classIdx);
toPrint(:,2) = TestPredictedLabels;
toPrint(:,3) = TestADS.Labels;
for i = 1: length(TestADS.Files)
    [~, toPrint(i,1), ~]= fileparts(char(TestADS.Files(i)));
end
display(toPrint);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 1 sec test
folders = fullfile('E:\Documenti\Datasets\Downloaded_sample\Processed\1sec\');
TestADS1 = audioDatastore(folders, 'FileExtension','.wav', 'LabelSource','foldernames', 'IncludeSubfolders', true);

%% Param
segmentLength  = 0.25;
segmentOverlap = 0.125;
windowLength   = 256;
samplesPerHop  = 128;
samplesOverlap = windowLength - samplesPerHop;
fftLength      = 2*windowLength;
numBands       = 64;
%% Prediction
predict_tall1 = tall(TestADS1);
zTest1 = cellfun(@(x)getSegmentedMelSpectrograms(x,fs, ...
    'SegmentLength',segmentLength, ...
    'SegmentOverlap',segmentOverlap, ...
    'WindowLength',windowLength, ...
    'HopLength',samplesPerHop, ...
    'NumBands',numBands, ...
    'FFTLength',fftLength), ...
    predict_tall1, ...
    'UniformOutput',false);
zTest1 = gather(zTest1);
zTest1 = cat(4,zTest1{:});

TestResponsesPerSegment1 = predict(trainedNet1sec,zTest1);
numFiles = numel(TestADS1.Files);

counter = 1;
TestResponses1 = zeros(numFiles,numel(classes));
for channel = 1:numFiles
    TestResponses1(channel,:) = sum(TestResponsesPerSegment1(counter:counter+numSegmentsPer1seconds-1,:),1)/numSegmentsPer1seconds;
    counter = counter + numSegmentsPer1seconds;
end
[~,classIdx1] = max(TestResponses1,[],2);
TestPredictedLabels1 = classes(classIdx1);
clear toPrint;
toPrint(:,2) = TestPredictedLabels1;
toPrint(:,3) = TestADS1.Labels;
for i = 1: length(TestADS1.Files)
    [~, toPrint(i,1), ~]= fileparts(char(TestADS1.Files(i)));
end
display(toPrint);


delete(pp);