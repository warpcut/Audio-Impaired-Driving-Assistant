%% Build struct and params
function X = extractFeature(data,audio,fs)
    
    % Compute pitch and MFCC for frames of the file
    [pitch1, mfcc1] = computeDeltaMFCC(data,fs);
    [~, infoTrain] = read(audio);
    [~,name,~] = fileparts(char(infoTrain.FileName));
    filenamesplit = regexp(name, filesep, 'split');

    % Output structure
    s = struct();
    s.Filename = repmat({filenamesplit{end}},size(pitch1));
    s.PITCH = pitch1;
    s.MFCC1 = mfcc1(:,1);
    s.MFCC2 = mfcc1(:,2);
    s.MFCC3 = mfcc1(:,3);
    s.MFCC4 = mfcc1(:,4);
    s.MFCC5 = mfcc1(:,5);
    s.MFCC6 = mfcc1(:,6);
    s.MFCC7 = mfcc1(:,7);
    s.MFCC8 = mfcc1(:,8);
    s.MFCC9 = mfcc1(:,9);
    s.MFCC10 = mfcc1(:,10);
    s.MFCC11 = mfcc1(:,11);
    s.MFCC12 = mfcc1(:,12);
    s.MFCC13 = mfcc1(:,13);

    s.Label = repmat({char(infoTrain.Label)},size(pitch1));

    X = struct2table(s);
end

%% Compute function
function [pitch1, mfcc1] = computeDeltaMFCC(x,fs)

    % Audio data will be divided into frames of 30 ms with 75% overlap
    frameTime = 30e-3;
    samplesPerFrame = floor(frameTime*fs);
    increment = floor(0.25*samplesPerFrame);
    overlapLength = samplesPerFrame - increment;
    [pitch1,~] = pitch(x,fs, ...
        'WindowLength',samplesPerFrame, ...
        'OverlapLength',overlapLength);
    [mfcc1] = mfcc(x,fs,'WindowLength',samplesPerFrame, ...
        'OverlapLength',overlapLength, 'LogEnergy', 'Replace');
end