deviceReader = audioDeviceReader(44100);
deviceWriter = audioDeviceWriter('SampleRate',deviceReader.SampleRate);
fileWriter = dsp.AudioFileWriter('SampleRate',deviceReader.SampleRate);
%% Specify an audio processing algorithm
process = @(x) x.*5;
threshold = 0.01;
set = false;
%% Stream processing
finalT = [];
classes = trainedNet.Layers(end).Classes;
numSegmentsPer4seconds = 7;
while true
    data = [];
    mySignal = deviceReader();
%     myProcessedSignal = process(mySignal);
    myProcessedSignal = mySignal;
    if(mean(abs(myProcessedSignal)) >= threshold)
       disp('Found');
       while size(data)<= 176400
           mySignal = deviceReader();
%            myProcessedSignal = process(mySignal);
           myProcessedSignal = mySignal;
           data = vertcat(data, myProcessedSignal);
           fileWriter(myProcessedSignal);
       end
       set = true;
    end
    release(deviceReader)
    release(deviceWriter)
    release(fileWriter)
   
    %% Prediction
    if(set == true)
        segmentLength  = 1;
        segmentOverlap = 0.5;
        windowLength   = 1024;
        samplesPerHop  = 512;
        fftLength      = 2*windowLength;
        numBands       = 128;
        data = data(1:176400);
        dataFeatures = getSegmentedMelSpectrograms(data,44100, ...
            'SegmentLength',segmentLength, ...
            'SegmentOverlap',segmentOverlap, ...
            'WindowLength',windowLength, ...
            'HopLength',samplesPerHop, ...
            'NumBands',numBands, ...
            'FFTLength',fftLength);
                
        trps = predict(trainedNet,dataFeatures);
        numFiles = 1;

        counter = 1;
        tr = zeros(numFiles,11);
        for channel = 1:numFiles
            tr(channel,:) = sum(trps(counter:counter+numSegmentsPer4seconds-1,:),1)/numSegmentsPer4seconds;
            counter = counter + numSegmentsPer4seconds;
        end
        [~,classIdx] = max(tr,[],2);
        tpl = classes(classIdx);
        if(max(tr,[], 2) > 0.5)
            clear toPrint;
            toPrint(:,2) = tpl;
            toPrint(1,1) = num2str(max(tr,[],2));
            display(toPrint);
            finalT = vertcat(finalT, toPrint);
        else
            fprintf('Too uncertain: %s - %d\n', tpl,max(tr,[], 2));
        end
        set = false;
    end
end