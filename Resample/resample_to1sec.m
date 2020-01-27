folders = fullfile('E:\Documenti\Datasets\UrbanSound8K\audio');
DwnADS = audioDatastore(folders, 'FileExtension','.wav','IncludeSubfolders',true);
lenDataTrain = length(DwnADS.Files);
counter = 0;
counterS = 0;
reset(DwnADS);
for i = 1:lenDataTrain %extract features for every element
    [aud, Fs] = read(DwnADS);
    N = length(aud);
    slength = N/Fs.SampleRate;
    
    if slength < 1
        counter = counter +1;
        aud = vertcat(aud, aud);
        N = length(aud);
        slength = N/Fs.SampleRate;
        if slength < 1
            counterS = counterS +1;
            [~, chn] = size(aud);
            aud = vertcat(aud, zeros(Fs.SampleRate - N, chn));
            N = length(aud);
            slength = N/Fs.SampleRate;
        end
    end
    [~,name,ext] = fileparts(char(DwnADS.Files(i)));
    filename = strcat('E:\Documenti\Datasets\8Kprocessed\to1sec\',name,ext); 
    if Fs.SampleRate ~= 44100
        [p,q] = rat(44100/Fs.SampleRate);
        aud = resample(aud, p, q);
    end
    audiowrite(filename,aud(1:44100, :),44100);
    disp(i);
end