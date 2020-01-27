folders = fullfile('E:\Documenti\Datasets\UrbanSound8K\audio');
FullADS = audioDatastore(folders,'LabelSource','foldernames', 'FileExtension','.wav','IncludeSubfolders',true);
lenDataTrain = length(FullADS.Files);
counter = 0;
reset(FullADS);
for i = 1:lenDataTrain 
    [aud, Fs] = read(FullADS);
    N = length(aud);
    slength = N/Fs.SampleRate;
    [~,name,ext] = fileparts(char(FullADS.Files(i)));
    filename = strcat('E:\Documenti\Datasets\8Kprocessed\resample441000\',name,ext); 
    if Fs.SampleRate ~= 44100
        [p,q] = rat(44100/Fs.SampleRate);
        aud = resample(aud, p, q);
    end
    audiowrite(filename,aud,44100);
    counter = counter +1;
end