function X = getSegmentedMelSpectrograms(x,fs,varargin)
% Copyright 2019 The MathWorks, Inc.

    p = inputParser;
    addParameter(p,'WindowLength',1024);
    addParameter(p,'HopLength',512);
    addParameter(p,'NumBands',128);
    addParameter(p,'SegmentLength',1);
    addParameter(p,'SegmentOverlap',0.5);
    addParameter(p,'FFTLength',1024);
    parse(p,varargin{:})
    params = p.Results;
    
    %Param extraction
   
    [~, chn] = size(x);
    if chn == 2
        x = [sum(x,2),x(:,1)-x(:,2)];
        else
        x = [x, x];
    end
    x = x./max(max(x));
    [xb_m,~] = buffer(x(:,1),round(params.SegmentLength*fs),round(params.SegmentOverlap*fs),'nodelay');
    [xb_s,~] = buffer(x(:,2),round(params.SegmentLength*fs),round(params.SegmentOverlap*fs),'nodelay');
    xb = zeros(size(xb_m,1),size(xb_m,2)+size(xb_s,2));
    xb(:,1:2:end) = xb_m;
    xb(:,2:2:end) = xb_s;
    spec = melSpectrogram(xb,fs, ...
        'WindowLength',params.WindowLength, ...
        'OverlapLength',params.WindowLength - params.HopLength, ...
        'FFTLength',params.FFTLength, ...
        'NumBands',params.NumBands, ...
        'FrequencyRange',[0,floor(fs/2)]);
    spec = log10(spec+eps);
    X = reshape(spec,size(spec,1),size(spec,2),size(x,2),[]);
end