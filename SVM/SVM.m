%% Loading
Table = readtable('E:\Documenti\Datasets\UrbanSound8K\metadata\UrbanSound8K.csv');
folders = fullfile('E:\Documenti\Datasets\8Kprocessed\to4sec');
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
%Split data
[audioTrain, audioTest] = splitEachLabel(ExtendedADS,0.80,'randomized'); %Divide dataset in two randomly, 80perc used to train
display(countEachLabel(audioTrain))
display(countEachLabel(audioTest))
%% Feature extraction
pp = parpool('IdleTimeout',inf);
train_set_tall = tall(audioTrain);
reset(audioTrain);
xTrain = cellfun(@(x)extractFeature(x,audioTrain,441000), train_set_tall,...
    'UniformOutput',false);
xTrain = gather(xTrain);
featuresLen = length(xTrain);
for i = 1:featuresLen
    xTrain{i}.Filename = xTrain{i}.Filename(:,1);
    xTrain{i}.PITCH = xTrain{i}.PITCH(:,1);
    xTrain{i}.Label = xTrain{i}.Label(:,1);
    disp(i);
end
xTrain = vertcat(xTrain{:});
xTrain.MFCC1(xTrain.MFCC1 == -inf,:) = nan; %remove mfcc -inf
xTrain.MFCC1(xTrain.DELTA1 == inf,:) = nan; %remove mfcc inf
xTrain = rmmissing(xTrain); %Remove if missing

%% Standardize
xVectors = xTrain{:,2:28};
m = mean(xVectors);
s = std(xVectors);
xTrain{:,2:28} = (xVectors-m)./s;
head(xTrain)

%% Training
inputTable     = xTrain;
predictorNames = xTrain.Properties.VariableNames;
predictors     = inputTable(:, predictorNames(2:28));
response       = inputTable.Label;
tic
MultiSVM = fitcecoc(xTrain{:,2:28},response,'Verbose',2,...
    'ClassNames',unique(response), 'Coding', 'onevsall','Options',statset('UseParallel',true));
elapsedTrain = toc;

MultiSVM.ClassNames; 
error = resubLoss(MultiSVM);
save('ws_trained', 'xTrain', 'inputTable', 'response',...
   'MultiSVM', 'error', 'm', 's', 'elapsedTrain');

%% Cross-Validation
tic
k = 10;
group = response;
c = cvpartition(group,'KFold',k); % 10-fold stratified cross validation
partitionedModel = crossval(MultiSVM,'CVPartition',c, 'Options',statset('UseParallel',true));
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
validationPredictions = kfoldPredict(partitionedModel, 'Verbose',1);
elapsedCross = toc;
figure
cmCross = confusionchart(response,validationPredictions,'title','Validation Accuracy');
cmCross.ColumnSummary = 'column-normalized';
cmCross.RowSummary = 'row-normalized';
save('ws_validation', 'c', 'partitionedModel', 'validationAccuracy',...
   'validationPredictions', 'elapsedCross', 'cmCross');

%% Closing
delete(pp);
