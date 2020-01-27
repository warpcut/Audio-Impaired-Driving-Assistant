%% Loading
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

%% Feature extraction
pp = parpool('IdleTimeout',inf);
train_set_tall = tall(train_set);
reset(train_set);
xTrain = cellfun(@(x)extractFeature(x,train_set,441000), train_set_tall,...
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
xTrain = rmmissing(xTrain); %Remove if missing

%% Standardize
xVectors = xTrain{:,2:15};
m = mean(xVectors);
s = std(xVectors);
xTrain{:,2:15} = (xVectors-m)./s;
head(xTrain)

%% Training
inputTable     = xTrain;
predictorNames = xTrain.Properties.VariableNames;
predictors     = inputTable(:, predictorNames(2:15));
response       = inputTable.Label;

tic
trainedClassifier = fitcknn( ...
    predictors, ...
    response, ...
    'Distance','euclidean', ...
    'NumNeighbors',9, ...
    'DistanceWeight','squaredinverse', ...
    'Standardize',false, ...
    'ClassNames',unique(response));
disp('trained');
elapsedTrain = toc;

save('ws_trained_9', 'inputTable', 'response',...
   'trainedClassifier','elapsedTrain');

%% Cross-Validation
tic
k = 10;
group = response;
c = cvpartition(group,'KFold',k); % 10-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',c);
validationAccuracy = 1 - kfoldLoss(partitionedModel, 'LossFun', 'ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
validationPredictions = kfoldPredict(partitionedModel);
elapsedCross = toc;
figure
cmCross = confusionchart(xTrain.Label,validationPredictions,'title','Validation Accuracy');
cmCross.ColumnSummary = 'column-normalized';
cmCross.RowSummary = 'row-normalized';

save('ws_validated_9', 'c', 'partitionedModel', 'validationAccuracy',...
   'validationPredictions', 'elapsedCross', 'cmCross');

%% Testing
test_set_tall = tall(test_set);
reset(test_set);
xTest = cellfun(@(x)extractFeature(x,test_set,441000), train_set_tall,...
    'UniformOutput',false);
xTest = gather(xTest);
featuresLen = length(xTest);
for i = 1:featuresLen
    xTest{i}.Filename = xTest{i}.Filename(:,1);
    xTest{i}.PITCH = xTest{i}.PITCH(:,1);
    xTest{i}.Label = xTest{i}.Label(:,1);
    disp(i);
end
xTest = vertcat(xTest{:});
xTest.MFCC1(xTest.MFCC1 == -inf,:) = nan; %remove mfcc -inf
xTest = rmmissing(xTest); %Remove if missing
xTest{:,2:15} = (xVectors-m)./s;
head(xTest)
result = HelperTestKNNClassifier(trainedClassifier, xTest);
figure;
cmTest = confusionchart(result.ActualSound,result.PredictedSound,'title','Validation Accuracy');
cmTest.ColumnSummary = 'column-normalized';
cmTest.RowSummary = 'row-normalized';
save('ws_tested_9', 'result', 'cmTest');

%% Closing
delete(pp);
