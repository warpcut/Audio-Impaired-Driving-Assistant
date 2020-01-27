function result = HelperTestKNNClassifier(trainedClassifier, featuresTest)

[Fidx,Filenames] = findgroups(featuresTest.Filename);
result = table('Size',[0 4], 'VariableTypes', {'string', 'string','string', 'double'}, 'VariableNames', {'Filename','ActualSound','PredictedSound', 'ConfidencePercentage'});
for idx = 1:length(Filenames)
    disp(idx);
    T = featuresTest(Fidx==idx,2:end);  % Rows that correspond to one file
    predictedLabels = string(predict(trainedClassifier,T(:,1:15))); % Predict
    totalVals = size(predictedLabels,1);

    [predictedLabel, freq] = mode(categorical(predictedLabels)); % Find most frequently predicted label
    match = freq/totalVals*100;
    result_file.Filename = Filenames(idx);
    result_file.ActualSound = T.Label{1};
    result_file.PredictedSound = char(predictedLabel);
    result_file.ConfidencePercentage = match;

    result = [result; struct2table(result_file)]; %#ok
end
end
