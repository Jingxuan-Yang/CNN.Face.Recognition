% subprogram for videoR
% perform CNN training to face data

% load image data
imds = imageDatastore('./data','IncludeSubfolders',...
                      true, 'FileExtensions',{'.tiff','.jpg'}, ...
                      'LabelSource','foldernames');
 
% distribution of images in each category
labelCount = countEachLabel(imds);
[nlabel,~] = size(labelCount);

% transform image size to 56x46 when traning
imds.ReadFcn = @(loc)imresize(imread(loc),[56,46]);

numTrainFiles = 10;
% random split database into two new database
[imdsTrain,imdsValidation] = splitEachLabel(imds, ...
                                          numTrainFiles,'randomize');

% define net, input image size 56x46
layers = [imageInputLayer([56 46 1])

    % convoluntional layer, 5 conv kernel, operation size 6x6
    convolution2dLayer(6,5,'Padding','same')
    batchNormalizationLayer
    reluLayer

    % pooling layer
    maxPooling2dLayer(2,'Stride',2)

    % convoluntional layer, 14 conv kernel, operation size 6x6
    convolution2dLayer(6,14,'Padding','same')
    batchNormalizationLayer
    reluLayer

    % pooling layer 
    maxPooling2dLayer([4,3],'Stride',2)

    % fully connected layer
    fullyConnectedLayer(60)     
    reluLayer

    % fully connected layer, 12 category
    fullyConnectedLayer(nlabel) 
    softmaxLayer
    classificationLayer  ];

% training parameters
options = trainingOptions('sgdm', ...
                          'InitialLearnRate',0.1, ...
                          'MaxEpochs',100, ...
                          'Shuffle','every-epoch', ...
                          'ValidationData',imdsValidation, ...
                          'ValidationFrequency',30, ...
                          'Verbose',false, ...
                          'Plots','training-progress');

% begin training
net = trainNetwork(imdsTrain,layers,options);

