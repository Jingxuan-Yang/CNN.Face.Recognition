% Author:  JingXuan Yang
% E-mail:  yangjingxuan@stu.hit.edu.cn
% Date:    2019.05.21
% Project: Artificial Intelligence final project 
% Purpose: face recognition of pictures
% Note   : 

clear;
clc;

% load image data
imds = imageDatastore('./data','IncludeSubfolders',...
                      true, 'FileExtensions',{'.tiff','.jpg'}, ...
                      'LabelSource','foldernames');
 
figure;
[nimage,~] = size(imds.Files);

nshow = 20;
perm = randperm(nimage,nshow);
% random choose 20 images and show them
for i = 1:nshow
    subplot(nshow/5,5,i);
    imshow(imds.Files{perm(i)});
end
 
% distribution of images in each category
labelCount = countEachLabel(imds);
[nlabel,~] = size(labelCount);

% imds.ReadSize = numpartitions(imds)
% transform image size to 56x46 when traning
imds.ReadFcn = @(loc)imresize(imread(loc),[56,46]);

% img = readimage(imds,5); imshow(img); 

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
 
% predicted labels of the imdsValidation
YPred = classify(net,imdsValidation);
% actual labels
YValidation = imdsValidation.Labels;

% calculate prediction accuracy
accuracy = sum(YPred == YValidation)/numel(YValidation);

% test prediction of 'YJX' categrary
testlabel = 'YJX';
i = 1;
figure;
% show images and its predicted label
while i < size(imdsValidation.Labels,1)
    if(imdsValidation.Labels(i) == testlabel) %#ok<BDSCI>
        
        imshow(imdsValidation.Files{i})            
        title(findClass(YPred(i)));           
        pause;
        
    end
    i = i + 1;
end
