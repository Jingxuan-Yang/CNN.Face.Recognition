% Author:  JingXuan Yang
% E-mail:  yangjingxuan@stu.hit.edu.cn
% Date:    2019.05.21
% Project: Artificial Intelligence final project 
% Purpose: face recognition of camera video
% Note   : !! require webcam support package

clc,clear,close all;

% train convolutional neural network
videoNet;

% create face detector object
faceDetector = vision.CascadeObjectDetector();     

% use point tracker to track the video frame 
pointTracker = vision.PointTracker('MaxBidirectionalError', 2);
 
% create camera object
cam = webcam();
 
% obtain current fram and its size
videoFrame = snapshot(cam);
frameSize = size(videoFrame); 
 
% creat video object
videoPlayer = vision.VideoPlayer('Position', ...
                                [100 100 [frameSize(2), frameSize(1)]+30]);        

% text position, messured from left top
textx = 640;
texty = 50;

% default start loop
runLoop = true;
% initial characteristic point is set to be zero
numPts = 0;
% initialize the number of frame
frameCount = 0;     
 
% begin loop
while runLoop && frameCount < 1000
    
    % obtain current frame
    videoFrame = snapshot(cam);
    % turn frame into gray style
    videoFrameGray = rgb2gray(videoFrame);
    % trun gray frame into size [56 46], for recognition
    videoFrameGrayResize = imresize(videoFrameGray, [56, 46]);
    % count number of frame
    frameCount = frameCount + 1;    
    
    if numPts < 10
        % detect method: bbox, 1*4 vector [x y width height]
        % x and y are measured from left top
        bbox = faceDetector.step(videoFrameGray);    
        
        if ~isempty(bbox)
            % detect eigen points on face
            points = detectMinEigenFeatures(videoFrameGray, 'ROI', ...
                                            bbox(1, :));       
            
            % position of the points
            xyPoints = points.Location;
            % number of points
            numPts = size(xyPoints,1);
            
            release(pointTracker);
            initialize(pointTracker, xyPoints, videoFrameGray);
            
            % store xypoints
            oldPoints = xyPoints;      
            
            % transform box to [x1 y1 x2 y2 x3 y3 x4 y4] 
            bboxPoints = bbox2points(bbox(1, :));    
            
            % adjust row and column
            bboxPolygon = reshape(bboxPoints', 1, []);    
            
            % classify frame using CNN
            YPred = classify(net,videoFrameGrayResize);
            % insert classified result
            position = [textx texty];
            videoFrame = insertText(videoFrame,position, ...
                                    char(findClass(YPred)),...
                                    'FontSize',40,'BoxColor',...
                                    'red','BoxOpacity',0.4, ...
                                    'TextColor','white');           
            
            % insert box
            videoFrame = insertShape(videoFrame, 'Polygon', ...
                                     bboxPolygon, 'LineWidth', 3);
            
            % insert point marker
            videoFrame = insertMarker(videoFrame, xyPoints, ...
                                      '+', 'Color', 'white');
            
        end
        
    else
        % detect point tracker
        [xyPoints, isFound] = step(pointTracker, videoFrameGray);
        visiblePoints = xyPoints(isFound, :);
        oldInliers = oldPoints(isFound, :);
                
        numPts = size(visiblePoints, 1);       
        
        if numPts >= 10
            % geometric transform
            [xform, oldInliers, visiblePoints] = ...
                estimateGeometricTransform(oldInliers, ...
                                           visiblePoints, ...
                                           'similarity', ...
                                           'MaxDistance', 4);            
            
            % transform boundary points
            bboxPoints = transformPointsForward(xform, bboxPoints);
            
            %reshape row and column
            bboxPolygon = reshape(bboxPoints', 1, []); 
            
            % classify frame using CNN
            YPred = classify(net,videoFrameGrayResize);
            % insert classified result
            position = [textx texty];
            videoFrame = insertText(videoFrame,position, ...
                                    char(findClass(YPred)),...
                                    'FontSize',40,'BoxColor',...
                                    'red','BoxOpacity',0.4, ...
                                    'TextColor','white');           
            
            % insert box
            videoFrame = insertShape(videoFrame, 'Polygon', ...
                                     bboxPolygon, 'LineWidth', 3);
            
            % insert point marker
            videoFrame = insertMarker(videoFrame, xyPoints, ...
                                      '+', 'Color', 'white');
                                    
            % reset points
            oldPoints = visiblePoints;
            % for redetection
            setPoints(pointTracker, oldPoints);
        end
 
    end
    
    % play video
    step(videoPlayer, videoFrame);
    % watch whether the video is closed
    runLoop = isOpen(videoPlayer);
end
              
% clear and reset
clear cam;
release(videoPlayer);
release(pointTracker);
release(faceDetector);
