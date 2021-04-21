%======================================================================
% Simple Non-Iterative Clustering (SNIC) demo
% Copyright (C) 2017 Ecole Polytechnique Federale de Lausanne
% File created by Radhakrishna Achanta (firstname.lastname@epfl.ch)
%
% If you use the code for your research, please cite:
%
% "Superpixels and Polygons using Simple Non-Iterative Clustering"
% Radhakrishna Achanta and Sabine Susstrunk
% CVPR 2017
%======================================================================
%Input parameters for snic_mex function are:
%
%[1] 8 bit images (color or grayscale)
%[2] The number of superpixels
% [3] Compactness factor [10, 40]
%
%Ouputs are:
% [1] The labels of the segmented image
% [2] The number of labels
%
%NOTES:
%[2] Before using this demo code you must compile the C++ file usingthe command:
% mex snic_mex.cpp
% ==========================================

close all;
filename = '../example.jpg';
img = imread(filename);
[height, width, colors] = size(img);
tic;
%-------------------------------------------------
numsuperpixels = 200;
compactness = 20.0;
[labels, numlabels] = snic_mex(img,numsuperpixels,compactness);
%-------------------------------------------------
timetaken = toc;
disp(num2str(timetaken));
vis = img;
for i = 1:numlabels
    [row, col] = find(labels == i);
    color_sum = [0; 0; 0];
    n_count = 0;
    for j = 1:size(row, 1)
        color_sum = color_sum + double(squeeze(img(row(j), col(j), :)));
        n_count = n_count + 1;
    end
    avg = color_sum / n_count;
    for j = 1:size(row, 1)
        vis(row(j), col(j), :) = avg;
    end
end
imshow(vis)
figure
imshow(img)
hold on
imcontour(int16(labels), numlabels, 'y')