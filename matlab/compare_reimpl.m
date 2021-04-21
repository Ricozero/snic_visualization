close all;
filename = '../example.jpg';
img = imread(filename);
[height, width, colors] = size(img);
tic;
%-------------------------------------------------
numsuperpixels = 200;
compactness = 20.0;
[labels, numlabels] = snic_mex(img,numsuperpixels,compactness);
timetaken = toc;
disp(num2str(timetaken));
%-------------------------------------------------
subplot(1, 2, 1)
imshow(img)
hold on
imcontour(int16(labels), numlabels, 'y')
title('原作者（C++）', 'FontSize', 16)
subplot(1, 2, 2)
imshow(img)
hold on
load('reimpl_labels.mat')
imcontour(int16(reimpl_labels), numlabels, 'y')
title('复现（Python）', 'FontSize', 16)