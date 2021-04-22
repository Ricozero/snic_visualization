close all;
subplot(1, 2, 1)
imshow(img)
hold on
load('labels.mat')
imcontour(int16(labels), numlabels, 'c')
title('Original（C++）', 'FontSize', 16)
subplot(1, 2, 2)
imshow(img)
hold on
load('reimpl_labels.mat')
imcontour(int16(reimpl_labels), numlabels, 'y')
title('Reimpl.（Python）', 'FontSize', 16)