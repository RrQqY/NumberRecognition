% 测试图像预处理效果用代码
close all; clear; clc;

% 加载图片
img = imread('testImgs\test5.bmp');

% 处理图片
[imgVec, imgNoise, imgFilter, imgBw, imgFt, imgCanny, imgRes] = imgPreProcessingWithFilter(img);

% 显示图片
subplot(2,3,1); imshow(imgNoise); title("加入人工噪声");
subplot(2,3,2); imshow(imgFilter); title("低通滤波器降噪");
subplot(2,3,3); imshow(imgBw); title("二值化");
subplot(2,3,4); imshow(imgFt); title("高通滤波器边缘检测");
subplot(2,3,5); imshow(imgCanny); title("Canny边缘检测");
subplot(2,3,6); imshow(imgRes); title("图片预处理结果");