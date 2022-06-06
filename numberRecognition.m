%%--------2022信号分析大作业：基于神经网络的手写体数字识别器--------%%
close all; clear; clc;
disp("┌────────────────────────────────────┐");
disp("│                          -手写体数字识别器-                         │")

%--------加载图像并进行预处理--------%
disp("│ LOADING IMAGES......                                               │");
filename =dir('trainImgs\*.bmp');

% 循环读入1-100个样本数字文件并进行预处理
imgVec = zeros(100,256);        % 存储数据集图像信息，每行为一张16*16的图片
for k = 1:100
    m = strcat('trainImgs\',filename(k).name);
    x = imread(m,'bmp');        % 循环读入0-99个样本数字文件 
    imgVec(k,:) = imgPreProcessingWithFilter(x);    % 预处理图像
end

%--------初始化数据集标签--------%
% [0,0,0,0,0,0,0,0,0,0]
labelVec = zeros(100,10);       % 存储数据集标签信息，每行为10长度的向量 
for i=0:9
    for j=1:10
        labelVec(i*10+j,i+1) = 1;
    end
end

%--------划分训练集和测试集--------%
trainImgVec = zeros(80,256);
testImgVec = zeros(20,256);
trainLabelVec = zeros(80,10);
testLabelVec = zeros(20,10);
k = 1;
for i = 1:10
    trainImgVec((i-1)*8+1:i*8, :) = imgVec((i-1)*10+1:i*10-2,:);
    testImgVec((i-1)*2+1:i*2, :) = imgVec(i*10-1:i*10,:);
    trainLabelVec((i-1)*8+1:i*8, :) = labelVec((i-1)*10+1:i*10-2,:);
    testLabelVec((i-1)*2+1:i*2, :) = labelVec(i*10-1:i*10,:);
end

disp("│ LOADING COMPLETED!                                                 │");

%--------通过神经网络训练数据集--------%
disp("│ TRAINING......                                                     │");

netSize = [256,30,10];            % 定义网络大小：输入层16*16，隐层30，输出层10
step = 3;                         % 学习率
epoches = 1000;                    % 迭代次数

% 训练集变量初始化
% 初始化biases和weights
% b1 = randn(netSize(2),1);           % 输入层和隐层间的偏差值
% b2 = randn(netSize(3),1);           % 隐层和输出层间的偏差值
% w1 = randn(netSize(2),netSize(1));  % 输入层和隐层间的权值
% w2 = randn(netSize(3),netSize(2));  % 隐层和输出层间的权值
load("parameters\w1_30.mat");        % 输入层和隐层间的偏差值
load("parameters\w2_30.mat");        % 隐层和输出层间的偏差值
load("parameters\b1_30.mat");        % 输入层和隐层间的权值
load("parameters\b2_30.mat");        % 隐层和输出层间的权值
loss = ones(epoches,80);             % 损失函数

% 迭代3000次
for j = 1:epoches
    % 随机打乱数据集
    r = randperm(80);
    trainImgVec0 = trainImgVec(r,:);
    trainLabelVec0 = trainLabelVec(r,:);
    % 对每一组数据进行处理
    for i = 1:80
        % 为各参数分配空间
        db1 = zeros(netSize(2),1);          % 输入层和隐层间的偏差值的变化量
        db2 = zeros(netSize(3),1);          % 隐层和输出层间的偏差值的变化量
        dw1 = zeros(netSize(2),netSize(1)); % 输入层和隐层间的权值的变化量
        dw2 = zeros(netSize(3),netSize(2)); % 隐层和输出层间的权值的变化量
        x1 = zeros(netSize(1),1);           % 隐层输入
        y1 = zeros(netSize(2),1);           % 隐层直接输出
        a1 = zeros(netSize(2),1);           % 隐层经过激发函数后输出
        x2 = zeros(netSize(2),1);           % 输出层输入（就是a1）
        y2 = zeros(netSize(3),1);           % 输出层直接输出
        a2 = zeros(netSize(3),1);           % 输出层经过激发函数后输出（最终输出）
        a_true = zeros(netSize(3),1);       % 输出标签的真值
        g = zeros(netSize(3),1);
        e = zeros(netSize(2),1);

        % 正向传播算法获得输出和损失值
        a_true = trainLabelVec0(i,:)';
        x1 = trainImgVec0(i,:)';
        y1 = w1 * x1 + b1;
        a1 = sigmoid(y1);
        x2 = a1;
        y2 = w2 * x2 + b2;
        a2 = sigmoid(y2);
        loss(j,i) = 1/2 * abs((a_true - a2)' * (a_true - a2));
        %disp(loss)

        % 反向传播算法获得参数梯度
        % 计算输出层神经元的梯度项g
        g = a2.*(1-a2).*(a_true-a2);
        % 计算隐层神经元的梯度项e
        e = a1.*(1-a1).*((w2.')*g);
        % 获得各个参数的变化值
        dw2 = dw2 + g * a1.';
        db2 = db2 - g;
        dw1 = dw1 + e * x1.';
        db1 = db1 - e;

        % 更新参数值
        w1 = w1 + step*dw1;
        w2 = w2 + step*dw2;
        b1 = b1 + step*db1;
        b2 = b2 + step*db2;
    end
    % 每次迭代后运用测试数据集来验证准确率，输出测试集正确输出的数目
    sum = 0;
    for i = 1:20
        % 为各参数分配空间
        x1_test = zeros(netSize(1),1);          % 隐层输入
        y1_test = zeros(netSize(2),1);          % 隐层直接输出
        a1_test = zeros(netSize(2),1);          % 隐层经过激发函数后输出
        x2_test = zeros(netSize(2),1);          % 输出层输入（就是a1）
        y2_test = zeros(netSize(3),1);          % 输出层直接输出
        a2_test = zeros(netSize(3),1);          % 输出层经过激发函数后输出（最终输出）
        a_true_test = zeros(netSize(3),1);      % 输出标签的真值

        % 获得一个测试集样本输入的神经网络输出
        a_true_test = testLabelVec(i,:)';
        x1_test = testImgVec(i,:)';
        y1_test = w1 * x1_test + b1;
        a1_test = sigmoid(y1_test);
        x2_test = a1_test;
        y2_test = w2 * x2_test + b2;
        a2_test = sigmoid(y2_test);
        % 对于神经网络的输出，取每一列的最大值为1，其他为0
        maxi = -1;
        max = 0;
        for k = 1:10
            if a2_test(k) > max
                maxi = k;
                max = a2_test(k);
            end
        end
        for k = 1:10
            if k == maxi
                a2_test(k) = 1;
            else
                a2_test(k) = 0;
            end
        end
        if a2_test(:).' == testLabelVec(i,:)
            sum = sum+1;
        end
    end
end
%fprintf('迭代1000次结果: %d/20\n',sum);

% 在数据集上进行100个数字的识别检测，得到正确率
sum_test = 0;
sum_test_array = zeros(10, 10);
for i=1:10
    for j=1:10
        k = (i-1)*10 + j;
        m = strcat('trainImgs\',filename(k).name);
        x = imread(m,'bmp');        % 循环读入0-99个样本数字文件 
        [runImgVec, imgNoise, imgFilter, imgBw, imgFt, imgCanny, imgRes] = imgPreProcessingWithFilter(x); % 预处理图像
        num = numPredict(runImgVec,netSize,w1,b1,w2,b2);    % 对图像进行识别
        for a = 1:10
            if labelVec(k,a) ~= 0
                break;
            end
        end
        sum_test_array(i,j) = num;
        if num == a-1
            sum_test = sum_test+1;
        end
    end
end
fprintf('│ TRAINING ACCURACY:  %d/100                                         │\n',sum_test);

% 绘制训练结果图
% x_test = 1:100;
% i=1;
% while i<=100
%     y_test(i) = sum_test_array(fix((i-1)./10)+1, mod(i-1,10)+1);
%     if sum_test_array(fix((i-1)./10)+1, mod(i-1,10)+1)==fix((i-1)./10)
%         scatter(i,sum_test_array(fix((i-1)./10)+1, mod(i-1,10)+1),"blue"); hold on;  % 如果正确
%     elseif sum_test_array(fix((i-1)./10)+1, mod(i-1,10)+1)~=fix((i-1)./10)
%         scatter(i,sum_test_array(fix((i-1)./10)+1, mod(i-1,10)+1),"red"); hold on;   % 如果不正确
%     end
%     i = i+1;
% end

disp("│ TRAINING COMPLETED!                                                │");

% 识别用户指定图像
while true
    p1 = ones(16,16);
    test = input('│ Enter image path and get recognition result:', 's');
    x = imread(test,'bmp');    % 加载图像
    [runImgVec, imgNoise, imgFilter, imgBw, imgFt, imgCanny, imgRes] = imgPreProcessingWithFilter(x); % 预处理图像
    % 显示图片
    subplot(2,3,1); imshow(imgNoise); title("加入人工噪声");
    subplot(2,3,2); imshow(imgFilter); title("低通滤波器降噪");
    subplot(2,3,3); imshow(imgBw); title("二值化");
    subplot(2,3,4); imshow(imgFt); title("高通滤波器边缘检测");
    subplot(2,3,5); imshow(imgCanny); title("Canny边缘检测");
    subplot(2,3,6); imshow(imgRes); title("图片预处理结果");
    num = numPredict(runImgVec,netSize,w1,b1,w2,b2);    % 对图像进行识别
    fprintf('│ 手写体数字识别结果: %d                                               │\n',num);
end