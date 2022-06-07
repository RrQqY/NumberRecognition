function num = numPredict(runImgVec,netSize,w1,b1,w2,b2)
%NUMPREDICT
%输入向量经过神经网络预测输出一组结果
    x1_run = zeros(netSize(1),1);          % 隐层输入
    y1_run = zeros(netSize(2),1);           % 隐层直接输出
    a1_run = zeros(netSize(2),1);           % 隐层经过激发函数后输出
    x2_run = zeros(netSize(2),1);           % 输出层输入（就是a1）
    y2_run = zeros(netSize(3),1);           % 输出层直接输出
    a2_run = zeros(netSize(3),1);           % 输出层经过激发函数后输出（最终输出）

    % 获得一个测试集样本输入的神经网络输出
    x1_run = runImgVec';
    y1_run = w1 * x1_run + b1;
    a1_run = sigmoid(y1_run);
    x2_run = a1_run;
    y2_run = w2 * x2_run + b2;
    a2_run = sigmoid(y2_run);
    % 对于神经网络的输出，取每一列的最大值为1，其他为0
    maxi = -1;
    max = 0;
    for k = 1:10
        if a2_run(k) > max
            maxi = k;
            max = a2_run(k);
        end
    end
    for k = 1:10
        if k == maxi
            a2_run(k) = 1;
            num = k-1;
        else
            a2_run(k) = 0;
        end
    end
end

