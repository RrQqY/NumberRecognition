function [p3, bw] = imgPreProcessing(x)
%IMGPREPROCESSING 图像预处理 
%灰度化 => 二值化 => 取反 => ROI => 变尺寸 => 向量化
    p1 = zeros(16,16);          % 建立全为1的样本矩阵

    % ----灰度化----%
    if(length(size(x))==3)
        x = rgb2gray(x);        % 将图像转灰度
    end

    % ----二值化----%
    bw=imbinarize(x,0.5);       % 将图像以0.5为阈值二值化处理
    %bw = edge(bw,'canny');     % 用canny算子进行边界提取
    [l,r]=size(bw);
    bw1 = bw;                   % 备份bw

    % ----取反色----%
    for i=1:l
        for j=1:r
            if bw1(i,j)==0
                bw1(i,j)=1;
            else
                bw1(i,j)=0;
            end
        end
    end

    % ----ROI----%
    [i,j] = find(bw1==1);            % 找到白色的像素部分（数字部分）
    imin=min(i);
    imax=max(i);
    jmin=min(j);
    jmax=max(j);
    bw2=bw1(imin:imax,jmin:jmax);    % 截取图像中的数字部分

    % ----尺寸归一化----%
    rate = 16/max(size(bw2));
    bw2 = imresize(bw2,rate);        % 对输入文件变尺寸处理，变为16*16的矩阵
    [i,j] = size(bw2);
    i1 = round((16-i)/2);
    j1 = round((16-j)/2);
    p1(i1+1:i1+i,j1+1:j1+j) = bw2;   % 得到16*16的图像
    p1 = bwmorph(p1,'thin',inf);     % 图像细化
    
    % ----向量化----%
    p2 = zeros(1,16*16);
    % 将16*16的图像矩阵转化为1*256的向量
    for i=0:15
        for j=1:16
            p2(i*16+j) = p1(i+1,j);
        end
    end
    p3 = p2;        % 返回处理完的图像
end