function [p3, imgNoise, imgFilter, imgBw, imgFt, imgCanny, imgRes] = imgPreProcessingWithFilter(x)
%IMGPREPROCESSINGWITHFILTER 图像预处理（含滤波器）
%返回参数分别为：最终处理结束的向量、加入人工噪声后原图、低通滤波器降噪后图、二值化后图
%               高通滤波器边缘检测后图、canny边缘检测后图、处理完成后图
%灰度化 => 人工噪声 => 高斯模糊/低通滤波器降噪 => 二值化 => 高通滤波器边缘检测 => 取反 => ROI => 变尺寸 => 向量化
    p1 = zeros(16,16);          % 建立全为1的样本矩阵

    % ----灰度化----%
    if(length(size(x))==3)
        x = rgb2gray(x);        % 将图像转灰度
    end

    % ----人工噪声----%
    imgNoise=imnoise(x,'gaussian',0.251,0.0615);
    imgNoise=imnoise(imgNoise,'salt & pepper',0.05);
%     imshow(imgNoise);

    % ----高斯模糊/低通滤波器进行去噪----%
    imgFilter=GauseFilter(imgNoise,80);       % 低通滤波
    imgFilter=GauseBlur(25,5,2,imgFilter);    % 高斯模糊
    imgFilter=uint8(imgFilter);
%     imshow(imgFilter);

    % ----二值化----%
    bw=imbinarize(imgFilter,0.38);       % 将图像以0.38为阈值二值化处理

    imgBw = bw;
 %     imshow(bw); 

    %bw = edge(bw,'canny');     % 用canny算子进行边界提取
    % ----高通滤波器边缘检测----%
    bw3 = ftFilter(bw);
%     imshow(bw3);
    [l,r]=size(bw3);

    % ----再次二值化----%
    imgFt=imbinarize(bw3,0.31);       % 将图像以0.5为阈值二值化处理
%     imshow(imgFt);
    imgCanny = edge(imgBw,'canny');   % canny边缘检测进行对比 

    % ----取反色----%
    for i=1:l
        for j=1:r
            if imgBw(i,j)==0
                imgBw(i,j)=1;
            else
                imgBw(i,j)=0;
            end
        end
    end
%     imgFt = imgBw;
%     imshow(imgFt);

    % ----ROI----%
    [i,j] = find(imgBw==1);            % 找到白色的像素部分（数字部分）
    imin=min(i);
    imax=max(i);
    jmin=min(j);
    jmax=max(j);
    bw2=imgBw(imin:imax,jmin:jmax);  % 截取图像中的数字部分

    % ----尺寸归一化----%
    rate = 16/max(size(bw2));
    bw2 = imresize(bw2,rate);        % 对输入文件变尺寸处理，变为16*16的矩阵
    [i,j] = size(bw2);
    i1 = round((16-i)/2);
    j1 = round((16-j)/2);
    p1(i1+1:i1+i,j1+1:j1+j) = bw2;   % 得到16*16的图像
    p1 = bwmorph(p1,'thin',inf);     % 图像细化
    imgRes = p1;
%     imshow(p1);
    
    % ----向量化----%
    p2 = zeros(1,16*16);
    % 将16*16的图像矩阵转化为1*256的向量
    for i=0:15
        for j=1:16
            p2(i*16+j) = p1(i+1,j);
        end
    end
    p3 = p2;        % 返回处理完的图像
%     imshow(p3);
end

