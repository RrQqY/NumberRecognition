function res = GauseFilter(img, d0)
%GAUSEFILTER 低通滤波器（参数1为图像，参数2为设定的阈值）
%     d0 = 80;    % 我们设定的阈值
    img_noise = img;
    img_fft = fftshift(fft2(double(img_noise)));   % 傅里叶变换得到频谱
    [m,n] = size(img_fft);
    m_mid = floor(m/2);    % 原图像高的一半 用于得到中心点坐标
    n_mid = floor(n/2);    % 原图像宽的一半 用于得到中心点坐标
    distance = zeros(m,n);
    h = zeros(m,n);
    img1 = zeros(m,n);
    for i = 1 : m
        for j = 1 : n
            % 对得到的频谱进行理想低通滤波
            distance(i,j) = sqrt((i - m_mid) ^ 2+(j - n_mid) ^ 2);
            h(i,j) = exp(-(distance(i,j) ^ 2) / (2 * d0 ^ 2));
            img1(i,j) = img_fft(i,j) * h(i,j);
        end
    end
    
    img1 = ifftshift(img1);    % 反傅里叶变换
    img1 = uint8(real(ifft2(img1)));
    res = img1;
end

