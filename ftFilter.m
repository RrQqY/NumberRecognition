function p = ftFilter(x)
%二维傅里叶变换+高通滤波器提取边缘
    ft_x = fft2(x);                                       % 二维傅里叶变换
    ft_x = fftshift(ft_x);                                % 将低频搬运至中间
    D0 = 10;                                              % 高通滤波器阻带半径
    [len_x,len_y] = size(ft_x);
    mid_x = fix(len_x/2);
    mid_y = fix(len_y/2);                                 % 取得中心点坐标

    for i=1:len_x
        for j=1:len_y
            D = sqrt((j-mid_x)^2 + (i-mid_y)^2);
            ft_x(i,j) = (1-exp(-D*D/(2*D0*D0)))*ft_x(i,j);
        end
    end                                                  % 高斯滤波
    
    ft_x = ifftshift(ft_x);                              % 中心搬运回原处
    bw2 = ifft2(ft_x);                                   % 进行逆变换
    p = real(bw2);
end
