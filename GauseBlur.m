function resImg = GauseBlur(sigma, wsize, contrast, img)
%GAUSEBLUR 高斯模糊（参数1为高斯分布sigma，参数2为卷积核大小，参数3为对比度，参数4为图片）
%高斯平滑，sigma越小，wsize越大，图像越不突出，contrast对比度，值越大，图像最终对比越明显
    border=(wsize-1)/2;
    [m,n]=size(img);
    imgborder=repmat(255,[m+wsize-1,n+wsize-1]);
    imgborder(border+1:border+m,border+1:border+n)=img;
    
    musk=zeros(wsize,wsize);
    sum=0;
    for i=-border:border
        for j=-border:border
            musk(i+border+1,j+border+1)=exp(-(i^2+j^2)/2/sigma/sigma);
            sum=sum+musk(i+border+1,j+border+1);
        end
    end
    musk=musk/sum;
    
    matsum=0;
    for i=1:m
        for j=1:n
            matsum=matsum+img(i,j);    % 图像像素值求和
        end
    end
    matavr=matsum\m\n;    % 图像像素平均值
    
    for i=1:m
        for j=1:n
            p=imgborder(i:i+2*border,j:j+2*border).*musk;
            res=0;
            for r=1:wsize
                for s=1:wsize
                res=p(r,s)+res;
                end
            end
            matavr = 200;
            res=(res-matavr)*contrast+matavr;    % 对比度
            if(res > 255)
                res=255;
            elseif(res < 0)
                res=0;
            end
            imgborder(i+border,j+border)=res;
    %imgborder(i+border,j+border)=1;
        end
    end
    img=imgborder(border:border+m,border:border+n);
    resImg=img;
end


