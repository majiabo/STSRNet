function y = dtcwt_fusion(path , cz , layers)

imgF = imread([path cz '_0.tif']) ; % read the layer_0 img

for i = 1 : (length(layers) / 2) % read other layer img and fusion
    left = layers(i) ;
    right = layers(length(layers) - i + 1) ;
    img1 = imread([path cz '_' num2str(left) '.tif']); 
    img2 = imread([path cz '_' num2str(right) '.tif']);
    temp = fusion_by_dtcwt(img1 , img2);
    
    imgF = fusion_by_dtcwt(imgF , temp);
end
y = imgF;
end
