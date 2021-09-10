function y = wt_fusion(path , cz , layers)

imgF = imread([path cz '_0.tif']) ; % read the layer_0 img

for i = 1 : (length(layers) / 2) % read other layer img and fusion
    left = layers(i) ;
    right = layers(length(layers) - i + 1) ;
    img1 = imread([path cz '_' num2str(left) '.tif']); 
    img2 = imread([path cz '_' num2str(right) '.tif']);
    temp = wfusimg(img1 , img2 , 5 , 'max' , 'max');
    
    imgF = wfusimg(imgF , temp , 'sym4' , 5 , 'max' , 'max');
end
y = imgF;
end
