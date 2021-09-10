source_path1 = 'D:\\20x_and_40x_data\\split_data\\' ;
target_path1 = 'D:\\20x_and_40x_data\\train_log\\compare\\fusion_3_layer\\DTCWT\\' ;

%source_path2 = 'X:\\GXB\\20x_and_40x_data\\test_result\\compares\\ori_-2-2_layer\\' ;
%target_path2 = 'D:\\20x_and_40x_data\\train_log\\compare\fusion_3_layer\\DTCWT\\' ;

layers1 = [-1 , 0 , 1] ;
layers2 = [-2 , -1 , 0 , 1 , 2];

fid = fopen('D:\\20x_and_40x_data\\test.txt');
tline = fgetl(fid);
while ischar(tline)
    s = regexp(tline, '.tif', 'split') ;
    
    fusion_3_layers = dtcwt_fusion(source_path1 , s{1} , layers1) ;
    imwrite(fusion_3_layers / 256 , [target_path1 s{1} '.tif']);
    
    %fusion_5_layers = dtcwt_fusion(source_path2 , s{1} , layers2) ;
    %imwrite(fusion_5_layers / 256 , [target_path2 s{1} '.tif']);
    
    tline = fgetl(fid);
end
fclose(fid);