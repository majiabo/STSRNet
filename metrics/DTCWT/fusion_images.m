

%all_models = ["EDSR", "Liff"]
all_models = ['CrossNet', 'CARN', "Deep-Z", "EDSR", "Liff", "MWCNN", "RCAN", "RFANet", "SRCNN","SRResNet","VDSR"]

for m =1:length(all_models)
    img_root = "/mnt/diskarray/mjb/Projects/3DSR/code/segmentation/temp/data/"+all_models(m)+"/original";
    save_root = "/mnt/diskarray/mjb/Projects/3DSR/code/segmentation/temp/data/"+all_models(m)+"/gt5/img"
    % tt = '10140074_36996_66237_0.png';
    s = 512;


    img_names = dir(img_root);
    if exist(save_root)==0
        mkdir(save_root);
    end

    for name = 1:length(img_names)
        name = img_names(name).name
        if isequal(name, '.') || isequal(name, '..')
            continue;
        end
        % for single target
    %     if ~isequal(name, tt)
    %         continue;
    %     end

        img_path = fullfile(img_root, name);
        img = imread(img_path);
        img = imresize(img, 4, 'bicubic');

        gen_2 = fusion_by_dtcwt(img(1:s, 1:s,:), img(1:s, 4*s+1:s*5,:));
%         gen_2 = fusion_by_dtcwt(img(s+1:2*s, 1:s,:), img(s+1:2*s, 4*s+1:s*5,:));
        %gen_1 = fusion_by_dtcwt(img(1+s:2*s, s+1:2*s, :), img(1+s:2*s, 3*s+1:4*s, :));
        gen_1 = fusion_by_dtcwt(img(1:s, s+1:2*s, :), img(1:s, 3*s+1:4*s, :));
        gen_21 = fusion_by_dtcwt(gen_2, gen_1);
        %gen_210 = fusion_by_dtcwt(gen_21, img(1+s:2*s, 2*s+1:3*s, :));
        gen_210 = fusion_by_dtcwt(gen_21, img(1:s, 2*s+1:3*s, :));

    %     hr_2 = fusion_by_dtcwt(img(s+1:end, 1:s,:), img(s+1:end, 4*s+1:s*5,:));
    %     hr_1 = fusion_by_dtcwt(img(1+s:end, s+1:2*s, :), img(1+s:end, 3*s+1:4*s, :));
    %     hr_21 = fusion_by_dtcwt(gen_2, gen_1);
    %     hr_210 = fusion_by_dtcwt(gen_21, img(1+s:end, 2*s+1:3*s, :));

    %     result = [gen_210;hr_210];
        result = gen_210;
        result(result>255) = 255;result(result<0) = 0;
        result = uint8(result);

        save_path = fullfile(save_root, name);
        % remeber to cancel this
    %     result = imresize(result, 0.25, 'bicubic')
        imwrite(result, save_path);
    end
end