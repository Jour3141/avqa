clc
clear

addpath('YUVtoolbox\')
videoPath = 'C:/Users/linji/Desktop/AVQA/released_code/Distorted AV/Boxing_QP50_S.yuv';

resolution = [720 1280];  
frameSkip = 2;
patchSize = 224;
position_width = [];
position_height = [];

for h = 1:200:resolution(1)
    if h < resolution(1) - patchSize
        for w = 1: 200: resolution(2)
            if w < resolution(2) - patchSize + 1
                position_width = [position_width, w];
                position_height = [position_height, h];
            else
                position_height = [position_height, h];
                position_width = [position_width, resolution(2) - patchSize + 1];
                break
            end
        end
    else
        for w = 1: 200: resolution(2)
            if w < resolution(2) - patchSize + 1
                position_height = [position_height, resolution(1) - patchSize + 1];
                position_width = [position_width, w];
            else
                position_height = [position_height, resolution(1) - patchSize + 1];
                position_width = [position_width, resolution(2) - patchSize + 1];
                break
            end
        end
        break
    end
end
position = int16([position_height; position_width]);

sort_frame = [];
disID = fopen(videoPath);

for iframe = 1:192
    iframe
    [disY, disCb, disCr] = readframeyuv420(disID, resolution(1), resolution(2));

    if mod(iframe,frameSkip)~=1
        continue
    end
    
    disY = reshape(disY, [resolution(2) resolution(1)])';
    disCb = reshape(disCb, [resolution(2)/2 resolution(1)/2])';
    disCr = reshape(disCr, [resolution(2)/2 resolution(1)/2])';
    
    disRGB = yuv2rgb(disY,disCb,disCr);
    sal_img = fes_index(disRGB);
    sal_img = imresize(sal_img,resolution(1)/size(sal_img,1),'bicubic');
    
    sal_sum = zeros(1,length(position));
    for iposition = 1:length(position)
        sal_sum(iposition) = sum(sum(sal_img(position(1,iposition):position(1,iposition)+patchSize-1, ...
        position(2,iposition):position(2,iposition)+patchSize-1)));
    end
    
    [sal_sum, sort_position] = sort(-sal_sum);
    sort_frame = [sort_frame;sort_position(1:25)];     
end

fclose(disID);
sal_index = sort_frame;
save('./V_position_single.mat','sal_index');
