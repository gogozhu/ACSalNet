clc,clear;

state = 'test';

folder = '/home/lab-zhu.huansheng/workspace/dataset/icme17_salicon_like/vr_saliency/images/test';
sub_folder = '/'
outfolder = '/home/lab-zhu.huansheng/workspace/dataset/icme17_salicon_like/vr_saliency_30';
mkdir([outfolder '/images/' state]);
imlist = dir([folder sub_folder]);

vfov = 90;
headmove_h = 0:45:315
headmove_v = -30:30:30
out_wid = 1024;
out_len = 512;

length(imlist)

parfor i = 3:length(imlist)
    im_filename = imlist(i).name;
    im_filename
    filenum = regexp(im_filename, 'P', 'split');
    filenum = regexp(cell2mat(filenum(2)), '\.', 'split');
    filenum = cell2mat(filenum(1));

    % read image
    img = imread([folder sub_folder im_filename]);
    imw = size(img, 2);
    iml = size(img, 1);

    tmp_sal = zeros(iml, iml, 6, 3);
    for hh = 1:length(headmove_h)
        offset = round(headmove_h(hh)/360*imw);
        im_turned = [img(:, imw-offset+1:imw, :) img(:, 1:imw-offset, :)];
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, iml, vfov, headmove_v(hv));
            out = cubic2equi(0,cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
            imwrite(imresize(out, [out_len out_wid]), [outfolder '/images/' state '/P' filenum '_' num2str(hv) num2str(hh) '.jpg']);
       end
    end
end
