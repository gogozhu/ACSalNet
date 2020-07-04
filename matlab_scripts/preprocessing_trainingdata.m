clc,clear;
folder = '/home/lab-zhu.huansheng/workspace/dataset/icme2017';
sub_folder = '/HeadEyeImages/'
outfolder = '../data/icme17_salicon_like/local/cube_9_noNorm';
state = 'train';
mkdir([outfolder '/images/train']);
mkdir([outfolder '/maps/train']);
imlist = dir([folder sub_folder]);

% folder = '/home/lab-zhu.huansheng/workspace/dataset/icme2017/Eval';
% sub_folder = '/HeadEyeImages/'
% outfolder = '../data/icme17_salicon_like/local/cube_9_noNorm';
% state = 'val';
% mkdir([outfolder '/images/val']);
% mkdir([outfolder '/maps/val']);
% imlist = dir([folder sub_folder]);

vfov = 90;
headmove_h = 0:30:60
headmove_v = 0:30:60


output_size = [512, 512]

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

    % read salmap
    fileId = fopen([folder '/HeadEyeSalMaps_bin/SHE' filenum '.bin'], 'rb');
    buf = fread(fileId, iml * imw, 'single');
    salmap = reshape(buf, [imw, iml])';
    salmap = salmap./max(salmap(:));
    fclose(fileId);

    for hh = 1:length(headmove_h)
        offset = round(headmove_h(hh)/360*imw);
        sal_offset = round(headmove_h(hh)/360*imw);
        im_turned = [img(:, imw-offset+1:imw, :) img(:, 1:imw-offset, :)];
        sal_turned = [salmap(:, imw-sal_offset+1:imw) salmap(:, 1:imw-sal_offset)];
        for hv = 1:length(headmove_v)
            [out] = equi2cubic(im_turned, iml, vfov, headmove_v(hv));
            [sal_out] = equi2cubic(sal_turned, iml, vfov, headmove_v(hv));
            for f=1:6
                imwrite(imresize(cell2mat(out(f)), output_size), [outfolder '/images/' state '/P' filenum '_' num2str(hv) num2str(hh) num2str(f) '.jpg']);
                sal_out_f = cell2mat(sal_out(f));
                % sal_out_f = sal_out_f./max(sal_out_f(:));
                imwrite(imresize(sal_out_f, output_size), [outfolder '/maps/' state '/SHE' filenum '_' num2str(hv) num2str(hh) num2str(f) '.png']);
            end
       end
    end
end
