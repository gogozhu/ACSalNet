clc,clear;

state = 'train';

folder = '/home/lab-zhu.huansheng/workspace/dataset/icme17_salicon_like/vr_saliency/maps/val';
sub_folder = '/'
% exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/silent360/hfpn5_triAttNet/s360_global_hfpn5/s11'
% exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/silent360/hfpn5_triAttNet/s360_cube_hfpn5/s25'
% exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/salicon17/ACSalNet4_noP/ACSalNet4_noP/s1'
exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/silent360/ACSalNet4_noP/ACSalNet4_s360_cube_hfpn5/s1/'
exp_sub_folder = '/test_vr_saliency_30_2048'
outfolder = '/MCP_vr_saliency_30_2048_drt';

mkdir([exp_folder outfolder]);

imlist = dir([folder sub_folder]);

vfov = 90;
headmove_h = 0:45:315
headmove_v = -30:30:30

length(imlist)

for i = 3:length(imlist)
    im_filename = imlist(i).name;
    im_filename
    filenum = regexp(im_filename, 'P', 'split');
    filenum = regexp(cell2mat(filenum(2)), '\.', 'split');
    filenum = cell2mat(filenum(1));

    img = imread([exp_folder exp_sub_folder '/P' filenum '_21.png']);
    imw = size(img, 1);
    out_wid = imw*2;
    out_len = imw;
    img = imresize(img, [out_len out_wid]);
    imwrite(img, [exp_folder outfolder '/SHE' filenum '.png']);
end

% parfor i = 3:length(imlist)
%     im_filename = imlist(i).name;
%     im_filename
%     filenum = regexp(im_filename, 'P', 'split');
%     filenum = regexp(cell2mat(filenum(2)), '\.', 'split');
%     filenum = cell2mat(filenum(1));

%     img = imread([exp_folder exp_sub_folder '/P' filenum '_11.png']);
%     imw = size(img, 1);
%     out_wid = imw*2;
%     out_len = imw;

%     sal_out = zeros(out_len, out_wid, 3);
%     im_salgan_0 = zeros(imw,imw*2);

%     for hh = 1:length(headmove_h)
%         for hv = 1:length(headmove_v)
%             % read image
%             img = double(imread([exp_folder exp_sub_folder '/P' filenum '_' num2str(hv) num2str(hh) '.png']));
%             sal_out(:,:,1) = img;
%             sal_out(:,:,2) = img;
%             sal_out(:,:,3) = img;
%             out = equi2cubic(sal_out, imw, vfov, -headmove_v(hv));
%             im_salgan = cubic2equi(-headmove_h(hh),cell2mat(out(5)),cell2mat(out(6)),cell2mat(out(4)),cell2mat(out(2)),cell2mat(out(1)),cell2mat(out(3)));
%             im_salgan = im_salgan(:,:,1);

%             % imwrite(im_salgan, [exp_folder outfolder '/SHE' filenum '_' num2str(hv) num2str(hh) '.png']);

%             im_salgan = double(im_salgan)+im_salgan_0;
%             im_salgan_0 = im_salgan;
%        end
%     end
%     im_salgan = im_salgan./(hh*hv);
%     im_salgan = im_salgan/max(max(im_salgan));
%     im_salgan = imresize(im_salgan, [out_len out_wid]);
%     imwrite(im_salgan, [exp_folder outfolder '/SHE' filenum '.png']);
% end
