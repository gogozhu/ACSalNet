clc,clear;

folderName = '/home/lab-zhu.huansheng/workspace/dataset/icme2017/Eval';
exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/silent360/ACSalNet4_noP/ACSalNet4_s360_cube_hfpn5/s1/MCP_1024'

imlist = dir(exp_folder);

length(imlist)

avgKL = 0.0;
avgCC = 0.0;
avgNSS = 0.0;
avgAUC = 0.0;

KL_list = zeros(1, length(imlist)-2);
CC_list = zeros(1, length(imlist)-2);
NSS_list = zeros(1, length(imlist)-2);
AUC_list = zeros(1, length(imlist)-2);

parfor i = 3:length(imlist)
    im_filename = imlist(i).name;
    im_filename
    filenum = regexp(im_filename, 'SHE', 'split');
    filenum = regexp(cell2mat(filenum(2)), '\.', 'split');
    filenum = cell2mat(filenum(1));

    SalMap2 = double(imread([exp_folder '/' im_filename]));

    % read image
    img = imread([folderName '/HeadEyeImages/P' num2str(filenum) '.jpg']);
    imw = size(img, 2);
    iml = size(img, 1);
    SalMap2 = imresize(SalMap2, [iml, imw]);

    [scoreKL,scoreCC,scoreNSS,scoreROC]=CompareHeadEyeSalMaps(SalMap2,folderName,filenum);
    % avgKL = avgKL + scoreKL;
    % avgCC = avgCC + scoreCC;
    % avgNSS = avgNSS + scoreNSS;
    % avgAUC = avgAUC + scoreROC;

    KL_list(i-2) = scoreKL;
    CC_list(i-2) = scoreCC;
    NSS_list(i-2) = scoreNSS;
    AUC_list(i-2) = scoreROC;
end

avgKL = mean(KL_list)
avgCC = mean(CC_list)
avgNSS = mean(NSS_list)
avgAUC = mean(AUC_list)

% avgKL = avgKL / (length(imlist)-2)
% avgCC = avgCC / (length(imlist)-2)
% avgNSS = avgNSS / (length(imlist)-2)
% avgAUC = avgAUC / (length(imlist)-2)
