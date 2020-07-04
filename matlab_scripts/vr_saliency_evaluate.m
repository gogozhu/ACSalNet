clc,clear;

folderName = '/home/lab-zhu.huansheng/workspace/dataset/icme17_salicon_like/vr_saliency/maps/val';
exp_folder = '/home/lab-zhu.huansheng/workspace/GradProj/hrnet_proj/output/silent360/ACSalNet4_noP/ACSalNet4_s360_cube_hfpn5/s1/MCP_vr_saliency_30_2048/'

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
    SalMap1 = imread([folderName '/P' num2str(filenum) '.jpg']);
    imw = size(SalMap1, 2);
    iml = size(SalMap1, 1);

    SalMap2 = imresize(SalMap2, [iml, imw]);

    % [scoreKL,scoreCC,scoreNSS,scoreROC]=CompareHeadEyeSalMaps(SalMap2,folderName,filenum);
    
    height=size(SalMap2,1);
    width=size(SalMap2,2);

    N=floor(4*pi*100000);   %we sample the sphere at every 100000 times for every steradian
    
    sampPoints=SpiralSampleSphere(N,0);
    spOne=zeros(N,1); spTwo=zeros(N,1);
    for k=1:N
        ang=atan2(sampPoints(k,2),sampPoints(k,1));
        if ang<0
            ang=ang+(2*pi);
        end;
        xCoord=max(min(1+floor(width*(ang/(2*pi))),width),1);                                                 % mapping the spherical to equirectangular
        yCoord=max(min(1+floor(height*(asin(sampPoints(k,3)/norm([sampPoints(k,:)]))/pi+0.5)),height),1);     % mapping the spherical to equirectangular
        spOne(k)=SalMap1(yCoord,xCoord);
        spTwo(k)=SalMap2(yCoord,xCoord);
    end;

    scoreCC = CC(spOne, spTwo);

    CC_list(i-2) = scoreCC;

end

CC_list

avgKL = mean(KL_list)
avgCC = mean(CC_list)
avgNSS = mean(NSS_list)
avgAUC = mean(AUC_list)

% avgKL = avgKL / (length(imlist)-2)
% avgCC = avgCC / (length(imlist)-2)
% avgNSS = avgNSS / (length(imlist)-2)
% avgAUC = avgAUC / (length(imlist)-2)
