%%% solution script test exam 02502 
%%% Bjørn Marius Schreblowski Hansen
%%% s183986
%%% 23/11-2021
%%
clc, clear, close all
path_folder = 'C:\Users\lowes\OneDrive\Skrivebord\DTU\TA_ting\Billedanalyse E21\Matlab exercises\solution_script' ;
cd(path_folder)
addpath data
%% 
M = load('data\irisdata.txt');
M = M(:,1:4);
M = M - mean(M,1); %subtracting mean
%% Q1
Cx = cov(M);
[PC, V] = eig(Cx);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
Vnorm = V / sum(V) * 100;
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
Vnorm(1)+Vnorm(2) %% 92.4%
%% Q2
signals = PC' * M';
signals(:,1)' %%-2.6841    0.3194    0.0279   -0.0023
%%
[vectors,values,psi] = pc_evectors(M',4);

sigs = vectors'*M';
sigs(:,1)
%% Q3
SG = double(imread('data\sky_gray.png'));
%imshow(SG)
st_SG = (200-10) /(max(SG(:)) - min(SG(:))) * (SG - min(SG(:))) + 10;
mean(st_SG(:)) %%87
%% Q4
SG_RGB = double(imread('data\sky.png'));
lgcl_SG = SG_RGB(:,:,1) < 100 & SG_RGB(:,:,2) > 85 & SG_RGB(:,:,2) < 200 & SG_RGB(:,:,3) > 150;
se1 = strel('disk',5);
out = imerode(lgcl_SG,se1);
sum(out(:)) %19977
%% Q5
flwer = imread('data\flower.png');
flwer_hsv = rgb2hsv(flwer);
lgcl_flwer = flwer_hsv(:,:,1) < 0.25 & flwer_hsv(:,:,2) > 0.8 & flwer_hsv(:,:,3) > 0.8;
se1 = strel('disk',5);
out = imopen(lgcl_flwer,se1);
sum(out(:)) %5665
%% Q6
Mc = zeros(800*600,5);
for i = 1:5
    str = ['data\car',num2str(i),'.jpg'];
    tmp_pic = double(imread(str));
    Mc(:,i) = tmp_pic(:) - mean(tmp_pic(:));
end
[vectors,values,psi] = pc_evectors(Mc,5);
values(1)/sum(values)
[nv,nd] = sortem(vectors',diag(values));
vnorm = nd/sum(nd(:)) * 100
%%

Cx = 1/size(Mc,1) * Mc' * Mc;
%Cx = cov(Mc);
[PC, V] = eig(Cx);
V = diag(V);
[junk, rindices] = sort(-1*V);
V = V(rindices);
PC = PC(:,rindices);
Vnorm = V / sum(V) * 100;
plot(Vnorm, '*-')
ylabel('Percent explained variance')
xlabel('Principal component')
Vnorm(1) %% 62.1%?
%% Q7
SG = double(imread('data\sky_gray.png'));
SG = 255*((SG / 255).^1.21);
SG = medfilt2(SG,[5,5]);
SG(40,50) %68.4494
%% Q8
FW = double(imread('data\flowerwall.png'));
windowWidth = 15;
se = ones(windowWidth) / windowWidth .^ 2;
out = imfilter(FW,se);
out(5,50) %167
%% Q9
FB = double(imread('data\floorboards.png'));
lgcl_FB = FB < 100;
se1 = strel('disk',10);
se2 = strel('disk',3);
fnl_FB = imclearborder(imopen(imclose(lgcl_FB,se1),se2));
L8 = bwlabel(fnl_FB,8);
imagesc(L8);
colormap(hot);
title('8 connectiviy')
sum(fnl_FB(:)) %6735
%% Q10
stats8 = regionprops(L8, 'Area');
bw2 = numel(find([stats8.Area] > 100)) %16
%% Q11
im = imread('books_bw.png');
im_lab = bwlabel(im,8);
stats = regionprops(im_lab,'all');
idx = find([stats.Area] > 100 & [stats.Perimeter] > 500 );
im_bw = ismember(im_lab,idx);
imshow(im_bw)
%% Q12
load('catfixedPoints.mat')
load('catmovingPoints.mat')
sum((fixedpoints-movingpoints).^2,'all')
%% Q13
mytform = fitgeotrans(movingpoints, fixedpoints,'NonreflectiveSimilarity');
forward = transformPointsForward(mytform,movingpoints);

% plot them together with the points from hand1. What do you observe?
cat = im2double(imread('cat2.png'));
cat_moved = imwarp(cat, mytform);

%Show the transformed version of hand2 together with hand1. What do you observe?
subplot(1,2,1)
imshow(cat)
title('cat')
subplot(1,2,2)
imshow(cat_moved)
title('cat moved')
%% Q14
dc = double(dicomread('1-179.dcm'));

liver = imread('LiverRoi.png');
dc_l = dc(liver);
T = mean(dc_l,'all')+[-1,1]*std(dc_l,[],'all')

dc_t = dc>T(1) & dc<T(2) ;
sum(dc_t(:))
%% Q15
dc = double(dicomread('1-179.dcm'));
T = [90, 140];
dc_t = dc>T(1) & dc<T(2) ;

se = strel('disk',3);

dc_close = imclose(dc_t,se);
dc_open = imopen(dc_close,se);

im_lab = bwlabel(dc_open,8);
stats = regionprops(im_lab,'area');
max([stats.Area])

%% Q16
T = [(3+7), (7+15)]/2

%% Q17
xrange = 0:0.01:20;
pdf1 = normpdf(xrange, 3, 5);
pdf2 = normpdf(xrange, 7, 2);
pdf3 = normpdf(xrange, 15, 5);

plot(xrange,[pdf1;pdf2;pdf3])
% The Gaussians crosses in 4.24 and 10.26

%% Q18
im = [167,193, 180;
      9, 189, 8;
      217, 100, 71];
tem = [208, 233, 71;
       231, 161, 139;
       32, 25, 244];
   
sum(im.*tem,'all')/sqrt(sum(im(:).^2)*sum(tem(:).^2))

%% Q19
f = 10;
g = 1100;
fish = 400;
pixel_mm = 6480/5.4;

b = 1/(1/f-1/g);
% assume b = f ?
b = f;
B = b*fish/g;
B*pixel_mm

%% Q20
X = [1, 1; 2.2, -3; 3.5, -1.4; 3.7, -2.7; 5, 0;
    0.1, 0.7; 0.22, -2.1; 0.35, -0.98; 0.37, -1.89; 0.5, 0];
T = [zeros(5,1); ones(5,1)];
W = LDA(X,T);
ex1 = [1; 1; 1];
Y = W*ex1;
exp(Y)./sum(exp(Y)) % 0.81 for class 1