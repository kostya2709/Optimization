%=======================================================%
% Programmer: Tsung-Han Chan
% National Tsing Hua University, Hsinchu, Taiwan 
% E-mail: d935608@oz.nthu.edu.tw
% Date: April. 19, 2009
% -------------------------------------------------------
% Reference: 
% T.-H. Chan, W.-K. Ma, C.-Y. Chi, and Y. Wang, ``A convex analysis 
% framework for blind separation of non-negative sources," IEEE Trans. Signal Process., 
% vol. 56, no. 10, pp. 5120-5134, Oct. 2008. 
%=======================================================
% This is a simple example for human face separation using CAMNS algorithm
% N is number of sources
% M is number of observations (sensors)
clear all;close all;
N=3;
M=3; 
%=============Read images===============
image1=double(imread('zhang1.jpg'));
image11=image1/255;
image2=double(imread('cao1.jpg'));
image22=image2/255;
image3=double(imread('ksiwek1.jpg'));
image33=image3/255;
[m,n]=size(image1);
L=m*n; % data length
s1=vec(image11);
s2=vec(image22);
s3=vec(image33);
SS=[s1 s2 s3]'; % source matrix N by L
%===========randomly unit row-sum mixing matrix===========
AA=rand(N,M);
A=(AA./(ones(N,1)*sum(AA))).'; % mixing matrix
X=A*SS;       % observation matrix M by L
%-------------------------------------------------
Y=CAMNS_LP(X',N);
% %============vector to matrix (image)=================
X1=reshape(X(1,:),m,n);
X2=reshape(X(2,:),m,n);
X3=reshape(X(3,:),m,n);
Y1=reshape(Y(:,1),m,n);
Y2=reshape(Y(:,2),m,n);
Y3=reshape(Y(:,3),m,n);
% % %==========plotting===============
figure;subplot(1,3,1);imshow(image11);
subplot(1,3,2);imshow(image22);title('Original sources');subplot(1,3,3);imshow(image33);
figure;subplot(1,3,1);imshow(X1);
subplot(1,3,2);imshow(X2);title('Source mixtures');subplot(1,3,3);imshow(X3);
figure; subplot(1,3,1);imshow(Y1);
subplot(1,3,2);imshow(Y2);title('Extracted sources by CAMNS_{LP}');subplot(1,3,3);imshow(Y3);

