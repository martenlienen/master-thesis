% fitplane
%   mm      - local patch of timestamps for estimating 3Dnormal
%   TH      - Threshold for refining 3D normal estimation (not used in this version).
%
% RETURN
%   vx 		- x component of motion flow field (in image NCOLSxNROWS coordinates)
%   vy 		- y component of motion flow field (in image NCOLSxNROWS coordinates)
%             
% DESCRIPTION
%   The function computes the local 3D normal to the data in a patch mm.
%   The output returns the X and Y components of that normal.
%   
%   Copyright (C) 2015  Francisco Barranco, 10/10/2016, Universidad de Granada.
%   License, GNU GPL, free software, without any warranty.

function [vx,vy]=fitplane(mm, TH)
% TH = 1e-4; %for artificial seqs
% TH = .1; %for real-world image seq
vx = 0; vy = 0;

% [M,N]=size(mm);

% Format data for using PCA
[XX,YY]=find(mm>0);
X=[];
for i=1:length(XX)
    X=[X; XX(i) YY(i) mm(XX(i),YY(i))];
end

% Do PCA to extract the local 3D normal
[coeff]=pca_modified(X');

if size(coeff,2)< 3 %get out when there are not enough points
    return
end

normal = coeff(:,3);

% This is just for debugging: Visualize the 3D plane fitted
% meanX = mean(X,1);
% 
% [xgrid,ygrid] = meshgrid(1:M,1:N);
% zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2)));
% 
% if (norm(cross(normal,[0, 1,0]))>TH)
%     [vx,vy]=gradient(zgrid);
%     vx = 1./vx*1e6;
%     vy = 1./vy*1e6;
% end

% Applying normalization proposed in Ruckaer and Delbruck, 2016 and translating into pix/s
vx = -normal(3)/(normal(2)^2+normal(1)^2)*normal(1)*1e6;
vy = -normal(3)/(normal(2)^2+normal(1)^2)*normal(2)*1e6;
end

% OLD VERSIONS
% function [vx,vy]=fitplane(mm)
% [M,N]=size(mm);
% 
% 
% [XX,YY]=find(mm>0);
% X=[];
% for i=1:length(XX)
%     X=[X; XX(i) YY(i) mm(XX(i),YY(i))];
% end
% 
% 
% [coeff]=pca_modified(X');
% 
% if size(coeff,2)< 3 %get out when there are no enough points
%     vx = 0; vy = 0;
%     return
% end
% 
% normal = coeff(:,3);
% 
% meanX = mean(X,1);
% 
% [xgrid,ygrid] = meshgrid(1:M,1:N);
% zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2)));
% 
% if (norm(cross(normal,[0, 1,0]))>.1)
% [vx,vy]=gradient(zgrid);
% vx = 1./vx*1e6; 
% vy = 1./vy*1e6;
% else vx=0;vy=0;
% end

% function [vx,vy]=fitplane(mm)
% % Generate some trivariate normal data for the example.  Two of
% % the variables are correlated fairly strongly.
% %X = mvnrnd([0 0 0], [1 .2 .7; .2 1 0; .7 0 1],50);
% %load mm
% %close all
% 
% [M,N]=size(mm);
% new_mm = mm;
% list = mm(:); list(list==0)=[]; new_mm=mm-min(list)+1; new_mm(new_mm<0)=0; 
% mm = new_mm;
% 
% [XX,YY]=find(mm>0);
% X=[];
% for i=1:length(XX)
%     X=[X; XX(i) YY(i) mm(XX(i),YY(i))];
% end
% 
% 
% % Next, fit a plane to the data using PCA.  The coefficients for the
% % first two principal components define vectors that span the plane; the
% % third PC is orthogonal to the first two, and its coefficients define the
% % normal vector of the plane.
% [coeff,score] = princomp(X);
% coeff(:,1:2);
% normal = coeff(:,3);
% 
% [n,p] = size(X);
% meanX = mean(X,1);
% Xfit = repmat(meanX,n,1) + score(:,1:2)*coeff(:,1:2)';
% residuals = X - Xfit;
% 
% % The equation of the fitted plane is (x,y,z)*normal - meanX*normal = 0.
% % The plane passes through the point meanX, and its perpendicular distance
% % to the origin is meanX*normal.  The perpendicular distance from each
% % point to the plane, i.e., the norm of the residuals, is the dot product
% % of each centered point with the normal to the plane.  The fitted plane
% % minimizes the sum of the squared errors.
% error = abs((X - repmat(meanX,n,1))*normal);
% sse = sum(error.^2);
% [XX,YY]=find(mm==0);
% 
% % figure,plot3(X(:,1),X(:,2),X(:,3),'*r'),hold,plot3(Xfit(:,1),Xfit(:,2),Xfit(:,3),'ob')
% 
% %[xgrid,ygrid] = meshgrid(linspace(min(X(:,1)),max(X(:,1)),5), ...
% %                         linspace(min(X(:,2)),max(X(:,2)),5));
%                      
% [xgrid,ygrid] = meshgrid(1:M,1:N);
% zgrid = (1/normal(3)) .* (meanX*normal - (xgrid.*normal(1) + ygrid.*normal(2)));
% %h = mesh(xgrid,ygrid,zgrid,'EdgeColor',[0 0 0],'FaceAlpha',0);
% 
% 
% 
% mm2=zeros(N,M);
% for i=1:N
%     for j=1:M
%         mm2(xgrid(i,j),ygrid(i,j))=zgrid(i,j);
%     end
% end
% if (norm(cross(normal,[0, 1,0]))>.1)
% [vx,vy]=gradient(mm2);
% vx = 1./vx*1e6; vy = 1./vy*1e6; % To give it in pix/s (initially in pix/us)
% else vx=0;vy=0;
% end