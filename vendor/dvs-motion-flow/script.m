%--------------------------------------------------------------------------
% Load the sequence of the asterisk (its motion is easy and well-known)
load('./dvs_circles.mat', 'x', 'y', 'ts', 'pol');

% x_selected_ = x(1:1e4); y_selected_ = y(1:1e4);
% ts_selected_ = ts(1:1e4); pol_selected_ = pol(1:1e4);
x_selected_ = x; y_selected_ = y;
ts_selected_ = ts; pol_selected_ = pol;

NCOLS = 340; NROWS = 340;
%--------------------------------------------------------------------------

flow_pathname='./results/flow';
addpath(genpath('./toolbox'));

%step_size = 4000;
% step_size = 20940;
step_size = 20940; % reading all the events for this sequence; in normal conditions, read less than that.
% step_size = 1e4-10;
step_size_small = 1000;

%curr_event = 6500; num_frame = 1;
curr_event = 1; num_frame = 1; % start from the first event

N = 3; TH1 = 0.99; TH2 = 1e-3; % for artificial seqs
% N is the size (2*N+1 x 2*N+1) of the neighborhood for estimating local normals
% TH1 is a threshold for locality of timestamp (see computeFlow())
% TH2 is not used in this version (in order to speed up processing), it is used for refining estimation
%N = 5; TH1 = 0.2;  TH2 = 0.1; % for real-world seqs

while (curr_event + step_size) < numel(ts_selected_)
    x = x_selected_(curr_event:curr_event+step_size); %TODO: x and y are switched!
    y = y_selected_(curr_event:curr_event+step_size);
    t = double(ts_selected_(curr_event:curr_event+step_size));
    pol = pol_selected_(curr_event:curr_event+step_size);
    
    t = t-t(1); % local times for each chunk of data
    [vx_tmp, vy_tmp, It_tmp] = computeFlow(x, y, t, pol, N, TH1, TH2, NCOLS, NROWS);
    
    % Formatting resutls for presenting the flow
    It = It_tmp; 
    It(It<(t(end)/2))=0; % show flow only in positions where events happened in the last haft part of the sequence
    mask=(It~=0); vx_tmp = vx_tmp.*mask; vy_tmp = vy_tmp.*mask;
    
    vx = medfilt2(vx_tmp); vy = medfilt2(vy_tmp); % use median filter to normalize results
    
    % Show the flow
    h=figure(1);
    set (h, 'Units', 'pixels', 'Position', [20,20,240*6,180*6]);
    imagesc(flipud(It_tmp)), hold on, axis off, axis equal, quiver(flipud(vx),flipud(-vy), 3, 'color', [1 0 0])   
    drawnow;

%     F = getframe(gcf);
%     [X, Map] = frame2im(F);
% 
%     imwrite(X, fullfile(flow_pathname, 'seq_2', strcat('frame', sprintf('_%05d', num_frame),'.png')));
%     save(fullfile(flow_pathname, 'seq_2', strcat('frame', sprintf('_%05d', num_frame),'.mat')), 'vx', 'vy');
     
    curr_event = curr_event + step_size_small;
    num_frame = num_frame +1;
    %close all
end

%This was for the search of 0 of the normal flow and the fittings

% % new_vx = vx_tmp; new_vy = vy_tmp;
% new_vx = vx; new_vy = vy;
% new_vx(:,1:169)=0; new_vy(:,1:169)=0; %use only the circle on the right side
% new_vx(:,1:169)=0; new_vy(:,1:169)=0; %use only the circle on the right side
% 
% % Now, just take a look at the data
% mod = sqrt(new_vx.^2+new_vy.^2);
% ang = atan2(new_vy,new_vx);
% 
% % Select only ang that correspond to the newest events
% perim = bwmorph(imdilate(~(It<(t(end)*0.8)),strel('disk',2)),'thin',Inf);
% perim(:,1:169)=0;
% 
% perim_ang = perim.*ang; 
% figure, imagesc(mod),title('flow value'),axis image
% figure, imagesc(ang),title('direction'),axis image
% figure, imagesc((perim==0).*mod),axis image
% %figure, plot(perim_ang(perim~=0), mod(perim~=0),'.')
% 
% [B,I]=sort(perim_ang(perim~=0));
% kk = mod(perim~=0);
% C = kk(I);
% figure, plot(B,C,'.')
% 
% % select only one half
% perim_half = zeros(size(perim));
% perim_half(150:end,:)=perim(150:end,:);
% % This is for removing the second sinusoid
% perim_half(perim_ang<0.1)=0;
% % figure, plot(perim_ang(perim_half~=0), mod(perim_half~=0),'.')
% 
% [Xdata,I2]=sort(perim_ang(perim_half~=0));
% kk2 = mod(perim_half~=0);
% Ydata = kk2(I2);
% figure, plot(Xdata,Ydata,'.'), hold on
% 
% foo = fit(Xdata,Ydata,'sin1'); % function type: a1*sin(b1*x+c1)
% % Checking foo
% plot(Xdata,foo.a1*sin(foo.b1*Xdata+foo.c1),'.');


% Now, compute the solution for the linear system in a lsq manner
%keyboard
vx = vx/25; vy = vy/25; % This is the flow between consecutive frames
                            % We assumed 25 fps, when we simulate them
SCENE_MAX_pix = [170 170 10];
SCENE_MIN_pix = [-170 -170 1];
[X_pix, Y_pix] = meshgrid(SCENE_MIN_pix(1)+1/2:1:SCENE_MAX_pix(1), SCENE_MIN_pix(2)+1/2:1:SCENE_MAX_pix(2));
Y_pix = flipud(Y_pix);
mask = (It>0); % focus only in positions where we have events

It(It<(t(end)*0.9))=0;

% vx = vx_tmp.*(It~=0); vy = vy_tmp.*(It~=0);
vx = vx.*(It~=0); vy = vy.*(It~=0);


un = sqrt(vx.*vx+vy.*vy);
nx = vx./un; ny = vy./un; 
nx(un==0)=0; ny(un==0)=0;
d = X_pix.*nx - Y_pix.*ny;

% Prepare data (diff columns means diff unknown)
C = [nx(mask), -ny(mask), un(mask)];
d = d(mask);
tic
x = lsqlin(C,d);
toc
numel(nx(mask))
x

