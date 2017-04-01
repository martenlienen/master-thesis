%--------------------------------------------------------------------------
% Load the sequence of the asterisk (its motion is easy and well-known)
load('~/WORK/flow/Simulator/dvs_asterisk.mat', 'x', 'y', 'ts', 'pol');

x_selected_ = x; y_selected_ = y;
ts_selected_ = ts; pol_selected_ = pol;
NCOLS = 450; NROWS = 600;
%--------------------------------------------------------------------------



% Apply first the background filter to the data
% NCOLS = 180; NROWS = 240;
x_selected_ = x_selected_ +1; % 1 to NCOLS
y_selected_ = y_selected_ +1; % 1 to NROWS


%[x_filt, y_filt, pol_filt, t_filt]= activity_filter_final(x_selected_, ...
%    y_selected_, pol_selected_, double(ts_selected_), NCOLS, NROWS, 3, 5e3);

%The activity filter here is not required
x_filt = x_selected_; y_filt = y_selected_;
t_filt = ts_selected_; pol_filt = pol_selected_;


x_selected_ = x_selected_ - 1; % 0 to NCOLS-1
y_selected_ = y_selected_ - 1; % 0 to NROWS-1


flow_pathname='./results/flow';
addpath(genpath('./toolbox'));

%step_size = 4000;
step_size = 364000;
step_size_small = 1000;

%curr_event = 6500; num_frame = 1;
curr_event = 1; num_frame = 1;
N = 3; TH1 = 0.99; TH2 = 1e-3; % for artificial seqs
%N = 5; TH1 = 0.2;  TH2 = 0.1; % for real-world seqs

while (curr_event + step_size) < numel(t_filt)
    x = x_filt(curr_event:curr_event+step_size); %TODO: x and y are switched!
    y = y_filt(curr_event:curr_event+step_size);
    t = double(t_filt(curr_event:curr_event+step_size));
    pol = pol_filt(curr_event:curr_event+step_size);
    
    t = t-t(1);
    [vx_tmp, vy_tmp, It_tmp] = computeFlow(x, y, t, pol, N, TH1, TH2, NCOLS, NROWS);
    
    It = It_tmp;
    It(It<(t(end)/2))=0;
    mask=(It~=0); vx_tmp = vx_tmp.*mask; vy_tmp = vy_tmp.*mask;
    
    vx = medfilt2(vx_tmp); vy = medfilt2(vy_tmp);
    
    keyboard
    
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
    close all
end