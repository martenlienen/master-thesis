% Load the sequence
%num_events = numel(allTs);
%[x,y,pol]=extractRetinaEventsFromAddr(allAddr);

%[x_selected_,y_selected_,pol_selected_,ts_selected_] = getDVSeventsDavis('./seq_1.aedat', 10e6); 
load('selectedData', 'pol_selected_', 'x_selected_', 'y_selected_', 'ts_selected_');

% Apply first the background filter to the data
NCOLS = 240; NROWS = 180;
x_selected_ = x_selected_ + 1; % 1 to NCOLS
y_selected_ = y_selected_ + 1; % 1 to NROWS


[x_filt, y_filt, pol_filt, t_filt]= activity_filter_final(x_selected_, ...
    y_selected_, pol_selected_, double(ts_selected_), NCOLS, NROWS, 3, 7e3);


x_selected_ = x_selected_ - 1; % 0 to NCOLS-1
y_selected_ = y_selected_ - 1; % 0 to NROWS-1


flow_pathname='./results/flow';
addpath(genpath('./toolbox'));

step_size = 4000;
step_size_small = 1000;

curr_event = 6500; num_frame = 1;
N = 7; TH = 0.2; TH2 = 0; %We are not using TH2 now

while (curr_event + step_size) < numel(t_filt)/2
    x = x_filt(curr_event:curr_event+step_size-1); %TODO: x and y are switched!
    y = y_filt(curr_event:curr_event+step_size-1);
    t = double(t_filt(curr_event:curr_event+step_size-1));
    pol = pol_filt(curr_event:curr_event+step_size-1);
    
    t = t-t(1);
    %[vx_tmp, vy_tmp, It_tmp] = computeFlow(x, y, t, pol, N, TH, NCOLS, NROWS);
    [vx_tmp, vy_tmp, It_tmp] = computeFlow(x, y, t, pol, N, TH, TH2, NCOLS, NROWS);
    
    keyboard
    
    It = It_tmp;
    It(It<(t(end)/2))=0;
    mask=(It~=0); vx_tmp = vx_tmp.*mask; vy_tmp = vy_tmp.*mask;
    
    vx = medfilt2(vx_tmp); vy = medfilt2(vy_tmp);
    
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