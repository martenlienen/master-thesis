% % Load the sequence
% load('selectedData', 'pol_selected_', 'x_selected_', 'y_selected_', 'ts_selected_');
% flow_pathname='./results/flow';
% addpath(genpath('./toolbox'));
% 
% 
% step_size = 4000;
% step_size_small = 2000;
% % t = double(ts_selected);
% % y = x_selected; % TODO: x is y and y is x; this is a bug, fix it!
% % x = y_selected;
% 
% curr_event = 15000; num_frame = 1;
% h=figure(1);
% set (h, 'Units', 'normalized', 'Position', [0,0,1,1]);
% 
% while (curr_event + step_size) < numel(ts_selected_)
%     y = x_selected_(curr_event:curr_event+step_size);
%     x = y_selected_(curr_event:curr_event+step_size);
%     t = double(ts_selected_(curr_event:curr_event+step_size));
%     t = t-t(1);
%     
%     
%     [vx_tmp, vy_tmp, It_tmp] = computeFlow(x, y, t);
%     
%     vx = medfilt2(vx_tmp);
%     vy = medfilt2(vy_tmp);
%     It = It_tmp;
%     It(It<(t(end)/2))=0;
%     mask=(It~=0); vx = vx.*mask; vy = vy.*mask;
%     
%     h=figure(1);
%     set (h, 'Units', 'normalized', 'Position', [0,0,1,1]);
%     imagesc(flipud(It_tmp)), hold on, axis off, axis equal, quiver(flipud(vx),flipud(-vy), 2, 'color', [1 0 0])
%     
%     drawnow;
%     F = getframe(gcf);
%     [X, Map] = frame2im(F);
% 
%     imwrite(X, fullfile(flow_pathname, 'seq_1', strcat('frame', sprintf('_%05d', num_frame),'.png')));
%     
%     curr_event = curr_event + step_size_small;
%     num_frame = num_frame +1;
%     close all
% end


% Load the sequence
% load('selectedData', 'pol_selected_', 'x_selected_', 'y_selected_', 'ts_selected_');

[x_selected_,y_selected_,pol_selected_,ts_selected_] = getDVSeventsDavis('./seq_2.aedat', 10e6); 
%num_events = numel(allTs);
%[x,y,pol]=extractRetinaEventsFromAddr(allAddr);

flow_pathname='./results/flow';
addpath(genpath('./toolbox'));


step_size = 4000;
step_size_small = 1000;
% t = double(ts_selected);
% y = x_selected; % TODO: x is y and y is x; this is a bug, fix it!
% x = y_selected;

curr_event = 5000; num_frame = 1;
h=figure(1);
set (h, 'Units', 'pixels', 'Position', [20,20,240*6,180*6]);

while (curr_event + step_size) < numel(ts_selected_)/2
    y = x_selected_(curr_event:curr_event+step_size);
    x = y_selected_(curr_event:curr_event+step_size);
    t = double(ts_selected_(curr_event:curr_event+step_size));
    
    t = t-t(1);
    
    [vx_tmp, vy_tmp, It_tmp] = computeFlow_telluride(x, y, t);
    
    vx = medfilt2(vx_tmp);
    vy = medfilt2(vy_tmp);
    It = It_tmp;
%     keyboard
    It_tmp_old = It_tmp;
    It(It<(t(end)/2))=0;
    mask=(It~=0); vx = vx.*mask; vy = vy.*mask;
    
    h=figure(1);
    set (h, 'Units', 'pixels', 'Position', [20,20,240*6,180*6]);
    imagesc(flipud(It_tmp)), hold on, axis off, axis equal, quiver(flipud(vx),flipud(-vy), 3, 'color', [1 0 0])   
    drawnow;

    F = getframe(gcf);
    [X, Map] = frame2im(F);

    imwrite(X, fullfile(flow_pathname, 'seq_2', strcat('frame', sprintf('_%05d', num_frame),'.png')));
    save(fullfile(flow_pathname, 'seq_2', strcat('frame', sprintf('_%05d', num_frame),'.mat')), 'vx', 'vy');
    
    curr_event = curr_event + step_size_small;
    num_frame = num_frame +1;
    close all
end