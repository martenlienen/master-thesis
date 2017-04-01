function [x_filt, y_filt, pol_filt, t_filt]= activity_filter_final(x_in, y_in, pol_in, t_in, width, height, block_size, deltaT)
% This is an implementation for the backgound noise substraction activity
% For every new event that comes in, the function checks the activity 
% that happened in an area of 'block_size' size surrounding the event
% position. If there is no new activity (nothing within a difference of
% 'deltaT' time, the event is filtered out
% Input: 
%      x_in:        x positions of stream of events 
%      y_in:        y positions of stream of events
%      pol_in:      polarities of stream of events 
%      t_in:        timestamps of stream of events 
%      block_size:  size of the area checked for recent activity surrounding each event position
%      deltaT:      time threshold to determine if there was recent activity
% Output:
%      x_filt:      x positions of stream of filtered events 
%      y_filt:      y positions of stream of filtered events
%      pol_filt:    polarities of stream of filtered events 
%      t_filt:      timestamps of stream of filtered events  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % Examples of params
% block_size = 3;
% deltaT = 15000;


N = block_size;
img_time_pos = zeros(height, width);
img_time_neg = zeros(height, width);
x_filt=zeros(size(x_in)); y_filt=zeros(size(x_in)); 
pol_filt=zeros(size(x_in)); t_filt=zeros(size(x_in));

cnt = 1;

for ii=1:numel(x_in)   
    
    if pol_in(ii)==1
        mm_pos = t_in(ii) - ...
            img_time_pos(max(y_in(ii)-floor(N/2),1):min(y_in(ii)+floor(N/2), height), ...
            max(x_in(ii)-floor(N/2),1):min(x_in(ii)+floor(N/2), width));
              
        if min(abs(mm_pos(:))) < deltaT
            x_filt(cnt)=x_in(ii);
            y_filt(cnt)=y_in(ii);
            pol_filt(cnt)=pol_in(ii);
            t_filt(cnt)=t_in(ii);
            cnt = cnt + 1;
        end
            
        img_time_pos(y_in(ii),x_in(ii))=t_in(ii);
    else
        mm_neg = t_in(ii) - ...
            img_time_neg(max(y_in(ii)-floor(N/2),1):min(y_in(ii)+floor(N/2), height), ...
            max(x_in(ii)-floor(N/2),1):min(x_in(ii)+floor(N/2), width));
        
        if min(abs(mm_neg(:))) < deltaT
            x_filt(cnt)=x_in(ii);
            y_filt(cnt)=y_in(ii);
            pol_filt(cnt)=pol_in(ii);
            t_filt(cnt)=t_in(ii);
            cnt = cnt + 1;
        end
    
        img_time_neg(y_in(ii),x_in(ii))=t_in(ii);
    end
end

% Get rid of extra positions
x_filt(cnt:end)=[]; y_filt(cnt:end)=[];
pol_filt(cnt:end)=[]; t_filt(cnt:end)=[];
