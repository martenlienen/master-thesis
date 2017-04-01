% computeFlow
%   x       - list of x positions of events
%   y       - list of y positions of events
%   t       - list of timestamps of events
%   pol     - list of polarities of events
%   N       - Size of neighborhood (2*N+1, 2*N+1) for local 3D normal estimation
%   TH1     - Threshold for selecting events that happen close enough in time in microseconds
%   TH2     - Threshold for refining 3D normal estimation (not used in this version).
%   NCOLS   - Number of columns of image
%   NROWS   - Number of rows of image
%
%
% RETURN
%   vx 		- x component of motion flow field (in image NCOLSxNROWS coordinates)
%   vy 		- y component of motion flow field (in image NCOLSxNROWS coordinates)
%   It 		- Timestamp of the motion flow field (in image NCOLSxNROWS coordinates)
%
% DESCRIPTION
%   The function computes the local flow estimation for a list of events.
%   This function makes use of fitplane() to compute the normal. The output
%   returns the X and Y components of the flow and the estimate timestamp in the
%   form of a frame, for the visualization of the data. Outputs are given
%   in pix/s and s.
%
%   The value for the params in normal conditions for real-world sequences
%   could be: N=5; TH1 = ?; TH2 = 0.1;
%
%   Copyright (C) 2015  Francisco Barranco, 10/10/2016, Universidad de Granada.
%   License, GNU GPL, free software, without any warranty.
%
function [vx, vy, It] = computeFlow(x, y, t, pol, N, TH1, TH2, NCOLS, NROWS)

    % Initialization
    It_pos=zeros(NROWS,NCOLS); % Matrix of timestamps of last event
    It_neg=zeros(NROWS,NCOLS); % Matrix of timestamps of last event
    vx=zeros(NROWS,NCOLS); vy=zeros(NROWS,NCOLS);

    for ii=1:1:length(t)
       ptx=x(ii)+1;
       pty=y(ii)+1;

       % Separate estimations for positive and negative events
       if pol(ii)==1
            It_pos(pty,ptx)=t(ii); %update timestamp of last event for x,y position
            m=It_pos(max(pty-N,1):min(pty+N, NROWS),max(ptx-N,1):min(ptx+N, NCOLS)); % get local timestamp patch around the event position
       else
            It_neg(pty,ptx)=t(ii);
            m=It_neg(max(pty-N,1):min(pty+N, NROWS),max(ptx-N,1):min(ptx+N, NCOLS));
       end

       if numel(m) == (2*N+1)*(2*N+1) % discarding events close to the image boundary

           % select events that happened close enough in time (TH1)
           m(abs(m(N+1,N+1)-m)>TH1)=0;

           % if there are any events
           if (sum(m(:)>0))
               %compute local 3D normal using fitplane function
               [vvx,vvy]=fitplane(m, TH2);

               % remove div by 0 values
               if isnan(vvx) || isinf(vvx), vvx = 0; end;
               if isnan(vvy) || isinf(vvy), vvy = 0; end;

               % write estimate in vx,vy output
               if (norm([vvx,vvy])>0)
                    vy(pty,ptx)=vvx;
                    vx(pty,ptx)=vvy;
               end
           end
       end
    end

   % Return only one time map
   It = max(cat(3, It_pos, It_neg), [], 3);
end
