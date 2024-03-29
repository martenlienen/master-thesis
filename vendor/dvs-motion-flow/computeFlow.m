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
%   toffset - Offset of first event for which a flow field should be computed
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
function [vx, vy, It] = computeFlow(x, y, t, pol, N, TH1, TH2, NCOLS, NROWS, toffset)

    if (nargin < 10)
        toffset = 1;
    end

    % Initialization
    It_pos=zeros(NROWS,NCOLS); % Matrix of timestamps of last event
    It_neg=zeros(NROWS,NCOLS); % Matrix of timestamps of last event
    vx=zeros(NROWS,NCOLS); vy=zeros(NROWS,NCOLS);

    jj = 1;
    for ii=toffset:1:length(t)
       % Register all events that were sent previously or simultaneously
       while jj < length(t) && t(jj) <= t(ii)
          ptx=x(jj)+1;
          pty=y(jj)+1;

          % Separate estimations for positive and negative events
          if pol(jj)==1
             %update timestamp of last event for x,y position
             It_pos(pty,ptx)=t(jj);
          else
             It_neg(pty,ptx)=t(jj);
          end

          jj = jj + 1;
       end

       ptx=x(ii)+1;
       pty=y(ii)+1;

       % Separate estimations for positive and negative events
       if pol(ii)==1
            m=It_pos(max(pty-N,1):min(pty+N, NROWS),max(ptx-N,1):min(ptx+N, NCOLS)); % get local timestamp patch around the event position
       else
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
