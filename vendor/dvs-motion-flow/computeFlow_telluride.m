function [vx, vy, It] = computeFlow_telluride(x, y, t)

It=zeros(180,240);
vx=zeros(180,240); vy=zeros(180,240);

N=7;

for i=1:1:length(t) 
   ptx=x(i)+1;
   pty=y(i)+1;
   
   It(ptx,pty)=t(i);

   if (ptx>N+1 && ptx<180-(N+1))
      if (pty>N+1 && pty<240-(N+1))

          m=It(max(ptx-N,1):min(ptx+N, NCOLS),max(pty-N,1):min(pty+N, NROWS));
           
           m(abs(m(N+1,N+1)-m)/m(N+1,N+1)>.2)=0;
%            m(abs(m(N+1,N+1)-m)>400)=0;
                     
            if (sum(m(:)>0))
                   [vvx,vvy]=fitplane(m);

                   vvx(isnan(vvx))= 0;
                   vvy(isnan(vvy)) = 0;
                   vvx(isinf(vvx)) = 0;
                   vvy(isinf(vvy)) = 0;

                   if (norm([vvx,vvy])>0)
                        aa=[vvx(N+1,N+1) vvy(N+1,N+1)];
                        vx(ptx,pty)=aa(1)/norm(aa);
                        vy(ptx,pty)=aa(2)/norm(aa);
%                         vx(ptx,pty)=aa(1);
%                         vy(ptx,pty)=aa(2);
                   end
            end
      end
   end
end
           
           
           
       
   
   
   
