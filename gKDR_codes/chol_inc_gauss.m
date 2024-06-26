function [G, Pvec] = chol_inc_gauss(x,sigma,tol)

% CHOL_INC_FUN - incomplete Cholesky decomposition of the Gram matrix defined
%                by data x, with the Gaussiab kernel with width sigma
%                Symmetric pivoting is used and the algorithms stops 
%                when the sum of the remaining pivots is less than TOL.
% 

% CHOL_INC returns returns an uPvecer triangular matrix G and a permutation 
% matrix P such that P'*A*P=G*G'.

% P is ONLY stored as a reordering vector PVEC such that 
%                    A(Pvec,Pvec)= G*G' 
% consequently, to find a matrix R such that A=R*R', you should do
% [a,Pvec]=sort(Pvec); R=G(Pvec,:);

% Copyright (c) Francis R. Bach, 2002.

n=size(x,2);
Pvec= 1:n;
I = [];

%calculates diagonal elements (all equal to 1 for gaussian kernels)
diagG=ones(n,1);
i=1;
G=[];

while ((sum(diagG(i:n))>tol)) 
   G=[G zeros(n,1)];
   % find best new element
   if i>1
      [diagmax,jast]=max(diagG(i:n));
      jast=jast+i-1;
      %updates permutation
      Pvec( [i jast] ) = Pvec( [jast i] );
      % updates all elements of G due to new permutation
      G([i jast],1:i)=G([ jast i],1:i);
      % do the cholesky update
      
      
   else
      jast=1;
   end
   
   
   
   G(i,i)=diagG(jast); %A(Pvec(i),Pvec(i));
   G(i,i)=sqrt(G(i,i));
   if (i<n)
      %calculates newAcol=A(Pvec((i+1):n),Pvec(i))
      newAcol = exp(-.5/sigma^2*sqdist(x(:, Pvec((i+1):n) ),x(:,Pvec(i))));
      if (i>1)
         G((i+1):n,i)=1/G(i,i)*( newAcol - G((i+1):n,1:(i-1))*(G(i,1:(i-1)))');
      else
         G((i+1):n,i)=1/G(i,i)*newAcol;
      end
      
   end
   
   % updates diagonal elements
   if (i<n) 
      diagG((i+1):n)=ones(n-i,1)-sum(   G((i+1):n,1:i).^2,2  );
   end
   i=i+1;
end





%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function d=sqdist(a,b)
% SQDIST - computes squared Euclidean distance matrix
%          computes a rectangular matrix of pairwise distances
% between points in A (given in columns) and points in B

% NB: very fast implementation taken from Roland Bunschoten

aa = sum(a.*a,1); bb = sum(b.*b,1); ab = a'*b; 
d = abs(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

