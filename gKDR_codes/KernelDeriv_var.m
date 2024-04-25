%  function KernelDeriv_var()
%
%  Computing effective directions for regression with RKHS.

% KernelDeriv_var()
%
% Arguments
%  X:  explanatory variables (input data)
%  Y:  response variables (teaching data)
%  K:  dimension of effective subspaces
%  SIGX:  sigma parameter for Gaussian kernel
%  SIGY:  sigma parameter for Gaussian kernel 
%  EPS:  regularization parameter
%
% Return value(s)
%  B:  orthonormal column vectors (M x K)
%  t:  sum of eigenvalues 
%
% Description
%  This program computes the projection matrix by gradient0-based KDR method
%  The incomplete Gholesky approximation for G_Y is used for reducing the
%  required memory.
% 
%-----------------------------------------------

%X=rand(100, 5);
%Y=rand(100,1);
%K=2;
%SGX=5;
%SGY=5.1;
%EPS=0.0001;
%NDIV=50;
%KernelDeriv_var(X,Y,K,SGX,SGY,EPS, NDIV)


function B=KernelDeriv_var(X,Y,K,SGX,SGY,EPS,NDIV)

[N,M]=size(X);  % N: data size, M: dim of X.


tol=0.000001;   % tolerance for incomplete cholesky approximation


ridx=randperm(N);  
X=X(ridx,:);
Y=Y(ridx,:);

sx2=2*SGX*SGX;

% Gram matrix of X
ab=X*X';
aa=diag(ab);
D=repmat(aa,1,N);
xx=max(D + D' - 2*ab, zeros(N,N));
Kx=exp(-xx./sx2);  

% incomplete cholesky approximation of Ky
[G, Pvec] = chol_inc_gauss(Y',SGY,tol);
[a,Pvec]=sort(Pvec); 
Ry=G(Pvec,:);
r=length(Ry(1,:));
Ty=Ry'/(Kx+(N*EPS).*eye(N));
clear Ry;

% random partition of data
lx=floor(N/NDIV);
ei=cumsum(lx.*ones(1,NDIV),2);
si=ei-(lx-1).*ones(1,NDIV);
ei(NDIV)=N;       % si: staring idx, ei: ending idx


Proj=zeros(M,M);
for i=1:NDIV
    % Derivative of k(X_i, x) w.r.t. x
    pi=si(i):ei(i);
    Xp=X(pi,:);
    Lp=length(pi);
    Xia=reshape(repmat(Xp,N,1),Lp,N,M);
    Xja=reshape(repmat(X,Lp,1),N,Lp,M);
    Xia=permute(Xia,[2 1 3]);
    Xij=(Xia-Xja)./SGX/SGX;     % N x Lp x M
    H=Xij.*repmat(Kx(:,pi),[1 1 M]);
    clear Xij Xja Xia;

    % compute the matrix for gKDR
    Hy=reshape(Ty*reshape(H,[N,Lp*M]), [r*Lp,M]);
    R=Hy'*Hy;
    clear Hy;


    % compute the first K eigenvectors
    [B L]=eigs(R,K);
    Proj=Proj+B*B';
end

[B L]=eigs(Proj,K);






    



