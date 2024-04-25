%  function KernelDeriv()
%
%  Computing effective directions for regression with RKHS
%
% Arguments
%  X:  explanatory variables (input data)
%  Y:  response variables (teaching data)
%  K:  dimension of effective subspaces
%  SGX:  bandwidth (deviation) parameter in Gaussian kernel for X
%  SGY:  bandwidth (deviation) parameter in Gaussian kernel for Y
%  EPS:  regularization coefficient
%
% Return value(s)
%  B:  orthonormal column vectors (M x K)
%  t:  value of the objective function
%
% 
%-----------------------------------------------

% We test the function but remember to add end at the end of the function
%X=rand(100, 5);
%Y=rand(100,1);
%K=1;
%SGX=5;
%SGY=5.1;
%EPS=0.0001;
%KernelDeriv(X,Y,K,SGX,SGY,EPS)




function [B, t]=KernelDeriv(X,Y,K,SGX,SGY,EPS)

[N,M]=size(X);  % N: data size, M: dim of X.

I=eye(N);

sx2=2*SGX*SGX;
sy2=2*SGY*SGY;

% Gram matrix of X
ab=X*X';
aa=diag(ab);
D=repmat(aa,1,N);
xx=max(D + D' - 2*ab, zeros(N,N));
Kx=exp(-xx./sx2);  

% Gram matrix of Y
ab=Y*Y';
aa=diag(ab);
D=repmat(aa,1,N);
yy=max(D + D' - 2*ab, zeros(N,N));
Ky=exp(-yy./sy2);  

% Derivative of k(X_i, x) w.r.t. x
Dx=reshape(repmat(X,N,1),N,N,M);
Xij=Dx-permute(Dx,[2 1 3]);
Xij=Xij./SGX/SGX;
H=Xij.*repmat(Kx,[1 1 M]);

% compute  sum_i H(X_i)'*Kx^-1*Ky*Kx^-1*H(X_i)
F=((Kx+N*EPS.*I)\Ky)/(Kx+N*EPS.*I);
Hm=reshape(H,N,N*M);
HH=reshape(Hm'*Hm,[N,M,N,M]);
HHm=reshape(permute(HH,[1 3 2 4]), N*N,M,M);
Fm=repmat(reshape(F,N*N,1),[1,M,M]);
R=reshape(sum(HHm.*Fm,1),[M,M]);

[V,L]=eig(R);
[e,idx]=sort(diag(L),'descend');
B=V(:,idx(1:K));
t=sum(e(idx(1:K)));



