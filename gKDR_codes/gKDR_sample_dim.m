%  gKDR_sample
% 
%  Gradient-based kernel dimension redeuction (gKDR) sample program
%  This is a matlab code for gKDR, gKDR-i, gKDR-v.  
%  For the details of these algorithms, see the follwoing paper
%   Gradient-based kernel dimension reduction for regression
%     by Kenji Fukumizu and Chenlei Leng
%
clear all;
close all;

tic
VERBOSE = 0;    % print the optimization process

%Method:  choose one of them. 
Method='gKDR';
%Method='gKDR-i';
%Method='gKDR-v';

% 
K=5;       % Dimension of Effective directions

% Data specification
M=10;   % dimensionality of X
N=120;  % sample size
AC=0.0; % parameter of the regressor 

NCV=5;      % Number of cross-validation
DEC=10;     % Number of decreasing dimensions for gKDR-i;LOOP

MLOOP=1;

B0=zeros(M,1);
B0(1,1)=1;      % B: Projection matrix onto the effective direction

candx=[0.25 0.5 0.75 1 2];  % candidates for CV
candy = [0.25 0.5 0.75 1 2];
eps = [0.00001];


fprintf('\ngKDR_sample:\n');
fprintf('Dataset C\n');
fprintf('#sample = %d, dim of X = %d, effective dim = %d\n\n', N, M, K);

err=zeros(MLOOP,1);

for loop=1:MLOOP
    a=clock;
    seed=floor(a(6));
    s = RandStream('mt19937ar','Seed',seed);
    RandStream.setGlobalStream(s);
    
    % Data: X = N(0,1/4) restricted on [-1 1]
    %aa=zeros(N,M);
    %X=zeros(N,M);
    %while nnz(aa)<N*M
        %tmpX=randn(N,M);
        %X=X.*aa + tmpX.*(1-aa);
        %aa=( abs(X) <= 2);
    %end
    %X=X./2;
    %Z=X*B0;    
    %Y= (Z(:,1)-AC).^4 .* randn(N,1);    % Y = (X_1 - AC)^4 * N(0,1)

    readData = readtable('DatatrainN2.csv');
    DataFile=table2array(readData);

    Y=DataFile(:, [11]);
    X=DataFile(:, [1:10]);
 
    % Gaussian kernels are used.  Deviation parameter are chosen by CV with kNN
    % classifier. 

    sgx0=MedianDist(X);   % Basic value for bandwidth for X
    sgy0=MedianDist(Y);   % Basic value for bandwidth for Y

    % For cross-validation
    ridx=randperm(N);  % random order 
    Xr=X(ridx,:);
    Yr=Y(ridx,:);   
    lx=ceil(N/NCV);
    ei=cumsum(lx.*ones(1,NCV),2);
    si=ei-(lx-1).*ones(1,NCV);
    ei(NCV)=N;       % si: staring idx, ei: ending idx
    err_tbl=zeros(length(candx)*length(candy)*length(eps), NCV);
    
    fprintf('Cross validation ...');
    
    for h=1:length(candx)
        sgx=sgx0*candx(h);
        for k=1:length(candy)
          sgy=sgy0*candy(k);            
          for ll=1:length(eps)
            EPS = eps(ll);
            for i=1:NCV
                ri=si(i):ei(i);
                Xe=Xr; Ye=Yr; 
                Xe(ri,:)=[];
                Ye(ri,:)=[];    % Xe, Ye: trainig sample for CV
                Xt=Xr(ri,:);
                Yt=Yr(ri,:);    % Xt, Yt: test sample for CV
                switch Method
                    case 'gKDR'
                        [B, t]=KernelDeriv(Xe,Ye,K,sgx,sgy,EPS);
                    case 'gKDR-i'
                        if M-K <=DEC
                            decd=ones(M-K,1);
                        else
                            dd=floor((M-K)/DEC);
                            r=M-K-dd*DEC;
                            decd=dd*ones(DEC,1);
                            decd(1:r,1)=(dd+1)*ones(r,1);
                        end
                        B=eye(M);
                        for ii=1:length(decd)
                            Ze=Xe*B;
                            Bp=B;
                            dim=M-sum(decd(1:ii,1),1);
                            [B, t]=KernelDeriv(Ze,Ye,dim,sgx*sqrt(dim/M),sgy,EPS);
                            B=Bp*B;
                            B=B/sqrtm(B'*B);  
                        end
                    case 'gKDR-v'
                        B=KernelDeriv_var(Xe,Ye,K,sgx,sgy,EPS,50);
                    otherwise
                        error('Error: method mismatch');
                end
                % kNN regression for CV
                nnidx=knnsearch(Xe*B,Xt*B, 'K', 5, 'NSMethod', 'kdtree');
    
                Yo=zeros(length(ri),length(Y(1,:)));
                for j=1:length(ri)
                    ii=nnidx(j,:);
                    Yo(j,:)=sum(Ye(ii',:),1)./5;
                end
    
                dd=Yt-Yo;      
                err_tbl((h-1)*length(candy)*length(eps)+(k-1)*length(eps)+ll,i)=sum(sum(dd.*dd,1),2)./length(ri);  
        
            end
          end
        end
        if VERBOSE    
            fprintf('.');
            drawnow;
        end
    
    end
    if VERBOSE
        fprintf('\n');
    end

    [c, midx]=min(mean(err_tbl,2));
    opth=ceil(midx/(length(candy)*length(eps)));
    rr=midx-(opth-1)*length(candy)*length(eps);
    optk=ceil(rr/length(eps));
    opte=mod(rr,length(eps));
    if opte==0
        opte=length(eps);
    end

    if VERBOSE
        fprintf('coef_x = %.2f  coef_y = %.2f  eps=%.10f err=%f\n', candx(opth),candy(optk), eps(opte),c);  
    end
    
    fprintf('done\n');

    % Parameter
    sgx=sgx0*candx(opth);
    sgy=sgy0*candy(optk);
    EPS=eps(opte);
    switch Method
        case 'gKDR'
            [B, t]=KernelDeriv(X,Y,K,sgx,sgy,EPS);
        case 'gKDR-i'
            B=eye(M);
            for ii=1:length(decd)
                Z=X*B;
                Bp=B;
                dim=M-sum(decd(1:ii,1),1);
                [B, t]=KernelDeriv(Z,Y,dim,sgx*sqrt(dim/M),sgy,EPS);
                B=Bp*B;
                B=B/sqrtm(B'*B);  
            end
        case 'gKDR-v'           
            B=KernelDeriv_var(X,Y,K,sgx,sgy,EPS,50);
    end
    err(loop)=sqrt(trace(B0*B0'*(eye(M)-B*B'))/trace(B0'*B0));
    if VERBOSE
        fprintf('[%d] %f\n', loop, err(loop));  
    end
    
end


fprintf('NCV=%d, DEC=%d, K=%d\n', NCV, DEC);
fprintf('sgx = %.3f  sgy = %.3f eps=%e\n', sgx,sgy,EPS);

fprintf('\nExtimation Error: %f\n\n', mean(err));

fprintf('\nExtimation Error standard deviation: %f\n\n', std(err))

errors_e = table(err);

Estimate_projection_matrix= table(B);

% Specify the file name
PC1 = 'project6Ngv.csv';

% Save the data as a CSV file
writematrix(B, PC1)
toc

