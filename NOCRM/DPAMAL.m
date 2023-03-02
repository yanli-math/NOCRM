function [W] = DPAMAL(fea,L,Y_old,alpha,beta,nu,maxIter)


[nSmp, nFea] = size(fea);
nClass = size(Y_old,2);
norminf=@(z) max(max(abs(z)));
lam_min=-1e2; lam_max=1e2;
c1=0.5; c2=c1; c3=c1;c4=c1;c5=c1;c6=c1;
rho=nClass/2;
In = eye(nSmp,nSmp);
Id=eye(nFea,nFea);

W_old= zeros(nFea,nClass);
V_old=zeros(nFea,nClass);
U_old=Y_old;
F_old=Y_old;
Yba_old=Y_old;
lambda1=zeros(nSmp,nClass);
lambda2=zeros(nFea,nClass);
lambda3=lambda1;
lambda4=lambda1;
lambda1_old =lambda1;
lambda2_old =lambda2;
lambda3_old =lambda3;
lambda4_old =lambda4;
R1_old = 0; R2_old=0; R3_old=0;R4_old=0;
for iter=1:maxIter
    num(iter)=1;
%update W
    W = (rho*fea'*fea+(2*nu+rho+c1)*Id)^(-1)*(fea'*lambda1_old+lambda2_old+rho*fea'*Y_old-rho*fea'*U_old+rho*V_old+c1*W_old);
%update U
    N=Y_old-fea*W+lambda1_old/rho;
    for i=1:nSmp
        U(i,:)=max(0,1-alpha/norm(rho*N(i,:)+c2*U_old(i,:),2))*(rho/(rho+c2)*N(i,:)+c2/(rho+c2)*U_old(i,:));
    end
%update V
    M=W-lambda2_old/rho;
    for i=1:nFea
        V(i,:)=max(0,1-beta/norm(rho*M(i,:)+c3*V_old(i,:),2))*(rho/(rho+c3)*M(i,:)+c3/(rho+c3)*V_old(i,:));
    end

%update Y
    Y=(2*L+(3*rho+c4)*In)^(-1)*(lambda4_old-lambda3_old-lambda1_old+rho*fea*W+rho*U+rho*F_old+rho*Yba_old+c4*Y_old);
    
%update F
    A=rho/(rho+c5)*(Y+lambda3_old/rho)+c5/(rho+c5)*F_old;
    for i=1:nSmp
        for j=1:nClass
            if A(i,j)<=1 & A(i,j)>=0
                F(i,j)=A(i,j);
            elseif A(i,j)>1
                F(i,j)=1;
            else
                F(i,j)=0;
            end
        end
    end
    
%update Yba
    B=rho/(rho+c6)*(Y-lambda4_old/rho)+c6/(rho+c6)*Yba_old;
    [U1,~,V1] = svd(B);
    Yba = U1*eye(nSmp,nClass)*V1';
    
theta1=rho*fea'*(Y_old-Y)+rho*fea'*(U-U_old)+rho*(V_old-V)+c1*(W_old-W);
theta2=rho*(Y_old-Y)+c2*(U_old-U);
theta3=c3*(V_old-V);
theta4=rho*(F_old-F)+rho*(Yba_old-Y)+c4*(Y_old-Y);
theta5=c5*(F_old-F);
theta6=c6*(Yba_old-Yba);
theta=[theta1',theta2',theta3',theta4',theta5',theta6']';
W_old=W; 
U_old=U;
V_old=V;
Y_old=Y;
F_old=F;
Yba_old=Yba;
while  norminf(theta)> max(1e-6, (0.995)^iter)   
        if num(iter)> 100
            break;
        end 
        
%update W
    W = (rho*fea'*fea+(2*nu+rho+c1)*Id)^(-1)*(fea'*lambda1_old+lambda2_old+rho*fea'*Y_old-rho*fea'*U_old+rho*V_old+c1*W_old);
%update U
    N=Y_old-fea*W+lambda1_old/rho;
    for i=1:nSmp
        U(i,:)=max(0,1-alpha/norm(rho*N(i,:)+c2*U_old(i,:),2))*(rho/(rho+c2)*N(i,:)+c2/(rho+c2)*U_old(i,:));
    end
%update V
    M=W-lambda2_old/rho;
    for i=1:nFea
        V(i,:)=max(0,1-beta/norm(rho*M(i,:)+c3*V_old(i,:),2))*(rho/(rho+c3)*M(i,:)+c3/(rho+c3)*V_old(i,:));
    end

%update Y
    Y=(2*L+(3*rho+c4)*In)^(-1)*(lambda4_old-lambda3_old-lambda1_old+rho*fea*W+rho*U+rho*F_old+rho*Yba_old+c4*Y_old);
    
%update F
    A=rho/(rho+c5)*(Y+lambda3_old/rho)+c5/(rho+c5)*F_old;
    for i=1:nSmp
        for j=1:nClass
            if A(i,j)<=1 & A(i,j)>=0
                F(i,j)=A(i,j);
            elseif A(i,j)>1
                F(i,j)=1;
            else
                F(i,j)=0;
            end
        end
    end
    
%update Yba
    B=rho/(rho+c6)*(Y-lambda4_old/rho)+c6/(rho+c6)*Yba_old;
    [U1,~,V1] = svd(B);
    Yba = U1*eye(nSmp,nClass)*V1';
    
theta1=rho*fea'*(Y_old-Y)+rho*fea'*(U-U_old)+rho*(V_old-V)+c1*(W_old-W);
theta2=rho*(Y_old-Y)+c2*(U_old-U);
theta3=c3*(V_old-V);
theta4=rho*(F_old-F)+rho*(Yba_old-Y)+c4*(Y_old-Y);
theta5=c5*(F_old-F);
theta6=c6*(Yba_old-Yba);
theta=[theta1',theta2',theta3',theta4',theta5',theta6']';
W_old=W; 
U_old=U;
V_old=V;
Y_old=Y;
F_old=F;
Yba_old=Yba;
num(iter)=num(iter)+1;
end


lambda1=lambda1_old+rho*(Y-fea*W-U);
lambda2=lambda2_old+rho*(V-W);
lambda3=lambda3_old+rho*(Y-F);
lambda4=lambda4_old+rho*(Yba-Y);
lambda1_old= max(lam_min,lambda1);  lambda1_old= min(lam_max,lambda1_old);
lambda2_old= max(lam_min,lambda2);  lambda2_old= min(lam_max,lambda2_old);
lambda3_old= max(lam_min,lambda3);  lambda3_old= min(lam_max,lambda3_old);
lambda4_old= max(lam_min,lambda4);  lambda4_old= min(lam_max,lambda4_old);
R1=Y-fea*W-U; R2=V-W;R3=Y-F;R4=Yba-Y;
if norminf(R1)>norminf(R1_old)*0.99 || norminf(R2)>norminf(R2_old)*0.99 || norminf(R3)>norminf(R3_old)*0.99 || norminf(R4)>norminf(R4_old)*0.99
    rho=1.01*rho;
end
R1_old=R1; R2_old=R2; R3_old=R3; R4_old=R4;
end
        
    
    
    
end
