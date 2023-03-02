function f = DPAMAL_single(dataset)
%run DPAMAL

%======================setup=======================
nKmeans = 20;
FeaNumCandi = [];

alfaCandi = [10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3,10^4,10^5,10^6];
betaCandi = [10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3,10^4,10^5,10^6];
nuCandi = [10^-6,10^-5,10^-4,10^-3,10^-2,10^-1,1,10,10^2,10^3,10^4,10^5,10^6];

maxIter = 20;
%==================================================

load('9_Tumors.mat');
fea=ins;
gnd=lab;
if min(gnd) ~=1
    gnd = gnd + (1-min(gnd));
end
[nSmp,nFea] = size(fea);
if nFea>=300
    FeaNumCandi = [50,100,150,200,250,300];
else
    FeaNumCandi = [50,80,110,140,170,200];
end

bestNMI_max = zeros(length(FeaNumCandi),1);
bestNMI_sqrt = zeros(length(FeaNumCandi),1);
bestACC = zeros(length(FeaNumCandi),1);

nClass = length(unique(gnd));

%print the setup information
disp(['Dataset: ',dataset]);
disp(['class_num=',num2str(nClass),',','num_kmeans=',num2str(nKmeans)]);

t_start = clock;

%construct the affinity matrix
disp(['construct the affinity matrix...']);
S = constructW(fea); 


disp(['get Laplacian matrix...']);


diag_ele_arr = sum(S,2);
D = diag(diag_ele_arr);
L = full(D)^(-1/2)*(D-S)*full(D)^(-1/2);

eY = eigY(L,nClass);
rand('twister',5489);
label = litekmeans(eY,nClass,'Replicates',20);


%Clustering using selected features
for alpha = alfaCandi
    for beta = betaCandi
        for nu = nuCandi
            Y=eye(size(fea,1),nClass);
            disp(['alpha=',num2str(alpha),',','beta=',num2str(beta),',','nu=',num2str(nu)]);
            result_path = strcat('newresult_Tumors9','/','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'_result.mat');
            mtrResult = [];
            W = DPAMAL(fea,L,Y,alpha,beta,nu,maxIter);

            [dumb idx] = sort(sum(W.*W,2),'descend');
            orderFeature_path = strcat(dataset,'\','feaIdx_','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'.mat');
            save(orderFeature_path,'idx');
            
            for feaIdx = 1:length(FeaNumCandi)
                feaNum = FeaNumCandi(feaIdx);
                newfea = fea(:,idx(1:feaNum));
                rand('twister',5489);
                %arrNMI_max = zeros(nKmeans,1);
                arrNMI_sqrt = zeros(nKmeans,1);
                arrACC = zeros(nKmeans,1);
                for i = 1:nKmeans
                    label = litekmeans(newfea,nClass,'Replicates',1);
                
                    arrNMI_sqrt(i) = NMI_sqrt(gnd,label);
                    arrACC(i) = ACC(gnd,label);
                end
                mNMI_sqrt = mean(arrNMI_sqrt);
                sNMI_sqrt = std(arrNMI_sqrt);
                mACC = mean(arrACC);
                sACC = std(arrACC);
                if mNMI_sqrt>bestNMI_sqrt(feaIdx)
                    bestNMI_sqrt(feaIdx) = mNMI_sqrt;
                end
                if mACC > bestACC(feaIdx)
                    bestACC(feaIdx) = mACC;
                end
                mtrResult = [mtrResult,[feaNum,mNMI_sqrt,sNMI_sqrt,mACC,sACC]'];
            end
            uppermACC=max(bestACC);
            uppermNMI=max(bestNMI_sqrt);
            result_path = strcat('newresult_Tumors9','/','alpha_',num2str(alpha),'_beta_',num2str(beta),'_nu_',num2str(nu),'_uppermACC_',num2str(uppermACC),'_uppermNMI_',num2str(uppermNMI),'_result.mat');
            save(result_path,'mtrResult');
        end
    end
end
t_end = clock;
disp(['exe time: ',num2str(etime(t_end,t_start))]);

%save the best results among all the parameters
%result_path = strcat(dataset,'\','best','_result_',dataset,'_RSFS','.mat');
save('newresult_Tumors9','FeaNumCandi','bestNMI_sqrt','bestACC');

f = 1;
end
