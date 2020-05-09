
load traintestLDA
load c_and_sigma

uniqueClass = unique(trainClassLDA);
c = 2^bestC;
sig = 2^bestSigma;


tic
for i=1:size(uniqueClass,1)       
    groups = ismember(trainClassLDA, uniqueClass(i));       
    opts = statset('Display','final','MaxIter',500000);
    svmStructRbf(i) = svmtrain(trainDataLDA,groups,'Method','SMO','SMO_Opts',opts,'BoxConstraint', c, 'Kernel_Function','rbf', 'RBF_Sigma', sig,'KernelCacheLimit',10000);
end
toc

save svmrbf.mat  svmStructRbf;
