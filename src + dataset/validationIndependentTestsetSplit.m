
load traintestLDA

tic
[train test] = crossvalind('holdout',testClassLDA,0.60);

splitDataForValidation = testDataLDA(train,:);
splitDataForTest = testDataLDA(test,:);
splitClassForValidation = testClassLDA(train,:);
splitClassForTest = testClassLDA(test,:);

toc

save traintestSplit2.mat splitClassForValidation splitClassForTest splitDataForValidation splitDataForTest

