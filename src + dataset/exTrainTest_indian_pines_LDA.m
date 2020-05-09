
load twoD
load classGT

data = twoD;
class = classGT;


dataIndex=ismember(classGT,[2,3,5,6,8,10,11,12,14]);

dataForNineClass=data(dataIndex,:);
class=classGT(dataIndex,:);

classIndexOne=ismember(class,2);
classIndexTwo=ismember(class,3);
classIndexThree=ismember(class,5);
classIndexFour=ismember(class,6);
classIndexFive=ismember(class,8);
classIndexSix=ismember(class,10);
classIndexSeven=ismember(class,11);
classIndexEight=ismember(class,12);
classIndexNine=ismember(class,14);


class(classIndexOne)=1;
class(classIndexTwo)=2;
class(classIndexThree)=3;
class(classIndexFour)=4;
class(classIndexFive)=5;
class(classIndexSix)=6;
class(classIndexSeven)=7;
class(classIndexEight)=8;
class(classIndexNine)=9;

[trainIdx, testIdx] = crossvalind('holdout',class,0.50);

trainData=dataForNineClass(trainIdx,:);
testData=dataForNineClass(testIdx,:);
trainClass=class(trainIdx);
testClass=class(testIdx);

trainData=double(trainData);
testData=double(testData);



[mappedLDA mappingLDA] = lda(trainData, trainClass, 220);


numValidfeatures = sum(mappingLDA.val>0);
[n,p] = size(testData);
testDataCenter = testData - repmat(mean(trainData,1),n,1);
validfeatures=mappingLDA.M(:,1:numValidfeatures);

testDataOnPc = testDataCenter*validfeatures;
trainDataOnPc = mappedLDA(:,1:numValidfeatures);

testDataLDA = testDataOnPc;
trainDataLDA = trainDataOnPc;
trainClassLDA = class(trainIdx,:);
testClassLDA = class(testIdx,:);


save traintestLDA trainDataLDA testDataLDA trainClassLDA testClassLDA;

