clear all;
clc;

load TrainTestSplit2

[mappedA mapping] = lda_modified(trainDataLDA, trainClassLDA, 22);

 
% for i = 1:size(mapping.val)
%  if mapping.val(i) > 0
%      lda_var(i,1) = mapping.val(i,1);
%  end
% end
% 
% 
