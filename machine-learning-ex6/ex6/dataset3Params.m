function [C,sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%Create a C vector
c1=[0.01 0.03 0.1 0.3 1 3 10 30];
sig=[0.01 0.03 0.1 0.3 1 3 10 30];

#Create all possible combinations
[p,q] = meshgrid(c1, sig);
pairs = [p(:) q(:)];

c=	[];
sigg=[];
iter=[];
err=[];

for(i = 1:rows(pairs))

%fit the model using parameters
model= svmTrain(X, y, pairs(i,1), @(x1, x2) gaussianKernel(x1, x2, pairs(i,2))); 

%do the prediction on val dataset
predictions = svmPredict(model, Xval);

cur_er=mean(double(predictions ~= yval));

iter=[iter i];
c=[c pairs(i,1)];
sigg=[sigg pairs(i,2)];
err=[err cur_er];

endfor

[val ind]=min(err);

C=c(ind);
sigma=sigg(ind);


% =========================================================================

end
