function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

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

fprintf('===========================================================\n');
fprintf('Looking for the best [C, sigma] values\n');
err = inf;
values = [0.01 0.03 0.10 0.30 1.00 3.00 10.00 30.00];

for _C = values
  for _sigma = values
    fprintf('Training and evaluating using cross validation when\n[_C, _sigma] = [%f %f]\n', _C, _sigma);
    model = svmTrain(X, y, _C, @(x1, x2) gaussianKernel(x1, x2, _sigma));
    e = mean(double(svmPredict(model, Xval) ~= yval));
    fprintf('prediction error: %f\n', e);
    if( e <= err )
      fprintf('err updated!\n');
      C = _C;
      sigma = _sigma;
      err = e;
      fprintf('[C, sigma] = [%f %f]\n', C, sigma);
    end
    fprintf('===========================================================\n');
  end
end

fprintf('\nSearch Complete.\nThe best values for [C, sigma] = [%f %f] with a prediction error = %f\n\n', C, sigma, err);
fprintf('===========================================================\n');

% =========================================================================

end
