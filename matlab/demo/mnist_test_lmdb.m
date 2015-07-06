function [ accuracy ] = mnist_test_lmdb( database, net, max_count, verbose)
%MNIST_TEST_LMDB test MNIST images from image net. 
%
% INPUTS
%   database: matlab_lmdb instance (preferable read only database)
%      e.g., database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);
%
%   net: matcaffe net instance
%      e.g., net = caffe.Net(net_model, net_weights, phase);
%
%   max_count: the maximum number of testing samples according to the order
%            of the database iterator. If it is less or equal than zero
%            then it will test every sample in the database (default
%            behavior) 
%
%   verbose: the amount of information displayed in console. 
%     0 -- no display at all. 
%     1 -- display a summary information. <-- default value. 
%     2 -- display at lof of information. 
%
% OUTPUT
%   accuracy: the accuracy on the test images. 
%
% By Jiayu Zhou, July 2nd, 2015. 
%

cursor = database.cursor('RDONLY', true);

if nargin<3, max_count = -1; end % maximum test cases
if nargin<4, verbose   = 1;  end


count = 0;       % number of samples processed. 
correctNum = 0;  % number of samples correctly predicted. 
while cursor.next()
  
    % key = cursor.key;
    value = cursor.value;
  
    % transform datum. 
    [image, label] = caffe.fromDatum(value);
  
    % prepare image
    data = single(image);
    data = permute(data, [2,1,3]);
  
    % generate prediction 
    scores = net.forward({data});
    predict_class = find(scores{1}==1) - 1; % shift 1
    
    if verbose >= 2
        fprintf('[%u] Class %u predicted as %u \n', count+1, label, predict_class)
    end
  
    if(predict_class == label)
        correctNum = correctNum + 1;
    end
  
    count = count + 1; 
    if (max_count >0 && count >= max_count)
        break;
    end
end

accuracy = correctNum/count;

if verbose >= 1
    fprintf('Correctly classified %d images out of %d ( %d percent)\n', correctNum, count, accuracy * 100)
end

clear cursor;


end

