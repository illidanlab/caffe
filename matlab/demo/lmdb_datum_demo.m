% Example of using lmdb and caffe protobuf (datum) in matlab. 
%
% by Jiayu, July 1, 2015. 
%
% NOTE 1. start matlab with a specified libtiff.5.dylib.
%    DYLD_INSERT_LIBRARIES=/usr/local/lib/libtiff.5.dylib /Applications/MATLAB_R2012b.app/bin/matlab &
%
%      2. install matlab-lmdb 
%   https://github.com/illidanlab/matlab-lmdb
%
%      3. the image num (the first input_num) in the model file should set to 1. 
%         will fix later. 

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

addpath ../../../matlab-lmdb/  % change to your matlab-lmdb path

% make sure you have followed the MNIST example to download the 
% MNSIT data and compiled the lmdb database. 
cur_director = pwd;
net_model    = strcat(cur_director, '/../../examples/mnist/lenet.prototxt');
net_weights  = strcat(cur_director, '/../../examples/mnist/lenet_iter_10000.caffemodel');
db_path      = strcat(cur_director, '/../../examples/mnist/mnist_test_lmdb');
use_gpu = 0;
phase = 'test';

% load an existing lmdb database (crated using the shell in example). 
database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);

% create caffe net instance
caffe.set_mode_cpu();
net = caffe.Net(net_model, net_weights, phase);
% reshape the last dimension so we can run image by image. 
model_shape = net.blobs('data').shape;
model_shape(4) = 1;
net.blobs('data').reshape(model_shape); % reshape blob 'data'
net.reshape();


max_count = 10; % maximum test cases
verbose   = 2;  % display everything. 

% compute accuracy. 
acc = mnist_test_lmdb( database, net, max_count, verbose);



