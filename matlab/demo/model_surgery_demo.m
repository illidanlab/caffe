% Example of using matcaffe for model surgery. 
%
% by Jiayu July 2nd, 2015.
%
% Please follow the lmdb_datum_demo for information on running matcaffe. 
%
clc
clear
close all

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

addpath ../../../matlab-lmdb/  % change to your matlab-lmdb path


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

% the evaluation functional. 
test_func = @(network) mnist_test_lmdb( database, network, 500, 0);

% test the original network 
acc0 = test_func( net);
fprintf('The orignal network has the accuracy of %.4f\n', acc0);

% investigate the ip1 level model parameter
ip1_mat     = net.params('ip1',1).get_data;
[U, S, V]   = svd(ip1_mat, 0);
S_diag_sqr  = diag(S).^2;
approx_rank = 20; % the rank used for approximation
info_loss = sum(S_diag_sqr(approx_rank+1:end))/sum(S_diag_sqr);
fprintf('A rank %d approximation on layer ip1 leads to approximation error %.4f\n', ...
    approx_rank, info_loss);
plot(diag(S))
title('Distribution of Singular Values in Layer ip1')

% generate the approximation 
U_r = U(:, 1:approx_rank); 
S_r = S(1:approx_rank, 1:approx_rank);
V_r = V(:, 1:approx_rank);
ip1_mat_r = U_r * S_r * V_r';

% adjust model and perform test
net.params('ip1',1).set_data(ip1_mat_r);
acc1 = test_func( net);
fprintf('The truncated network has the accuracy of %.4f\n', acc1);

% write the model out so you can use it later. 
net.save('model_after_surgery.caffemodel')
