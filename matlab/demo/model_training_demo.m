% Example of using matcaffe for model surgery. 
%
% by Jiayu July 4th, 2015.
%
% Please follow the lmdb_datum_demo for information on running matcaffe. 
%
clc
close all

if exist('../+caffe', 'dir')
  addpath('..');
else
  error('Please run this demo from caffe/matlab/demo');
end

addpath ../../../matlab-lmdb/  % change to your matlab-lmdb path



model_dir   = '/Users/jiayu.zhou/workspace/caffe/matlab/demo/';
net_model   = [model_dir 'lenet.prototxt'];
net_weights = [model_dir 'lenet_iter_10000.caffemodel'];
net_solver  = [model_dir 'lenet_solver.prototxt'];

pretrained_net = caffe.Net(net_model, net_weights,'test');

% if the lenet_solver.prototxt specified a relative path, then it is 
% likely to crash the Matlab. 
solver = caffe.Solver(net_solver);
solver.net.copy_from(net_weights)
solver.solve()

