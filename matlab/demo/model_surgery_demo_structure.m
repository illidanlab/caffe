% Example of using matcaffe for model surgery with structural change.
%
% by Jiayu July 13th, 2015.
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

python_bin = '/usr/local/bin/python';
setenv('DYLD_LIBRARY_PATH', '/usr/local/bin/'); % python lib path.
%setenv('LD_LIBRARY_PATH', '/usr/lib64;/usr/local/cuda-7.0/lib64;/usr/local/cudnn/'); %(Cuda Ubuntu)

addpath ../../../matlab-lmdb/  % change to your matlab-lmdb path
cur_director = pwd;

db_path      = strcat(cur_director, '/../../examples/mnist/mnist_test_lmdb');
% load an existing lmdb database (crated using the shell in example).
database = lmdb.DB(db_path, 'RDONLY', true, 'NOLOCK', true);
caffe.set_mode_cpu();

%% Load the original model and perform testing

% create caffe net instance
net_model    = strcat(cur_director, '/../../examples/mnist/lenet.prototxt');
net_weights  = strcat(cur_director, '/../../examples/mnist/lenet_iter_10000.caffemodel');
phase_test = 'test';
net = caffe.Net(net_model, net_weights, phase_test);

% the evaluation functional.
test_func = @(network) mnist_test_lmdb( database, network, 3000, 0);

% test the original network
[acc0, elps0 ] = test_func( net);
fprintf('** The orignal network has the accuracy of %.4f with time cost %.4f sec.\n', acc0, elps0);

%% Investigate model and test model surgery without structure change.
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
[acc1, elps1] = test_func( net);
fprintf('** The truncated network (without structure change) has the accuracy of %.4f and time cost %.4f sec.\n', acc1, elps1);

% write the model out so you can use it later.
%net.save('model_after_surgery.caffemodel')

%% Model surgery with structure change.
% retain the weights (so we can later assign to a new structure)
source_layer_name    = 'ip1';
target_layer_names   = {'ip1a'; 'ip1b'};
param_ip1a = {U_r,         zeros(20, 1)};
param_ip1b = {S_r * V_r',  net.params(source_layer_name,2).get_data};
target_layer_params  = {param_ip1a, param_ip1b};

% construct new parameter names and parameter storage.
source_layer_id = net.name2layer_index(source_layer_name);
net_layer_name = net.layer_names;

net_ip1rk20_layer_name = ...
    [net_layer_name(1:(source_layer_id-1)); ...
    target_layer_names; ...
    net_layer_name((source_layer_id+1):length(net_layer_name))];
net_ip1rk20_param      = cell(length(net_ip1rk20_layer_name), 1);

disp('Populating parameter weight blobs.')
for i = 1: length(net_ip1rk20_layer_name)
    layer_name = net_ip1rk20_layer_name{i};

    if sum(strcmp(net.layer_names, layer_name))>0
        % if the layers are in previous layers.

        % explore how many blobs we need to store.
        layer_blobs_num = size(net.layer_vec(net.name2layer_index(layer_name)).params, 2);

        % store the blobs so we can re-assign it later.
        layer_blobs     = cell(layer_blobs_num, 1);
        for j = 1: layer_blobs_num
            layer_blobs{j} = net.params(layer_name, j).get_data;
        end
        net_ip1rk20_param{i} = layer_blobs;

    elseif ~isempty(find(strcmp(target_layer_names, layer_name), 1))
        % if the layers are newly inserted layers.

        target_layer_idx = find(strcmp(target_layer_names, layer_name));
        net_ip1rk20_param{i} = target_layer_params{target_layer_idx};
    else
        error('Unexpected layer name: %s', layer_name)
    end
end

%%%% strategy 1: load model file if there is an exsiting one.
% disp('Read new model file.')
% net_ip1rk20_model   = strcat(cur_director, '/../../examples/mnist/lenet_ip1rk20.prototxt');
% net_ip1rk20_weights = strcat(cur_director, '/../../examples/mnist/lenet_ip1rk20_iter_10000.caffemodel');
% net = caffe.Net(net_ip1rk20_model, net_ip1rk20_weights, phase);

% %%% strategy 2: use solver to instantiate an empty network with modified
% %%% net prototxt
% net_ip1rk20_solver      = strcat(cur_director, '/lenet_ip1rk20_solver.prototxt');
% net_ip1rk20_model       = strcat(cur_director, '/lenet_ip1rk20.prototxt');
% net_ip1rk20_init_weight = strcat(cur_director, '/lenet_ip1rk20_init.caffemodel');
%
% solver = caffe.Solver(net_ip1rk20_solver);
% % cannot directly use the net from solver because it is at TRAIN phase
% % and it seems caffe::set_phase is not available any more:
% %    https://groups.google.com/forum/#!topic/caffe-users/GnlGGnut424
% solver.net.save(net_ip1rk20_init_weight);
% net = caffe.Net(net_ip1rk20_model, net_ip1rk20_init_weight, phase_test);

%%%
net_solver              = strcat(cur_director, '/lenet_solver.prototxt');
net_ip1rk20_solver      = strcat(cur_director, '/lenet_ip1rk20_solver_gen.prototxt');
net_ip1rk20_model       = strcat(cur_director, '/lenet_ip1rk20_gen.prototxt');
net_train_test          = strcat(cur_director, '/lenet_train_test.prototxt');
net_ip1rk20_train_test  = strcat(cur_director, '/lenet_ip1rk20_train_test_gen.prototxt');
net_ip1rk20_init_weight = strcat(cur_director, '/lenet_ip1rk20_init.caffemodel');

disp('Transforming solver configureation...')
state = system([python_bin, ' netsurg_solvercfg.py ',net_solver, ...
            ' net=', net_ip1rk20_train_test , ...
            ' 2>/dev/null 1> ', net_ip1rk20_solver]);
if state~= 0, error('failed'); end

disp('Transforming model structure file...')
state = system([python_bin, ' netsurg_fclw.py ',source_layer_name, ...
            ' ', target_layer_names{1}, ' ' ,target_layer_names{2}, ...
            ' ', int2str(approx_rank),  ' ', net_model, ...
            ' 2>/dev/null 1> ', net_ip1rk20_model]);
if state~= 0, error('failed'); end

disp('Transforming model train/test file...')
state = system([python_bin, ' netsurg_fclw.py ',source_layer_name, ...
            ' ', target_layer_names{1}, ' ' ,target_layer_names{2}, ...
            ' ', int2str(approx_rank),  ' ', net_train_test, ...
            ' 2>/dev/null 1> ', net_ip1rk20_train_test]);
if state~= 0, error('failed'); end

disp('Initializing solver using new structure...')
solver = caffe.Solver(net_ip1rk20_solver);
solver.net.save(net_ip1rk20_init_weight);
disp('Loading new model weight...')
net = caffe.Net(net_ip1rk20_model, net_ip1rk20_init_weight, phase_test);

% "fill in" the parameters layer by layer.
disp('Filling in new weights...')
for i = 1: length(net_ip1rk20_layer_name)
    layer_name = net_ip1rk20_layer_name{i};

    % check the consistency of the layer parameters.
    layer_blobs_num  = size(net.layer_vec(net.name2layer_index(layer_name)).params, 2);
    stored_blobs_num = length(net_ip1rk20_param{i});
    if(layer_blobs_num ~= stored_blobs_num)
        error('Blob number at layer [%s] (%d blobs) is inconsistently stored (%d blobs).', layer_name, layer_blobs_num, stored_blobs_num)
    end

    % set parameters.
    for j = 1: layer_blobs_num
        net.params(layer_name,j).set_data(net_ip1rk20_param{i}{j});
    end
end
[acc3, elps3 ] = test_func( net);
fprintf('** The truncated network (with structure change) has the accuracy of %.4f with time cost %.4f sec.\n', acc3, elps3);
