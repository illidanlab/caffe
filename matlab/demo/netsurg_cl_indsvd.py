#!/usr/bin/env python
'''
A python program that reads a caffe network definition, and replaces a specific
convolutional layer (restricted to 3 input channels) with:
    a slice layer (slicing the input channels),
    a convolution layer for each channel,
    another layer for each channel,
    and a elementary sum layer.

Have pycaffe installed and configured before you use this script.

By Jiayu Zhou, July 20, 2015.
'''
import sys
import caffe.proto
from google.protobuf import text_format

def create_ind_svd_network(network, old_layer_name, rank_num, subchannel_suffix_arr = ['R', 'G', 'B']):
    '''
    Remove a layer and replace it by decomposed convolutional layers.

    NOTE: The input subchannel_suffix_arr also decides how many channels.
          This script does not check the dimensionality consistency.
          It has to be consistent with the input blobs.
    '''

    # find old_layer instance.
    old_layer_instance = None
    old_layer_index    = None

    layer_list = []
    for i in range(0, len(network.layer)):
        layer_instance = network.layer[i]
        if layer_instance.name == old_layer_name:
            old_layer_instance = layer_instance
            old_layer_index    = i

            # add slice layer
            slice_layer = network.layer.add()
            slice_layer.name = old_layer_name + '_slice'
            slice_layer.type = 'Slice'

            slice_layer.slice_param.axis = 1
            slice_layer.slice_param.slice_point.append(1)
            slice_layer.slice_param.slice_point.append(2)

            for subchannel_suffix in subchannel_suffix_arr:
                slice_layer.top.append(old_layer_name + '_channel_'+subchannel_suffix)
            for buttom_layers in layer_instance.bottom:
                slice_layer.bottom.append(buttom_layers)

            layer_list = layer_list + [slice_layer]

            # add conv1_11x11_R/G/B
            for subchannel_suffix in subchannel_suffix_arr:
                new_layer = network.layer.add()
                new_layer.CopyFrom(old_layer_instance)
                new_layer.name  = old_layer_name + '_11x11_' + subchannel_suffix
                new_layer.convolution_param.num_output = rank_num

                for bottom_layer in new_layer.bottom:
                    new_layer.bottom.remove(bottom_layer)
                new_layer.bottom.append(old_layer_name + '_channel_' + subchannel_suffix)
                for top_layer in new_layer.top:
                    new_layer.top.remove(top_layer)
                new_layer.top.append(old_layer_name + '_response_' + subchannel_suffix)

                layer_list = layer_list + [new_layer]

            # add conv1_1x1_R/G/B
            for subchannel_suffix in subchannel_suffix_arr:
                new_layer = network.layer.add()
                new_layer.CopyFrom(old_layer_instance)
                new_layer.name  = old_layer_name + '_1x1_' + subchannel_suffix
                new_layer.convolution_param.kernel_size = 1
                new_layer.convolution_param.stride = 1

                for bottom_layer in new_layer.bottom:
                    new_layer.bottom.remove(bottom_layer)
                new_layer.bottom.append(old_layer_name + '_response_' + subchannel_suffix)
                for top_layer in new_layer.top:
                    new_layer.top.remove(top_layer)
                new_layer.top.append(old_layer_name + '_LinCom_' + subchannel_suffix)

                layer_list = layer_list + [new_layer]

            # add cross_channel_sum
            new_layer = network.layer.add()
            new_layer.name = old_layer_name + '_cross_channel_sum'
            new_layer.type = 'Eltwise'
            for subchannel_suffix in subchannel_suffix_arr:
                new_layer.bottom.append(old_layer_name + '_LinCom_' + subchannel_suffix)
            for top_layer in old_layer_instance.top:
                new_layer.top.append ( top_layer )
            new_layer.eltwise_param.operation = 1
            for subchannel_suffix in subchannel_suffix_arr:
                new_layer.eltwise_param.coeff.append(1.0)
            layer_list = layer_list + [new_layer]

        else:
            # copy to default list.
            layer_list = layer_list + [layer_instance]
            #layer_instance.CopyFrom(network.layer[i-1])

    # remove everything.
    while len(network.layer) > 0:
        network.layer.remove(network.layer[0])

    # add newly constructed layers.
    for layer in layer_list:
        new_layer = network.layer.add()
        new_layer.CopyFrom(layer)

def print_usage():
    print 'Usage: '
    print '    1> Use a specific prototxt file as input'
    print '       ./netsurg_cl_indsvd.py source_layer rank_num proto_file'
    print '       Example:  ./netsurg_cl_indsvd.py conv1 40 train_val.prototxt 2>/dev/null'
    print '    2> Use stdin as input'
    print '       Example:  cat train_val.prototxt | ./netsurg_cl_indsvd.py conv1 40 2>/dev/null'
    print '    3> You can connect multiple pipes'
    print '       Example: cat train_val.prototxt | ./netsurg_cl_indsvd.py conv1 40 2>/dev/null | ./netsurg_fclw.py ip2 ip2a ip2b 10 2>/dev/null'

if __name__ == "__main__":

    if len(sys.argv) < 3:
        print 'Insufficient number of parameters. '
        print_usage()
        sys.exit(-1)

    old_layer_name   = sys.argv[1]; # e.g., 'conv1'
    rank_num         = int(sys.argv[2]);

    sys.stderr.write('Individual low-rank factorization on convolutional layer [%s] with rank [%d]\n'
            % (old_layer_name, rank_num))

    #choose input accordingly.
    if len(sys.argv) >= 4:
        filename         = sys.argv[3];
        sys.stderr.write( 'reading from file: %s \n' % filename)
        f = open(filename, "rb")
    else:
        sys.stderr.write( 'try reading from stdin. \n')
        f = sys.stdin

    # Create the network and modify it.

    ## 1. step read it from a proto text file.
    net = caffe.proto.caffe_pb2.NetParameter()
    network = text_format.Merge(str(f.read()), net)
    new_net_func   = lambda net: create_ind_svd_network(net, old_layer_name, rank_num)
    f.close()

    ## 2. modify the network structure
    new_net_func(network)

    ## 3. output the changed prototxt via stdout.
    str_proto = text_format.MessageToString(net)
    sys.stdout.write(str_proto)

    # everything is done. Hooray!
    sys.exit(0)
