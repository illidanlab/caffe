#!/usr/bin/env python
'''
A python program that reads a caffe network definition, and replaces a specific
inner product layer with two connected inner product layers with a specific
bottomneck rank.

Have pycaffe installed and configured before you use this script.

By Jiayu Zhou, July 16, 2015.
'''
import sys
import caffe.proto
from google.protobuf import text_format

def create_low_rank_network(network, old_layer_name, layer_a_name, layer_b_name, rank_num):
    ''' Remove a layer and replace it by two layers with a bottleneck.

        Network structure:
           network start ... -> new_layer_a -> new_layer_b -> ... network end

        The old name will serve as a template, so we can only modify the rank.

        network:        the original network
        old_layer_name: the source layer (e.g., ip1)
        layer_a_name:   the name of the lower-level in the factorization (e.g., ip1a)
        layer_b_name:   the name of the higher-level in the factorization (e.g., ip1b)
        rank_num:       the rank (dimension of the bottleneck)
    '''
    # find old_layer instance.
    old_layer_instance = None;
    old_layer_index    = None;


    new_layer_end = network.layer.add()
    for i in reversed(range(0, len(network.layer))):
        layer_instance = network.layer[i]
        if layer_instance.name == old_layer_name:
            old_layer_instance = layer_instance
            old_layer_index    = i
            break
        else:
            if i == 0:
                # we end up with nothing.
                break
            layer_instance.CopyFrom(network.layer[i-1])

    if old_layer_instance is None:
        raise ValueError('Cannot find layer named %s in the input network.' % old_layer_name )
    if old_layer_instance.type != 'InnerProduct':
        raise ValueError('The source layer %s (type: %s) in not an inner product layer.' % (old_layer_name, old_layer_instance.type) )

    # create new layers.
    new_layer_a = network.layer[old_layer_index]
    new_layer_b = network.layer[old_layer_index+1]
    new_layer_a.name = layer_a_name
    new_layer_b.name = layer_b_name
    new_layer_a.inner_product_param.num_output = rank_num

    # connect new layers.
    # assign layer_a top
    for layer in new_layer_a.top:
        new_layer_a.top.remove(layer)
    new_layer_a.top.append(layer_a_name)

    # assign layer_b bottom
    for layer in new_layer_b.bottom:
        new_layer_b.bottom.remove(layer)
    new_layer_b.bottom.append(layer_a_name)

    # the rest of the references are changed to layer_b_name
    # in order to avoid the loop, whose top and bottom have the same
    # blob id (old_layer_name), we change it to layer_b_name
    for layer_instance in network.layer:
        # check bottom
        if old_layer_name in layer_instance.bottom:
            layer_instance.bottom.remove(old_layer_name)
            layer_instance.bottom.append(layer_b_name)
        # check top
        if old_layer_name in layer_instance.top:
            layer_instance.top.remove(old_layer_name)
            layer_instance.top.append(layer_b_name)

def print_usage():
    print 'Usage: '
    print '    1> Use a specific prototxt file as input'
    print '       ./netsurg_fclw.py source_layer target_layer_1 target_layer_2 rank_num proto_file'
    print '       Example:  ./netsurg_fclw.py ip1 ip1a ip1b 30 lenet.prototxt 2>/dev/null'
    print '    2> Use stdin as input'
    print '       Example:  cat lenet.prototxt | ./netsurg_fclw.py ip1 ip1a ip1b 30 2>/dev/null'
    print '    3> You can connect multiple pipes'
    print '       Example: cat lenet.prototxt | ./netsurg_fclw.py ip1 ip1a ip1b 30 2>/dev/null | ./netsurg_fclw.py ip2 ip2a ip2b 10 2>/dev/null'

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print 'Insufficient number of parameters. '
        print_usage()
        sys.exit(-1)

    old_layer_name   = sys.argv[1]; # e.g., 'ip1'

    new_net_bottom   = sys.argv[2];
    # e.g., 'ip1a' # top pointed to ip1 should now point to ip1a

    new_net_top      = sys.argv[3];
    # e.g., 'ip1b' # bottom pointed to ip1 should now point to ip1b

    rank_num         = int(sys.argv[4]);

    sys.stderr.write('Low-rank factorization layer [%s] into layer [%s] and layer [%s] with rank [%d]\n'
            % (old_layer_name, new_net_top, new_net_bottom, rank_num))

    # choose input accordingly.
    if len(sys.argv) >= 6:
        filename         = sys.argv[5];
        sys.stderr.write( 'reading from file: %s \n' % filename)
        f = open(filename, "rb")
    else:
        sys.stderr.write( 'try reading from stdin. \n')
        f = sys.stdin

    # Create the network and modify it.

    ## 1. step read it from a proto text file.
    net = caffe.proto.caffe_pb2.NetParameter()
    network = text_format.Merge(str(f.read()), net)
    new_net_func   = lambda net: create_low_rank_network(net, old_layer_name, new_net_bottom, new_net_top, rank_num)
    f.close()

    ## 2. modify the network structure
    new_net_func(network)

    ## 3. output the changed prototxt via stdout.
    str_proto = text_format.MessageToString(net)
    sys.stdout.write(str_proto)

    # everything is done. Hooray!
    sys.exit(0)
