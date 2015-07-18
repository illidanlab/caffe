#!/usr/bin/env python
'''
A python program that modifies solver configurations

Have pycaffe installed and configured before you use this script.

By Jiayu Zhou, July 17, 2015.
'''
import sys
import caffe.proto
from google.protobuf import text_format

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print 'Insufficient number of parameters. '
        print_usage()
        sys.exit(-1)

    filename         = sys.argv[1];

    ## 1. step read it from a proto text file.
    sys.stderr.write( 'reading from file: %s \n' % filename)
    f = open(filename, "rb")
    # Create the solver config
    solver=caffe.proto.caffe_pb2.SolverParameter()
    solverCfg=text_format.Merge(str(f.read()), solver)

    f.close()

    ## 2. modify the network structure

    param_set_pairs = sys.argv[2:len(sys.argv)]
    sys.stderr.write('there are in total %s modifications\n' % len(param_set_pairs))
    for param_pair in param_set_pairs:
        param_pair_parse = param_pair.split('=')
        if len(param_pair_parse) != 2:
            raise ValueError('Cannot process argument %s', param_pair)
        sys.stderr.write('Set [%s] field to [%s]\n' % (param_pair_parse[0], param_pair_parse[1]))
        setattr(solverCfg,param_pair_parse[0],param_pair_parse[1])

    ## 3. output the changed prototxt via stdout.
    str_proto = text_format.MessageToString(solverCfg)
    sys.stdout.write(str_proto)

    # everything is done. Hooray!
    sys.exit(0)
