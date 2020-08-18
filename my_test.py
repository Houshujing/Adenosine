# from theano import function, config, shared, tensor
# import numpy
# import time
#
# vlen = 10 * 30 * 768  # 10 x #cores x # threads per core
# iters = 1000
#
# rng = numpy.random.RandomState(22)
# x = shared(numpy.asarray(rng.rand(vlen), config.floatX))
# f = function([], tensor.exp(x))
# print(f.maker.fgraph.toposort())
# t0 = time.time()
# for i in range(iters):
#     r = f()
# t1 = time.time()
# print("Looping %d times took %f seconds" % (iters, t1 - t0))
# print("Result is %s" % (r,))
# if numpy.any([isinstance(x.op, tensor.Elemwise) and
#               ('Gpu' not in type(x.op).__name__)
#               for x in f.maker.fgraph.toposort()]):
#     print('Used the cpu')
# else:
#     print('Used the gpu')

import os
import pandas as pd

base_dir = '/home/jianping/PycharmProjects/neural-fingerprint-theano-master/druglike_1119/druglike_data/csv_files'
file_dir = os.path.join(base_dir, 'yatcm_training_fix.csv')
# df = pd.read_csv(file_dir)
# print(df.columns)
test_list = []
for line in open(file_dir):
    test_list.append(line.split(',')[1])
print(test_list)