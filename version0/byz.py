import mxnet as mx
from mxnet import nd
import random
import yaml
import numpy as np


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# random.seed(cfg['seed'])
# mx.random.seed(cfg['seed'])

# no faulty workers
def no_byz(v, f):
    pass

# failures that add Gaussian noise
def gaussian_attack(v, f):
    for i in range(f):
        v[i] = mx.nd.random.normal(0, 200, shape=v[i].shape)

# bit-flipping failure
def bitflip_attack(v, f):
    for i in range(f):
        if i > 0:
            v[i][:] = -v[0]
    v[0][:] = -v[0]

def signflip_attack(rsu):
        for i in random.sample(range(10), 2):
            for j in range(20):
                if j % 2:      
                    rsu.accumulative_gradients[i][j] = nd.array(5000*np.negative(rsu.accumulative_gradients[i][j].asnumpy()))
            # rsu.accumulative_gradients[i][2] = nd.array(5000*np.negative(rsu.accumulative_gradients[i][2].asnumpy()))

