import mxnet as mx
from mxnet import nd, autograd, gluon
import numpy as np
import math
import yaml

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# np.random.seed(cfg['seed'])

def simple_mean_filter(gradients, net, f, byz):
    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in gradients]
    byz(param_list, f)
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    grad_collect = []
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    return grad_collect

def multiply_norms(gradients, f):
    euclidean_distance = []
    for i, x in enumerate(gradients):
        norms = [nd.norm(p) for p in x]
        norm_product = 1
        for each in norms:
            norm_product *= float(each.asnumpy()[0])
        euclidean_distance.append((i, norm_product))
    # euclidean_distance = sorted(euclidean_distance, key=lambda x: x[1], reverse=True)
    # output = []
    # for i in range(f, len(gradients)):
    #     output.append(gradients[euclidean_distance[i][0]])
    output = [gradients[x[0]] for x in sorted(euclidean_distance, key=lambda x: x[1], reverse=True)[f:]]
    return output

def cgc_by_layer(gradients, f):
    layer_list = []
    for layer in range(len(gradients[0])):
        grads = [x[layer] for x in gradients]
        norms = [nd.norm(p) for p in grads]
        euclidean_distance = [(i, norms[i]) for i in range(len(grads))]
        layer_output = [grads[x[0]] for x in sorted(euclidean_distance, key=lambda x: x[1], reverse=True)[f:]]
        layer_list.append(layer_output)

    output = []
    for i in range(len(gradients) - f):
        grad = []
        for layer in range(len(gradients[0])):
            grad.append(layer_list[layer][i])
        output.append(grad)
    return output

def cgc_filter(gradients, net, f, byz):
    """Gets rid of the largest f gradients away from the norm"""
    cgc_method = cfg['cgc_method']
    if cgc_method == 'by-layer':
        output = cgc_by_layer(gradients, f)
    else:
        output = multiply_norms(gradients, f)

    # X is a 2d list of nd array
    param_list = [nd.concat(*[xx.reshape((-1, 1)) for xx in x], dim=0) for x in output]
    byz(param_list, f)
    mean_nd = nd.mean(nd.concat(*param_list, dim=1), axis=-1)
    grad_collect = []
    idx = 0
    for j, (param) in enumerate(net.collect_params().values()):
        if param.grad_req != 'null':
            # mapping back to the collection of ndarray
            # append to list for uploading to cloud
            grad_collect.append(mean_nd[idx:(idx+param.data().size)].reshape(param.data().shape))
            idx += param.data().size
    return grad_collect