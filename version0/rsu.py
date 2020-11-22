from neural_network import Neural_Network 
import byz
import nd_aggregation
import numpy as np
import yaml
import random
from mxnet import nd

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])

class RSU:
    """
    Road Side Unit object for Car ML Simulator.
    Attributes:
    - rsu_id
    - rsu_x
    - rsu_y
    - rsu_range
    - accumulative_gradients
    """
    def __init__(self, rsu_id, rsu_x, rsu_y, rsu_range, traffic_proportion):
        self.rsu_id = rsu_id
        self.rsu_x = rsu_x
        self.rsu_y = rsu_y
        self.rsu_range = rsu_range
        self.accumulative_gradients = []

    def aggregate(self, net, grad_list, byz=byz.no_byz):
        f = cfg['num_faulty_grads']
        aggre_method = cfg['aggregation_method']
        if aggre_method == 'cgc':
            return nd_aggregation.cgc_filter(grad_list, net, f, byz)
        elif aggre_method == 'simplemean':
            return nd_aggregation.simple_mean_filter(grad_list, net, f, byz)

    # The RSU updates the model in the central server with its accumulative gradients and downloads the 
    # latest model from the central server
    def communicate_with_central_server(self, central_server):
        # Different methods of attacking
        if cfg['attack'] == 'signflip':
            byz.signflip_attack(self)
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients)
        elif cfg['attack'] == 'gaussian':
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients, byz.gaussian_attack)
        elif cfg['attack'] == 'bitflip':
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients, byz.bitflip_attack)
        else:
            # NO attack
            aggre_gradients = self.aggregate(central_server.net, self.accumulative_gradients)
            
        self.accumulative_gradients = []
        central_server.accumulative_gradients.append(aggre_gradients)
        # if enough gradients accumulated in cloud, then update model
        if len(central_server.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            central_server.update_model()