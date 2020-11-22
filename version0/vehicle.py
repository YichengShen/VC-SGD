import math
from neural_network import Neural_Network
import random
import numpy as np
import yaml
import heapq

import mxnet as mx
from mxnet import nd, autograd, gluon


file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# random.seed(cfg['seed'])
# np.random.seed(cfg['seed'])

class Vehicle:
    """
    Vehicle object for Car ML Simulator.
    Attributes:
    - car_id
    - x
    - y
    - speed
    - model
    - training_data_assigned
    - training_label_assigned
    - gradients
    """
    def __init__(self, car_id):
        self.car_id = car_id
        self.x = 0
        self.y = 0
        self.speed = 0
        self.net = None
        self.training_data_assigned = []
        self.training_label_assigned = []                     
        self.gradients = None
        # self.rsu_assigned = None


    def set_properties(self, x, y, speed):
        self.x = x
        self.y = y
        self.speed = speed

    def download_model_from(self, central_server):
        self.net = central_server.net

    def compute(self, simulation, closest_rsu):
        neural_net = Neural_Network()
        with autograd.record():
            output = self.net(self.training_data_assigned)
            if cfg['attack'] == 'label' and len(closest_rsu.accumulative_gradients) < cfg['num_faulty_grads']:
                loss = neural_net.loss(output, 9 - self.training_label_assigned)
            else:
                loss = neural_net.loss(output, self.training_label_assigned)
        loss.backward()

        grad_collect = []
        for param in self.net.collect_params().values():
            if param.grad_req != 'null':
                grad_collect.append(param.grad().copy())
        self.gradients = grad_collect
        # print(self.gradients)
        # print(len(self.gradients))
        # for i in range(len(self.gradients)):
        #     print(len(self.gradients[i]))

    def upload(self, simulation, closest_rsu):
        rsu = closest_rsu
        rsu.accumulative_gradients.append(self.gradients)
        # RSU checks if enough gradients collected
        if len(rsu.accumulative_gradients) >= cfg['simulation']['maximum_rsu_accumulative_gradients']:
            rsu.communicate_with_central_server(simulation.central_server)

    def compute_and_upload(self, simulation, closest_rsu):
        self.compute(simulation, closest_rsu)
        self.upload(simulation, closest_rsu)


    
    # Return the RSU that is cloest to the vehicle
    def closest_rsu(self, rsu_list):
        shortest_distance = 99999999 # placeholder (a random large number)
        closest_rsu = None
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range and distance < shortest_distance:
                shortest_distance = distance
                closest_rsu = rsu
        return closest_rsu

    # Return a list of RSUs that is within the range of the vehicle
    # with each RSU being sorted from the closest to the furtherst
    def in_range_rsus(self, rsu_list):
        in_range_rsus = []
        for rsu in rsu_list:
            distance = math.sqrt((rsu.rsu_x - self.x) ** 2 + (rsu.rsu_y - self.y) ** 2)
            if distance <= rsu.rsu_range:
                heapq.heappush(in_range_rsus, (distance, rsu))
        return [heapq.heappop(in_range_rsus)[1] for i in range(len(in_range_rsus))]
