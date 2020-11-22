import numpy as np
import yaml
import xml.etree.ElementTree as ET 

from vehicle import Vehicle
from rsu import RSU

file = open('config.yml', 'r')
cfg = yaml.load(file, Loader=yaml.FullLoader)

# np.random.seed(cfg['seed'])

class SUMO_Dataset:
    """
    Data read from SUMO XML files.
    Attributes:
    - ROU_file
    - NET_file
    """
    def __init__(self, ROU_file, NET_file):
        self.ROU_file = ROU_file
        self.NET_file = NET_file

    def vehicleDict(self):
        tree = ET.parse(self.ROU_file)
        root = tree.getroot()
        vehicle_dict = {}
        for vehicle in root.findall('trip'):
            vehicle_dict[vehicle.attrib['id']] = Vehicle(vehicle.attrib['id'])
        return vehicle_dict

    def rsuList_random(self, rsu_range, rsu_nums):
        tree = ET.parse(self.NET_file)
        root = tree.getroot()
        rsu_list = []
        junction_list = np.random.choice(root.findall('junction'), rsu_nums, replace=False)
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            rsu_list.append(RSU(id, float(junction_list[i].attrib['x']), float(junction_list[i].attrib['y']), rsu_range, 1/cfg['simulation']['num_rsu']))
        return rsu_list

    def rsuList(self, rsu_range, rsu_nums, junction_list):
        rsu_list = []
        for i in range(rsu_nums):
            id = 'rsu' + str(i)
            rsu_list.append(RSU(id, float(junction_list[i][0].attrib['x']), float(junction_list[i][0].attrib['y']), rsu_range, junction_list[i][1]))
        return rsu_list