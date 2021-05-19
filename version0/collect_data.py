import ruamel.yaml
import yaml
import os



for num_vehicular_clouds in [6, 9]:
    for I in range(1,6):
        with open('config.yml', 'r') as file:
            cfg = yaml.load(file, Loader=yaml.FullLoader)
            cfg['num_vc'] = num_vehicular_clouds
            cfg['simulation']['num_vc'] = num_vehicular_clouds

        with open('config.yml', 'w') as fp:
            yaml.dump(cfg, fp)

        os.system(f"python3 main.py --num-round {I}")