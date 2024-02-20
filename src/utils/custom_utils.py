import os
import configparser

import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))


def read_yaml():
    yaml_path = os.path.join(ROOT_DIR, "data", "roboflow", "data.yaml")
    with open(yaml_path, 'r') as yaml_file:
        data = yaml.safe_load(yaml_file)
    return data


def get_roboflow_api_key():
    """
    retrieve the api-key for roboflow from the config.ini
    :return: api_key
    """
    CFG_PATH = os.path.join(ROOT_DIR, 'cfg/config.ini')
    config = configparser.ConfigParser()
    config.read(CFG_PATH)

    api_key = config['Roboflow_API_KEY']['private_api_key']
    return api_key
