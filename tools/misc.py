import os 
import numpy as np
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import matplotlib
import yaml
from scipy.misc import imsave
import logging
import shutil

def load_cf(yaml_filepath):
    """
    Load a YAML configuration file.

    Parameters
    ----------
    yaml_filepath : str

    Returns
    -------
    cf : dict
    """
    # Read YAML experiment definition file
    with open(yaml_filepath, 'r') as stream:
        cf = yaml.load(stream)
    # Copy config file
    if not os.path.exists(cf['experiments']+cf['exp_name']):
        os.makedirs(cf['experiments']+cf['exp_name'])    
    if cf['train']:
        shutil.copyfile('config.yaml', os.path.join(cf['experiments'],cf['exp_name'], "config.yaml"))        
    return cf


def set_logger(cf):
    log=logging.getLogger()
    logging.basicConfig(filename=os.path.join(cf['experiments'],cf['exp_name'],'logfile.log'),level=logging.INFO)
    console = logging.StreamHandler()
    if (log.hasHandlers()):
        log.handlers.clear()
    logging.getLogger('').addHandler(console)

