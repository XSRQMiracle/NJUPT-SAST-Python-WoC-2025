import os
import random
import numpy as np
import torch
from datetime import datetime


def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def create_output_dir(base_dir='./training_results', prefix=''):
    os.makedirs(base_dir, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')

    existing_dirs = [d for d in os.listdir(base_dir)
                     if os.path.isdir(os.path.join(base_dir, d)) and d.startswith('第')]
    max_num = 0
    for dir_name in existing_dirs:
        try:
            num = int(dir_name.split('次')[0].replace('第', ''))
            max_num = max(max_num, num)
        except:
            continue

    new_num = max_num + 1
    dir_name = f'第{new_num}次_{date_str}'
    output_dir = os.path.join(base_dir, dir_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f'输出目录: {os.path.abspath(output_dir)}')
    return output_dir
