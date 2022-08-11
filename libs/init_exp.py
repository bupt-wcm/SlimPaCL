import os
import random
import string
from datetime import datetime
from os.path import join

import torch.cuda


def init_exp(data_name, bs, nw, ns, nq, gpu, prefix='./checkpoints', prefix_describe='', postfix_describe=''):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print(torch.cuda.is_available())

    exp_name = "{DataName}".format(
        DataName=data_name
    )

    exp_path = join(prefix, exp_name)
    if not os.path.exists(exp_path):
        os.mkdir(exp_path)

    code_str = string.ascii_letters + string.digits
    time_str = datetime.now().strftime("%Y%m%d%H%M")
    dic_code = random.sample(code_str, 5)
    train_name = prefix_describe
    if postfix_describe is not None:
        train_name += '-' + postfix_describe
    train_name = ''.join(dic_code) + '-' + time_str + '-' + '%dshot' % ns + '-' + train_name

    saved_path = join(exp_path, train_name)
    if os.path.exists(saved_path):
        ioerror_info = 'The Train Name {0} has already been created, which is caused by the `Duplicate Identification Code` {1}\n'.format(
            train_name, dic_code)
        ioerror_info += 'This error can be solved by \n'
        ioerror_info += '\t 1. Re-run the program, because this is a small probability event; \n'
        ioerror_info += '\t 2. Clean the Dir: exp_path, if there contains many log dirs; \n'
        ioerror_info += '\t 3. Adjust the length of the DIC in `./fs_models/ioes/init_exp.py` to generate longer DIC. \n'
        raise IOError(ioerror_info)
    os.mkdir(saved_path)

    exp_info = {
        'exp_name': exp_name,
        'train_name': train_name,
        'saved_path': saved_path
    }

    return exp_info
