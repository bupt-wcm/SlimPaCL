from functools import reduce

import numpy as np
import prettytable as pt


def count_params(net, depth):
    tb = pt.PrettyTable()
    tb.field_names = ['Modules', 'Type', 'Params']

    for name, module in net.named_modules():
        if len(name.split('.')) > depth:
            continue
        num_p = 0
        for p in module.parameters():
            num_p += reduce(lambda x, y: x * y, np.array(p.shape))
        tb.add_row([name, type(module), num_p])
    return tb
