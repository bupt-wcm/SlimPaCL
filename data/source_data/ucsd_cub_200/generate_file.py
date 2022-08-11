'''
input : cub_im_path: /path/to/image/dir/of/CUB200
outputs:
    for train, valid, and test:
        file_list: all image files with one index
        mete_cls: a dict contains all classes and its image indexes in the file list

    for simplicity, we use one pickled file with only one object to serialize all these files
    pickled_obj:
        for each train, valid, test.
        file: image_path and its label (range from 0 to cls_num-1 of cub dataset)
        cls_meta: key is the index in each train/valid/test, and its value is the index of the image in the dependent set.
'''

import os
import pickle

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def generate_splits(mode='dn4'):
    if mode == 'dn4':
        cls_index = []
        cls_number = [130, 20, 50]
        csv_path = './splits_dn4'
        for name in ['train', 'val', 'test']:
            df = pd.read_csv(os.path.join(csv_path, name + '.csv'))
            cls_index.extend(np.unique([int(item[:3]) for item in df['label']]).tolist())
        cls_index = [item - 1 for item in cls_index]
    elif mode == 'cl':  # CloserLook
        cls_number = [100, 50, 50]
        cls_index = range(200)
        tmp_list = [[], [], []]
        for idx, tmp in enumerate(cls_index):
            if idx % 2 == 0:
                tmp_list[0].append(tmp)
            elif idx % 4 == 1:
                tmp_list[1].append(tmp)
            elif idx % 4 == 3:
                tmp_list[2].append(tmp)
        cls_index = tmp_list[0] + tmp_list[1] + tmp_list[2]

    return cls_index, cls_number


def generate_file(data_path, mode, with_box=False):
    # Read Source Data
    bbox_list = []
    with open(data_path + '/bounding_boxes.txt', 'r') as f:
        for line in f.readlines():
            bbox_list.append([float(x) for x in line.split(' ')])

    ipath_list = []
    with open(data_path + '/images.txt', 'r') as f:
        for line in f.readlines():
            ipath_list.append(line.split(' ')[1])

    cls_dir = []
    with open(data_path + '/classes.txt', 'r') as f:
        for line in f.readlines():
            cls_dir.append(line.split(' ')[1].strip())

    img2cls = []
    with open(data_path + '/image_class_labels.txt', 'r') as f:
        for line in f.readlines():
            img2cls.append(int(line.split(' ')[1]) - 1)

    cls_index, cls_num = generate_splits(mode=mode)
    train_cls_num, valid_cls_num, test_cls_num = cls_num
    pickled_obj = {}
    for name in ['train', 'valid', 'test']:
        pickled_obj[name] = {
            'label_names': [],
            'file': [],
            'cls_meta': {}
        }

    for im_idx, im_path in tqdm(enumerate(ipath_list)):
        im_cls = img2cls[im_idx]
        im_bbox = bbox_list[im_idx]
        with Image.open(os.path.join(data_path, 'images', im_path.strip())) as im:
            im = im.convert('RGB')
            if with_box:
                x, y, width, height = im_bbox[1:]
                if width < height:
                    x = x - (height - width) // 2
                    width = height
                if height < width:
                    y = y - (width - height) // 2
                    height = width
                x1, y1, x2, y2 = x, y, x + width, y + height
                im.crop([x1, y1, x2, y2])

                if im.height > im.width:
                    factor = 256 / im.width
                else:
                    factor = 256 / im.height
                im = im.resize((int(im.width * factor), int(im.height * factor)), resample=Image.BILINEAR)

            if cls_index.index(im_cls) < train_cls_num:
                pickled_obj['train']['file'].append((im, im_cls))
                local_cls_index = cls_index.index(im_cls)
                if local_cls_index not in pickled_obj['train']['cls_meta'].keys():
                    pickled_obj['train']['cls_meta'][local_cls_index] = []
                pickled_obj['train']['cls_meta'][local_cls_index].append(len(pickled_obj['train']['file']) - 1)
            elif cls_index.index(im_cls) < train_cls_num + valid_cls_num:
                pickled_obj['valid']['file'].append((im, im_cls))
                local_cls_index = cls_index.index(im_cls) - train_cls_num
                if local_cls_index not in pickled_obj['valid']['cls_meta'].keys():
                    pickled_obj['valid']['cls_meta'][local_cls_index] = []
                pickled_obj['valid']['cls_meta'][local_cls_index].append(len(pickled_obj['valid']['file']) - 1)
            else:
                pickled_obj['test']['file'].append((im, im_cls))
                local_cls_index = cls_index.index(im_cls) - train_cls_num - valid_cls_num
                if local_cls_index not in pickled_obj['test']['cls_meta'].keys():
                    pickled_obj['test']['cls_meta'][local_cls_index] = []
                pickled_obj['test']['cls_meta'][local_cls_index].append(len(pickled_obj['test']['file']) - 1)

    with open('../pickle_data/cub_cls_info-%d-%d-%d-256x256.pkl' % (train_cls_num, valid_cls_num, test_cls_num),
              'wb') as f:
        pickle.dump(pickled_obj, f)


if __name__ == '__main__':
    import fire

    fire.Fire()
