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
from glob import glob

import pandas as pd
from PIL import Image
from tqdm import tqdm


def generate_file(data_path):
    dog_im_path = data_path
    csv_path = './splits'
    train_cls_num = 70
    valid_cls_num = 20
    test_cls_num = 30

    # cls_dirs = [join(dog_im_path, dir_path) for dir_path in os.listdir(dog_im_path)]
    # cls_dirs = sorted(cls_dirs, key=lambda x: int(x.split('/')[-1].split('.')[0]))

    train_df = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(csv_path, 'test.csv'))
    tmp_list = []

    tmp_list.append(
        [os.path.join(dog_im_path, item) for item in set(train_df['label'])]
    )
    tmp_list.append(
        [os.path.join(dog_im_path, item) for item in set(valid_df['label'])]
    )
    tmp_list.append(
        [os.path.join(dog_im_path, item) for item in set(test_df['label'])]
    )
    print(tmp_list)
    cls_dirs = tmp_list[0] + tmp_list[1] + tmp_list[2]
    global_cls_ind = 0
    pickled_obj = {}

    for name, cls_num in zip(['train', 'valid', 'test'], [train_cls_num, valid_cls_num, test_cls_num]):
        local_img_ind = 0
        local_cls_ind = 0
        local_dirs = cls_dirs[global_cls_ind: global_cls_ind + cls_num]

        local_file_list = []
        local_cls_meta = {}

        for cls in tqdm(local_dirs, desc='Generate %s DataList' % name):
            im_paths = glob(cls + '/*')

            for item in im_paths:
                with Image.open(item) as im:
                    im = im.convert('RGB')
                    if im.height > im.width:
                        factor = 96 / im.width
                    else:
                        factor = 96 / im.height
                    im = im.resize((int(im.width * factor), int(im.height * factor)), resample=Image.BILINEAR)
                    local_file_list.append((im, global_cls_ind))
            local_cls_meta[local_cls_ind] = list(range(local_img_ind, local_img_ind + len(im_paths)))

            global_cls_ind += 1
            local_img_ind += len(im_paths)
            local_cls_ind += 1

            print(global_cls_ind, local_img_ind)

        pickled_obj[name] = {
            'label_names': [item.split('/')[-1] for item in local_dirs],
            'file': local_file_list,
            'cls_meta': local_cls_meta
        }
    with open('../pickle_data/dog_cls_info-%d-%d-%d.pkl' % (train_cls_num, valid_cls_num, test_cls_num), 'wb') as f:
        pickle.dump(pickled_obj, f)


if __name__ == '__main__':
    import fire

    fire.Fire()
