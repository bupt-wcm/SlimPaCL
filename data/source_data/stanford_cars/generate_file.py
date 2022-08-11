import os
import pickle

import pandas as pd
from PIL import Image
from tqdm import tqdm


def generate_file(data_path):
    train_cls_num = 130
    valid_cls_num = 17
    test_cls_num = 49

    csv_path = './splits'
    train_df = pd.read_csv(os.path.join(csv_path, 'train.csv'))
    valid_df = pd.read_csv(os.path.join(csv_path, 'val.csv'))
    test_df = pd.read_csv(os.path.join(csv_path, 'test.csv'))

    name_cls_dict = {}
    for n, v in zip(train_df['filename'], train_df['label']):
        if v not in name_cls_dict:
            name_cls_dict[v] = []
        name_cls_dict[v].append(n)
    for n, v in zip(valid_df['filename'], valid_df['label']):
        if v not in name_cls_dict:
            name_cls_dict[v] = []
        name_cls_dict[v].append(n)

    for n, v in zip(test_df['filename'], test_df['label']):
        if v not in name_cls_dict:
            name_cls_dict[v] = []
        name_cls_dict[v].append(n)

    tmp_list = []
    tmp_list.append(
        list(set(train_df['label']))
    )
    tmp_list.append(
        list(set(valid_df['label']))
    )
    tmp_list.append(
        list(set(test_df['label']))
    )
    cls_dirs = tmp_list[0] + tmp_list[1] + tmp_list[2]
    print(tmp_list)
    global_cls_ind = 0
    pickled_obj = {}

    for name, cls_num in zip(['train', 'valid', 'test'], [train_cls_num, valid_cls_num, test_cls_num]):
        local_img_ind = 0
        local_cls_ind = 0
        local_dirs = cls_dirs[global_cls_ind: global_cls_ind + cls_num]

        local_file_list = []
        local_cls_meta = {}

        for cls in tqdm(local_dirs, desc='Generate %s DataList' % name):
            im_paths = name_cls_dict[cls]

            for item in im_paths:
                with Image.open(os.path.join(data_path, item)) as im:
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
            'label_names': local_dirs,
            'file': local_file_list,
            'cls_meta': local_cls_meta
        }
    with open('../pickle_data/car_cls_info-%d-%d-%d.pkl' % (train_cls_num, valid_cls_num, test_cls_num), 'wb') as f:
        pickle.dump(pickled_obj, f)


if __name__ == '__main__':
    import fire

    fire.Fire()
