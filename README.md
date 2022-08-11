# PaCL: Part-level Contrastive Learning for Fine-grained Few-shot Image Classification

## Requirements
- PyTorch >= 1.8
- torchvision
- easydict
- numpy
- pyyaml
- prettytable
- tqdm

## Folder Structure
```
pytorch-template/
│
├── train_exp.py - main script to start training
├── valid_exp.py - evaluation of trained model
│
├── config/ - configure files for the experiments
│   ├── conv-idt.yml - a basic config file
│
├── data/ - code for creating data loader
│   ├── source_data - scripts to create cached data
│   ├── collect_fn  - utils for a fast dataloader
│   ├── fs_dataset  - class file for few-shot dataset
│   ├── fs_sampler  - sampler for the dataloader to sample episodes
│   ├── transforms  - image transform functions
│
├── libs/ - some utils for the experiments
│   ├── checkpoint - save model and its weights
│   ├── count_params - calculate the number of parameters in the model
│   ├── init_exp  - create some basic info for the exp.
│   ├── lr_scheduler  - a learning rate scheduler with warm up
│
├── model/ - some utils for the experiments
│   ├── backbone - used backbone for the model
│   ├── modules - IDT, PaProCL, and some metric-based classifiers for PaCL
│   ├── pacl_net.py  - class file for the pacl-net
│   ├── pacl_model.py  - control the training process of pacl-net
│   ├── utils.py  - some utils for the model
```

## Usage
1. create cached data
```bash
cd data/source_data/ucsd_cub_200
python generate_file.py generate_file --mode 'cl' --data_path 'path/to/cub-200-2011'
```
2. modify values in `conv-idt.yml`
3. run the Experiment
```bash
python train_exp.py --config ./config/conv-idt.yml
```

## Citation
If you find this paper or our code useful in your research, please consider citing:

