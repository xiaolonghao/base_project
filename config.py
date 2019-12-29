# -*- coding: utf-8 -*-
class Default_config(object):

    stage = 'test'
    model = 'resnet50'

    train_data_path = r'D:\Data\datasets\3'
    val_data_path = r'D:\Data\datasets\3'
    test_query_data_path = r'D:\Data\datasets\3'
    test_gallery_data_path = r'D:\Data\datasets\3'
    save_model_path = 'checkpoints'
    load_model_path = 'epoch2'

    seed = 0
    batch_size = 2
    num_workers = 4
    print_info_epoch = 1
    all_epoch = 3
    save_model_epoch = 1
    lr = 0.1
    lr_decay_ratio = 0.1
    lr_decay_step = 2
    weight_decay = 5e-4

    continue_train = False
    display_train_statistics = True
