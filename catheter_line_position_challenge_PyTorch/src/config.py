class Config:
    debug = False
    data_path = 'data'
    train_path = 'data/train'
    test_path = 'data/test'
    output_dir = 'output'
    seed = 42

    size = 400

    model_name = 'resnext50_32x4d'

    n_folds = 4
    batch_size = 8
    epochs = 6
    num_workers = 0

    lr=1e-4
    min_lr=1e-6
    weight_decay=1e-6
    T_max=6

    target_cols = ['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
                 'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal', 
                 'CVC - Abnormal', 'CVC - Borderline', 'CVC - Normal',
                 'Swan Ganz Catheter Present']

    target_size = 11

    train = True

    train_folds=[0, 1, 2, 3]