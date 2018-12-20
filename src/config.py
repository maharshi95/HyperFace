class Config:
    base_path = '../'

    data_path = base_path + 'data/hyf_data.npy'

    models_path = base_path + 'weights'

    log_path = {
        'train': base_path + 'logs/train',
        'test': base_path + 'logs/test',
    }

    best_models_path = base_path + 'best_weights'
