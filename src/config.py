class Config:
    base_path = '../'

    data_path = base_path + 'data/hyf_data.npy'

    iter_weights_dir = base_path + 'weights_iter/'
    best_weights_dir = base_path + 'weights_best/'

    models_path = iter_weights_dir + 'iter'
    best_models_path = best_weights_dir + 'best'

    log_path = {
        'train': base_path + 'logs/train',
        'test': base_path + 'logs/test',
    }
