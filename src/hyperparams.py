class Hyperparams:
    model_type = 'alexnet' # [alexnet | resnet]
    batch_size = 32
    img_height = 227
    img_width = 227
    channel = 3

    num_epochs = 10

    weight_detect = 1
    weight_landmarks = 5
    weight_visibility = 0.5
    weight_pose = 5
    weight_gender = 2

    save_after_steps = 200
    print_after_steps = 1
    log_after_steps = 10
    val_after_steps = 100
