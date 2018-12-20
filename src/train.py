import os

from config import Config
from dataloader import DataLoader
from hyperparams import Hyperparams
from model import HyperFaceModel

C = Config()
H = Hyperparams()

load_weights = False

# Create Required Directories
for path in [C.models_path, C.best_models_path, C.log_path['train'], C.log_path['test']]:
    if not os.path.exists(path):
        os.makedirs(path)

# DataLoader Initialization
dl = DataLoader(C.data_path, H.batch_size, 128)

# Model Creation
model = HyperFaceModel(H, C)
model.build()

min_loss = -1

for i_epoch in range(H.num_epochs):
    n_iters = dl.n_batches['train']

    for iter_no in range(n_iters):
        inputs, outputs = dl.next_batch('train')

        global_step = i_epoch * n_iters + iter_no

        log_summary = global_step % H.log_after_steps == 0

        train_losses = model.train_step(inputs, outputs, global_step, log_summary)

        # Print on console
        if global_step % H.print_after_steps == 0:
            print 'Epoch %d Iter %d Train Loss: %.03f' % (i_epoch + 1, iter_no + 1, train_losses.total)

        # Validation
        if global_step % H.val_after_steps == 0:
            val_inputs, val_outputs = dl.next_batch('test')
            val_losses = model.val_step(inputs, outputs, global_step)
            print 'Epoch %d Iter %d Val Loss: %.03f' % (i_epoch + 1, iter_no + 1, val_losses.total)

            if min_loss == -1 or min_loss < val_losses.total:
                model.save_best_model(global_step)

        # Save Weights
        if global_step % H.save_after_steps == 0:
            model.save_model(global_step)
