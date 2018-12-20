import cv2
import numpy as np

from utils import NetOutput, OutputBatch


class DataLoader(object):
    def __init__(self, path, train_batch_size, test_batch_size):
        data = np.load(path)
        data = data[np.random.permutation(len(data))]
        n_train = int(len(data) * 0.8)

        self.n_train = n_train

        self.data = {
            'train': data[:n_train],
            'test': data[n_train:]
        }

        self.batch_size = {
            'train': train_batch_size,
            'test': test_batch_size,
        }

        self.n_batches = {
            split: (len(self.data[split]) + self.batch_size[split] - 1) / self.batch_size[split]
            for split in ['train', 'test']
        }

        self.batch_index = {
            'train': 0,
            'test': 0
        }

    def next_batch(self, split):
        start_index = self.batch_index[split] * self.batch_size[split]
        end_index = start_index + self.batch_size[split]
        self.batch_index[split] = (self.batch_index[split] + 1) % self.n_batches[split]

        batch = self.data[split][start_index:end_index]

        input_batch = np.array([cv2.resize(img, (227, 227)) for img in batch[:, 0]])[:, :, :, [2, 1, 0]]
        outputs = batch[:, 1:]

        outputs = [NetOutput(*output) for output in outputs]

        detections = np.array([out.detection[:1] for out in outputs])
        landmarks = np.array([out.landmarks for out in outputs])
        poses = np.array([out.pose for out in outputs])
        genders = np.array([out.gender[:1] for out in outputs])
        visibility = np.array([out.visibility for out in outputs])

        return input_batch, OutputBatch(detections, landmarks, visibility, poses, genders)

    def shuffle_train_data(self):
        perm = np.random.permutation(self.n_train)
        self.data['train'] = self.data['train'][perm]
