from __future__ import print_function
import torch
import scipy.io
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)
import torch.utils.data
import numpy as np
import scipy.io
import random


class BCICompIV2b:
    def __init__(self, root, sub, memory, sub_id, train=True):
        """
        Initialize the dataset loader.

        Args:
            root (str): Root directory of the dataset.
            sub (int): Subject number.
            memory (dict): Memory object for subject-specific data.
            sub_id (int): Subject ID for labeling.
            train (bool, optional): Whether to load training data. Defaults to True.
        """
        self.root = root
        self.train = train
        self.sub_id = sub_id

        # Load and preprocess data
        data_dict = scipy.io.loadmat(f"{self.root}B{sub}.mat")
        trainX, trainY = np.array(data_dict['trainX']), np.array(data_dict['trainY']).squeeze()
        self.data = np.transpose(trainX, (2, 1, 0))  # Rearrange dimensions
        self.targets = list(trainY)

        # Initialize labels
        self.sub_module_label = [self.sub_id] * len(self.data)
        self.dis_label = [self.sub_id + 1] * len(self.data)

        # Append memory data if in training mode
        if train and memory is not None:
            self._append_memory(memory)

    def _append_memory(self, memory):
        """Appends memory data to the dataset."""
        for i in range(len(memory['x'])):
            self.data = np.append(self.data, [memory['x'][i]], axis=0)
            self.targets.append(memory['y'][i])
            self.sub_module_label.append(memory['sub_module_label'][i])
            self.dis_label.append(memory['dis_label'][i])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the data point.

        Returns:
            tuple: (data, target, sub_module_label, dis_label) for the given index.
        """
        return (
            self.data[index],
            int(self.targets[index]),
            self.sub_module_label[index],
            self.dis_label[index],
        )

    def __len__(self):
        """Returns the total number of data points."""
        return len(self.data)


class IncrementalDataStreaming(object):
    def __init__(self, args):
        super(IncrementalDataStreaming, self).__init__()
        self.args = args
        self.use_memory = args.use_memory

        self.n_workers = args.n_workers
        self.pin_memory = True
        self.seed = args.seed


        self.batch_size = args.batch_size
        self.pc_valid = args.pc_valid
        self.root = args.data_dir
        self.latent_dim = args.latent_dim
        self.n_samples = args.n_samples
        self.pc_valid = args.pc_valid
        self.pc_test = args.pc_test

        self.sub_cls = [[s, args.n_classes] for s in range(
            args.n_subjects)]

        self.subject_queue = np.arange(1, args.n_subjects + 1).tolist()                           # sequential order of subjects
        # self.subject_queue = np.random.permutation(np.arange(1, args.n_subjects + 1)).tolist()   # randomize the order of subjects
        self.subject_index = [[s] for s in range(args.n_subjects)]
        self.global_reservoir_size = args.global_reservoir_size
        self.dataloaders = {}
        self.train_set = {}
        self.test_set = {}
        self.train_split = {}
        self.valid_split = {}
        self.global_memory = {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []}


    def load_data(self, sub_id):
        self.dataloaders[sub_id] = {}
        sys.stdout.flush()
        if sub_id == 0:
            memory = None
        else:
            memory = self.global_memory

        self.train_set[sub_id] = BCICompIV2b(root=self.root, sub=self.subject_queue[sub_id], memory=memory,
                                             sub_id=sub_id)

        split_valid = int(np.floor(self.pc_valid * len(self.train_set[sub_id])))
        split_test = int(np.floor(self.pc_test * len(self.train_set[sub_id])))
        split_train = len(self.train_set[sub_id]) - split_valid - split_test
        train_split, valid_split, test_split = torch.utils.data.random_split(
            self.train_set[sub_id], [split_train, split_valid, split_test]
        )

        self.train_split[sub_id] = train_split

        train_loader = torch.utils.data.DataLoader(
            self.train_split[sub_id], batch_size=self.batch_size,
            shuffle=True, num_workers=self.n_workers,
            pin_memory=self.pin_memory, drop_last=True
        )
        self.valid_split[sub_id] = valid_split


        valid_loader = torch.utils.data.DataLoader(
            self.valid_split[sub_id], batch_size=int(self.batch_size * self.pc_valid),
            shuffle=False, num_workers=self.n_workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        self.test_set[sub_id] = test_split

        test_loader = torch.utils.data.DataLoader(
            self.test_set[sub_id], batch_size=int(self.batch_size * self.pc_valid),
            shuffle=False, num_workers=self.n_workers,
            pin_memory=self.pin_memory, drop_last=True
        )


        combined_train_valid_dataset = torch.utils.data.ConcatDataset(
            [self.valid_split[sub_id], self.train_split[sub_id]])

        combined_loader = torch.utils.data.DataLoader(
            combined_train_valid_dataset, batch_size=1,
            shuffle=False, num_workers=self.n_workers,
            pin_memory=self.pin_memory, drop_last=True
        )

        self.dataloaders[sub_id]['train'] = train_loader
        self.dataloaders[sub_id]['valid'] = valid_loader
        self.dataloaders[sub_id]['test'] = test_loader
        self.dataloaders[sub_id]['name'] = f'BCI-COMP2B-subject{self.subject_queue[sub_id]}'

        shape = self.train_set[sub_id].data.shape[:]
        print(f"Training set size:      {len(train_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Validation set size:    {len(valid_loader.dataset)}  EEG signals of shape {shape}")
        print(f"Test set size:          {len(test_loader.dataset)}  EEG signals of shape {shape}")

        if self.use_memory == 'yes' and self.n_samples > 0:
            self.update_memory(sub_id)
        return self.dataloaders

    def update_memory(self, sub_id):
        """
        Update global memory using reservoir sampling to retain knowledge across subjects.

        Args:
            sub_id (int): The subject identifier for which to preserve knowledge.
        """
        # Calculate the number of samples to retain per subject
        num_samples_per_subject = self.n_samples // len(self.subject_index[sub_id])

        # Mapping each subject index to its own class label
        mem_class_mapping = {i: i for i in range(len(self.subject_index[sub_id]))}
        reservoir_size = self.global_reservoir_size

        # Initialize global memory if not already initialized
        if 'x' not in self.global_memory:
            self.global_memory = {'x': [], 'y': [], 'sub_module_label': [], 'dis_label': []}

        # Load data for the specified subject
        data_loader = torch.utils.data.DataLoader(self.train_split[sub_id], batch_size=1,
                                                  num_workers=self.n_workers, pin_memory=self.pin_memory)

        # Randomly select indices to sample from
        rand_indices = torch.randperm(len(data_loader.dataset))[:num_samples_per_subject]

        # Add samples to global memory with reservoir sampling
        for i, ind in enumerate(rand_indices):
            # Retrieve sample components
            sample = {
                'x': data_loader.dataset[ind][0],
                'y': mem_class_mapping.get(i, 0),  # Using mem_class_mapping.get to handle missing keys safely
                'sub_module_label': data_loader.dataset[ind][2],
                'dis_label': data_loader.dataset[ind][3]
            }
            self.reservoir_sample(sample, reservoir_size)

        print(f'Global memory updated using reservoir sampling, retaining {len(self.global_memory["x"])} samples')

    def reservoir_sample(self, sample, reservoir_size):
        """
        Helper function to perform reservoir sampling on global memory.

        Args:
            sample (dict): The sample to consider for adding to memory.
            reservoir_size (int): The maximum size of the reservoir memory.
        """
        if len(self.global_memory['x']) < reservoir_size:
            # Append sample if memory is not full
            for key in sample:
                self.global_memory[key].append(sample[key])
        else:
            # Replace a random sample with a small probability if memory is full
            j = random.randint(0, len(self.global_memory['x']) - 1)
            if random.random() < reservoir_size / (reservoir_size + len(self.global_memory['x'])):
                for key in sample:
                    self.global_memory[key][j] = sample[key]
