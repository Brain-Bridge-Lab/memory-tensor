"""This file implements the memory game, reframing the memorability task as a decision making game with imperfect
information. """

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from csv import reader
import matplotlib.pyplot as plt


class MemCatDataset(Dataset):
    def __init__(self, loc='./Sources/memcat/', transform=transforms.ToTensor()):
        self.loc = loc
        self.transform = transform
        with open(f'{loc}data/memcat_image_data.csv', 'r') as file:
            r = reader(file)
            next(r)
            data = [d for d in r]
            self.memcat_frame = np.array(data)

    def __len__(self):
        return len(self.memcat_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.memcat_frame[idx, 1]
        cat = self.memcat_frame[idx, 2]
        scat = self.memcat_frame[idx, 3]
        img = Image.open(f'{self.loc}images/{cat}/{scat}/{img_name}').convert('RGB')
        y = self.memcat_frame[idx, 12]
        y = torch.Tensor([float(y)])
        image_x = self.transform(img)
        return [image_x, y, img_name]


class LamemDataset(Dataset):
    def __init__(self, loc='./Sources/lamem/', transform=transforms.ToTensor()):
        self.lamem_frame = np.array(np.loadtxt(f'{loc}splits/full.txt', delimiter=' ', dtype=str))
        self.loc = loc
        self.transform = transform

    def __len__(self):
        return self.lamem_frame.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.lamem_frame[idx, 0]
        image = Image.open(f'{self.loc}/images/{img_name}')
        image = image.convert('RGB')
        y = self.lamem_frame[idx, 1]
        y = torch.Tensor([float(y)])
        image_x = self.transform(image)
        return [image_x, y, img_name]


class MemGym:
    def __init__(self, dataset, periods=100):
        self.dataset = dataset
        self.datasize = len(dataset)
        self.total = set(range(self.datasize))
        self.done = False
        self.targets = set(np.random.randint(0, self.datasize, 30))
        self.fillers = self.total - self.targets
        self.imgidx = None
        self.seen = set()
        self.t_last_target = 0
        self.t = 0
        self.periods = periods

    def reset(self):
        self.done = False
        self.targets = set(np.random.randint(0, self.datasize, 30))
        self.fillers = self.total - self.targets
        self.seen = set()
        self.t = 0
        self.imgidx = np.random.choice(tuple(self.targets))
        self.t_last_target = 0
        return self.dataset[self.imgidx][0], 0, False, {}

    def step(self, action):
        if self.done or self.t >= self.periods:
            self.done = True
            return None, 0, True, {}
        else:
            actual = self.imgidx in self.seen
            score = actual and action
            if self.imgidx in self.targets:
                self.seen.add(self.imgidx)
            if self.t - self.t_last_target >= 30:
                it_time = np.random.choice([True, False], p=[1/3, 2/3])
            else:
                it_time = False
            if it_time:
                self.imgidx = np.random.choice(tuple(self.targets))
                self.t_last_target = self.t
            else:
                self.imgidx = np.random.choice(tuple(self.fillers))
            self.t += 1
            return self.dataset[self.imgidx][0], int(score), False, {}


if __name__ == '__main__':
    env = MemGym(MemCatDataset(), periods=1000)
    img, score, done, _ = env.reset()
    sc = 0
    resp = False
    seen = set()
    while not done:
        img, score, done, _ = env.step(resp)
        if img is not None:
            seen.add(img)
            resp = img in seen
            sc += score
    print(sc)
