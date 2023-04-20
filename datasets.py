from torch.utils.data import Dataset
import os
import torch


class TensorDataSet(Dataset):
    def __init__(self, dataset_dir, transforms=None):

        self. paths = [os.path.join(dataset_dir, file_name) for file_name in os.listdir(dataset_dir)
                       if file_name.endswith('.pt')]
        self.transforms = transforms

    def __getitem__(self, index):
        cur_path = self.paths[index]
        saved_data = torch.load(cur_path)
        x = saved_data['x']
        if self.transforms is not None:
            x = self.transforms(x)
        y = saved_data['y']
        return x, y

    def __len__(self):
        return len(self.paths)

