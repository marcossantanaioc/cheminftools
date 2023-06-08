from typing import Dict, Tuple
import torch
from torch.utils.data import DataLoader, Dataset, default_collate
from cheminftools.tools.featurizer import MolFeaturizer

__all__ = ['MolDataset', 'MolDataLoader', 'MolDataLoaders']

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'


class MolDataset(Dataset):
    """
    Creates a dataset of chemical examples
    Data should be passed as a list of tuples,
    where each element of the list is a tuple of
    SMILES and a target variable (e.g. ('c1ccccc1', 1.5))

    Attributes
    ----------
    data
        A list of tuples of SMILES and target variables.
    descriptor_type
        Name of the descriptor to use.
        Check ´MolFeaturizer´ documentation.
    params
        A dictionary of parameters to pass to
        MolFeaturizer.

    """

    def __init__(self, data, descriptor_type: str = 'morgan', params: Dict = {}):
        self.data = data
        self.descriptor_type = descriptor_type
        self.params = params
        self.featurizer = MolFeaturizer(self.descriptor_type, params)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple of features and target variable.

        Parameters
        ----------
        idx
            The index of the element in ´self.examples´ to return
        Returns
        -------
        x, target
            A tuple of features and target variable
        """
        target = torch.tensor(self.data[idx][1]).to(torch.float32).view(-1, ).to(DEVICE)
        smi = self.data[idx][0]

        x = torch.from_numpy(self.featurizer.transform_one(smi)).to(torch.float32).squeeze().to(DEVICE)
        return x, target


class MolDataLoaders:
    def __init__(self, *dataloaders: DataLoader):
        self.train_dl, self.valid_dl = dataloaders


class MolDataLoader:

    def __init__(self,
                 datasets: Tuple[MolDataset],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 collate_fn=None,
                 drop_last: bool = True):
        self.datasets = datasets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = default_collate if collate_fn is None else collate_fn
        self.drop_last = drop_last

    def dataloaders(self):
        train_shuffle = self.shuffle
        valid_shuffle = not train_shuffle

        train_dl = DataLoader(self.datasets[0], batch_size=self.batch_size, shuffle=train_shuffle,
                              collate_fn=self.collate_fn,
                              drop_last=self.drop_last)
        valid_dl = DataLoader(self.datasets[1], batch_size=self.batch_size * 2, shuffle=valid_shuffle,
                              collate_fn=self.collate_fn,
                              drop_last=False)

        dls = MolDataLoaders(train_dl, valid_dl)

        return dls
