from typing import Dict, Tuple, List
from torch.utils.data import DataLoader, Dataset
from chemtools.tools.featurizer import MolFeaturizer
import torch

__all__ = ['MolDataset', 'MolDataLoader', 'DataLoaders']


class MolDataset(Dataset):
    """
    Creates a dataset of chemical data
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
            The index of the element in ´self.data´ to return
        Returns
        -------
        x, target
            A tuple of features and target variable
        """
        target = torch.tensor(self.data[idx][1]).to(torch.float32).view(-1, 1)
        smi = self.data[idx][0]

        x = torch.from_numpy(self.featurizer.transform_one(smi)).to(torch.float32).squeeze()
        return x, target


class DataLoaders:
    def __init__(self, dataloaders: List[DataLoader]):
        if len(dataloaders) == 3:
            self.train_dl, self.valid_dl, self.test_dl = dataloaders
        elif len(dataloaders) == 2:
            self.train_dl, self.test_dl = dataloaders


class MolDataLoader(DataLoader):

    def __call__(self,
                 datasets: Tuple[MolDataset],
                 batch_size: int = 32,
                 shuffle: bool = True,
                 collate_fn=None,
                 drop_last: bool = True):

        if collate_fn is None:
            raise ValueError('The collate function is invalid. Please pass a valid function.')

        train_shuffle = shuffle
        valid_shuffle = not train_shuffle
        dls = []

        if len(datasets) == 2:

            train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle, collate_fn=collate_fn,
                                  drop_last=drop_last)
            test_dl = DataLoader(datasets[1], batch_size=batch_size, shuffle=valid_shuffle, collate_fn=collate_fn,
                                 drop_last=drop_last)
            dls.extend([train_dl, test_dl])

        elif len(datasets) == 3:
            train_dl = DataLoader(datasets[0], batch_size=batch_size, shuffle=train_shuffle, collate_fn=collate_fn,
                                  drop_last=True)
            valid_dl = DataLoader(datasets[1], batch_size=batch_size, shuffle=valid_shuffle, collate_fn=collate_fn,
                                  drop_last=drop_last)
            test_dl = DataLoader(datasets[2], batch_size=batch_size, shuffle=valid_shuffle, collate_fn=collate_fn,
                                 drop_last=drop_last)
            dls.extend([train_dl, valid_dl, test_dl])

        dls = DataLoaders(dls)

        return dls
