__all__ = ['get_delta_act', 'convert_smiles', 'MolBatcher']

from typing import Callable, List
import numpy as np
from fastcore.foundation import chunked, L
from fastcore.parallel import ProcessPoolExecutor
from rdkit import Chem
from tqdm import tqdm


def get_delta_act(x):
    delta = np.abs(np.max(x) - np.min(x))
    if delta == 0.0:
        return 'Not duplicate'
    elif delta >= 1.0:
        return 'to_keep'
    elif delta < 1.0 and delta != 0:
        return 'to_merge'


def convert_smiles(mol, sanitize=False):
    if isinstance(mol, str):
        try:
            mol = Chem.MolFromSmiles(mol, sanitize=sanitize)
            return mol
        except:
            return None
    elif isinstance(mol, Chem.Mol):
        return mol


class MolBatcher:
    """Same as Python's ProcessPoolExecutor

    This class splits a dataset of SMILES into chunks
    a function can be called on each chunk while running
    in parallel.

    Attributes
    ----------
    func
        A function with arguments ´func(idxs, smiles_list)´,
        where ´idxs´ is a list of indices and ´smiles_list´
        is a list of SMILES to be transformed.
    smiles_list
        A list of SMILES to be transformed.
    chunk_size
        Number of chunks to divide ´smiles_list´ into.
    n_workers
        Number of parallel processes to start.
    pause
        Number of seconds to wait before processing a batch.
    kwargs
        kwargs to be passed to ´func´

    """

    def __init__(self, func: Callable,
                 smiles_list: List[str],
                 chunk_size: int = 100,
                 n_workers: int = 1,
                 pause: int = 0,
                 **kwargs):
        self.func = func
        self.smiles_list = smiles_list
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.pause = pause
        self.kwargs = kwargs

    def __len__(self):
        return len(self.smiles_list)

    def __iter__(self):
        idxs = L.range(self.smiles_list)
        chunks = L(chunked(idxs, chunk_sz=self.chunk_size))
        with ProcessPoolExecutor(self.n_workers, pause=self.pause) as ex:
            yield from tqdm(ex.map(self.func,
                                   chunks,
                                   smiles_list=self.smiles_list,
                                   **self.kwargs),
                            total=len(chunks))

#
# def return_one(smi):
#     return smi + 'CCC'
#
# def return_same(idxs, smiles_list):
#     res = [return_one(smiles_list[i]) for i in idxs]
#     return res
#
#
# smiles_list = ['CCCC', 'CO']
#
# batcher = MolBatcher(func=return_same, smiles_list=smiles_list, chunk_size=2, n_workers=1, pause=0)
# print(list(batcher))