__all__ = ['MolFeaturizer']

import numpy as np
from rdkit import Chem
from ctypes import ArgumentError
from cheminftools.utils import convert_smiles
from rdkit.Chem import MACCSkeys, rdFingerprintGenerator, Descriptors
from rdkit.Chem.rdReducedGraphs import GetErGFingerprint
from rdkit.DataStructs.cDataStructs import ConvertToNumpyArray
from functools import partial
from typing import List
from tqdm import tqdm


class MolFeaturizer:
    """
    Creates a featurizer object to perform molecular transformations
    on SMILES.

    Attributes
    ----------
    params
        A dictionary of parameters for an rdkit generator.
    descriptor_type
        A string representing a descriptor available in ´rdFingerprintGenerator´
        Available descriptors are morgan, atom_pairs, rdkit, rdkit2d, torsion and maccs, erg
    generator
        A fingerprinter generator available in ´rdFingerprintGenerator´

          """

    def __init__(self, descriptor_type: str, params: dict = {}):

        self.params = params
        self.descriptor_type = descriptor_type

        self.DESCS = {'morgan': rdFingerprintGenerator.GetMorganGenerator,
                      'atom_pairs': rdFingerprintGenerator.GetAtomPairGenerator,
                      'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator,
                      'rdkit2d': self.get_rdkit2d_descriptors,
                      'torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator,
                      'maccs': MACCSkeys.GenMACCSKeys,
                      'erg': GetErGFingerprint}

        if descriptor_type in ['morgan', 'atom_pairs', 'rdkit', 'torsion']:
            self.generator = self.set_params(self.DESCS[descriptor_type], params)
        else:
            self.generator = self.DESCS[descriptor_type]

    def set_params(self, generator, params: dict):

        """
        Set parameters ´params´ for ´generator´
        
        """

        try:
            generator = generator(**params)

        except ArgumentError:
            print(
                f'The parameters {params} are not valid for generator {self.DESCS[self.descriptor_type].__name__}.'
                f'\nSee RDKit: https://www.rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html')
            print('Returning the generator with default parameters.')
            generator = generator()

        return generator

    def transform_one(self, smi: str, use_counts: bool = False) -> np.array:

        """
        Generate features for one SMILES.
        
        Arguments
        ---------
        smi
            A SMILES representing a molecular structure
        use_counts
            Whether to consider feature's counts for fingerprint generation.

        """
        mol = convert_smiles(smi, sanitize=True)

        if not mol:
            return None

        if self.descriptor_type == 'erg':
            return self.generator(mol).reshape(1, -1)

        if self.descriptor_type == 'maccs':
            fps = np.array([])
            ConvertToNumpyArray(self.generator(mol), fps)
            return fps.reshape(1, -1)

        elif self.descriptor_type == 'rdkit2d':
            return self.generator(mol)

        else:
            if self.descriptor_type == 'morgan' and use_counts:
                fps = self.generator.GetCountFingerprintAsNumPy(mol)
                return fps.reshape(1, -1)

            fps = self.generator.GetFingerprintAsNumPy(mol)
            return fps.reshape(1, -1)

    def transform(self, smiles_list: List[str], use_counts: bool = False) -> np.array:

        """
        Generate features for a list of SMILES.
        
        Arguments
        ---------   
        use_counts
            Whether to use count during fingerprint calculation.
        smiles_list
            A list of SMILES.
            
        Returns
        -------  
        fps
            A fingerprint array.
                                    
        """

        func = partial(self.transform_one, use_counts=use_counts)
        fps = list(tqdm(map(func, smiles_list), total=len(smiles_list)))

        if len(fps) > 1:
            return np.vstack(fps)
        return fps[-1]

    def get_rdkit2d_descriptors(self, mol: Chem.rdchem.Mol):

        """
        Generates 200 RDKit constitutional descriptors for a `mol` object.

        Arguments
        ---------
        mol : Chem.rdchem.Mol
            A RDKit Mol object.

        Returns
        -------
        descs : numpy.array
            An array with the calculated descriptors.


        """
        descriptor_dict = {name: func for name, func in Descriptors.descList}
        descs = np.array([func(mol) for name, func in descriptor_dict.items()]).reshape(1, -1)
        return descs
