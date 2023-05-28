# __all__ = ['mol_to_inchi', 'add_nitrogen_charges', 'remove_unwanted',
#            'normalize_mol', 'get_stereo_info', 'process_stereo_duplicates', 'MolCleaner']

from typing import List
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Atom
from chemtools.utils import MolBatcher, convert_smiles, get_delta_act
from tqdm import tqdm
import logging

tqdm.pandas()
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

RDLogger.DisableLog('rdApp.*')

_allowed_atoms = [Atom(i) for i in
                  [1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
                   35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 55, 56, 72, 73, 74, 75, 76, 77, 78,
                   79, 80, 81, 82, 83, 84, 87, 88]]


def get_stereo_info(mol):
    bonds = {Chem.BondStereo.STEREONONE: 0,
             Chem.BondStereo.STEREOANY: 1,
             Chem.BondStereo.STEREOZ: 2,
             Chem.BondStereo.STEREOE: 3,
             Chem.BondStereo.STEREOCIS: 4,
             Chem.BondStereo.STEREOTRANS: 5
             }
    labels = {2: 'Z', 3: 'E', 4: 'CIS', 5: 'TRANS'}
    chiral_centers = ''
    cis_trans = ''

    try:
        mol = convert_smiles(mol, sanitize=True)
        chiral_centers = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True,
                                                   useLegacyImplementation=False)

        for bond in mol.GetBonds():
            bond_id = bonds[bond.GetStereo()]
            if bond_id > 1:
                stereo_str = f'{bond.GetBeginAtomIdx()}_{labels[bond_id]}_{bond.GetEndAtomIdx()}'
                cis_trans += stereo_str

        centers = []
        for center in chiral_centers:
            st = ''.join(map(str, center))
            centers.append(st)
        if centers and cis_trans != '':
            return '_'.join(centers) + '|' + cis_trans
        elif not centers and cis_trans != '':
            return cis_trans
        elif centers and cis_trans == '':
            return '_'.join(centers)
        elif not centers and cis_trans == '':
            return 'No stereo flag'
    except RuntimeError:
        return 'RDKIT failed to find chiral centers or cis/trans flags.'


def mol_to_inchi(mol):
    """Converts a rdchem.Mol object into InCHI keys"""
    mol = convert_smiles(mol, sanitize=True)
    inchi = Chem.MolToInchiKey(mol)
    return inchi


def add_nitrogen_charges(mol):
    """Fixes charge on nitrogen if its valence raises an Exception on RDKit
    See the discussion on: https://github.com/rdkit/rdkit/issues/3310
    """

    mol.UpdatePropertyCache(strict=False)
    ps = Chem.DetectChemistryProblems(mol)
    if not ps:
        Chem.SanitizeMol(mol)
        return mol
    for p in ps:
        if p.GetType() == 'AtomValenceException':
            at = mol.GetAtomWithIdx(p.GetAtomIdx())
            if at.GetAtomicNum() == 7 and at.GetFormalCharge() == 0 and at.GetExplicitValence() == 4:
                at.SetFormalCharge(1)
    Chem.SanitizeMol(mol)
    return mol


def remove_unwanted(mol, allowed_atoms=_allowed_atoms, use_druglike: bool = True):
    """Remove molecules with unwanted elements (check the _unwanted definition) and isotopes"""

    if use_druglike:
        allowed_atoms = [Atom(i) for i in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]]

    remover = rdMolStandardize.AllowedAtomsValidation(allowed_atoms)

    if len(remover.validate(mol)) != 0:
        return None

    if sum([atom.GetIsotope() for atom in mol.GetAtoms()]) != 0:
        return None

    return mol


def normalize_mol(mol):
    """
    Standardize a rdchem.Mol object.
    
    Arguments:
    
        mol : rdchem.Mol
        
    
    Returns:
        mol : rdchem.Mol
            A standardized version of `mol`.
            
    Steps:
    
    1) Convert a SMILES to a rdchem.Mol object or return `mol` if it's already a rdchem.Mol
    
    2) Kekulize, check valencies, set aromaticity, conjugation
     and hybridization, remove hydrogens,
     disconnect metal atoms, normalize the molecule and reionize it.

    3) Get parent fragment if multiple molecules (e.g. mixtures) are present
    
    4) Neutralize parent molecule
    
    
    """
    mol = convert_smiles(mol, sanitize=False)

    # Correction of nitro groups
    patt = Chem.MolFromSmarts('[O-]N(=O)')
    if mol.HasSubstructMatch(patt):
        mol = add_nitrogen_charges(mol)

    mol = remove_unwanted(mol)  # Remove unwanted atoms

    # removeHs, disconnect metal atoms, normalize the molecule, reionize the molecule
    clean_mol = rdMolStandardize.Cleanup(mol)

    # if many fragments, get the "parent" (the actual mol we are interested in) 
    parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)

    # try to neutralize molecule
    uncharger = rdMolStandardize.Uncharger()  # annoying, but necessary as no convenience method exists
    uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)

    return uncharged_parent_clean_mol


def process_stereo_duplicates(data: pd.DataFrame,
                              smiles_column: str,
                              act_column: str,
                              cols_to_check: List[str],
                              keep: str = 'first'):
    assert keep in ['first', 'last']
    data = data.copy()

    # Generate inchikeys and stereo info
    logger.info(f'Converting {smiles_column} into inchikeys.')
    data['inchikey'] = data[smiles_column].progress_apply(mol_to_inchi)
    data['Stereo'] = data[smiles_column].progress_apply(lambda x: get_stereo_info(x))

    c = [smiles_column] + cols_to_check

    # Get potential duplicates
    data['duplicated'] = data.groupby(c)[act_column].transform(get_delta_act)

    to_keep = data[data['duplicated'] == 'to_keep']
    to_merge = data[data['duplicated'] == 'to_merge']
    no_duplicates = data[~data['duplicated'].isin(['to_merge', 'to_keep'])]

    num_potential_duplicates = len(no_duplicates)

    if num_potential_duplicates == len(data):
        logger.info('No duplicates found')
        return data

    to_keep.reset_index(drop=True, inplace=True)
    to_merge.reset_index(drop=True, inplace=True)
    no_duplicates.reset_index(drop=True, inplace=True)

    logger.info(f'Duplicates to merge (delta act < 1.0) : {len(to_merge)}')
    logger.info(f'Duplicates to keep (delta act >= 1.0 as separate entries: {len(to_keep)}')

    # Collect results
    results = []

    if not to_merge.empty:
        logger.info(f'Merging potential duplicates - '
                    f'This is usually caused by isomers with undefined chiral centers or E/Z information on SMILES.')

        merged = to_merge.groupby(c).agg({'inchikey': lambda x: x[0],
                                          'Stereo': lambda x: x[0],
                                          act_column: 'median',
                                          'duplicated': lambda x: x[0]})

        logger.info(f'Number of duplicates merged: {len(merged)} out of {len(to_merge)}.')
        results.append(merged)

    if not to_keep.empty and keep == 'first':
        logger.info(f'Check which duplicates should be kept -'
                    f' This usually happens with isomers with very different activities.')

        duplis_kept = to_keep.groupby(smiles_column, group_keys=False).apply(
            lambda x: x.loc[x[act_column].idxmax()]).copy().reset_index(drop=True)

        duplis_kept.reset_index(drop=True, inplace=True)
        results.append(duplis_kept)

    # Recombine data

    recombined_data = pd.concat([no_duplicates, *results], axis=0, ignore_index=True)
    logger.info(f'Duplicates removal reduced the number of rows from {len(data)} to {len(recombined_data)}')
    recombined_data.reset_index(drop=True, inplace=True)

    return recombined_data


class MolCleaner:

    @classmethod
    def process_mol(cls, mol):
        """Fully process one molecule"""

        mol = convert_smiles(mol)

        try:
            # Remove salts and molecules with unwanted elements (See _allowed_atoms definition above)
            mol = normalize_mol(mol)

            if isinstance(mol, Chem.Mol):
                smi = Chem.MolToSmiles(mol)
                inchikey = mol_to_inchi(mol)
                stereo = get_stereo_info(mol)

                return np.array([smi, inchikey, stereo]).reshape(1, 3)
            else:
                return np.array([None, None, None]).reshape(1, 3)

        except Exception:

            return np.array([None, None, None]).reshape(1, 3)

    @classmethod
    def process_smiles_list(cls, idxs, smiles_list: List[str]):

        res = [cls.process_mol(smiles_list[i]) for i in idxs]
        return res

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                smiles_column: str,
                output_column: str = 'RDKIT_SMILES',
                chunk_size: int = 64,
                n_workers: int = 1,
                pause: int = 0):

        data = df.copy()
        data.reset_index(drop=True, inplace=True)

        logger.info('Sanitizing dataset.')
        logger.info(f'Number of SMILES: {len(data)}.')
        logger.info(f'Input column: {smiles_column}.')
        logger.info(f'Output column: {output_column}.')

        batcher = MolBatcher(cls.process_smiles_list,
                             smiles_list=data[smiles_column].values,
                             chunk_size=chunk_size,
                             n_workers=n_workers,
                             pause=pause)

        results = pd.DataFrame(np.concatenate(list(batcher), axis=0).squeeze(),
                               columns=[output_column, 'inchikey', 'Stereo'])

        data = pd.concat([data, results], axis=1)

        logger.info('Removing unprocessed SMILES.')
        data.dropna(subset=[output_column], inplace=True)
        data.reset_index(drop=True, inplace=True)

        logger.info(f'Finished removing {len(df) - len(data)} failed SMILES.')

        return data

#
# if __name__ == '__main__':
#     data = pd.read_csv('/home/marcossantana/DL/data/Lipophilicity.csv')
#
#     smiles_column = 'smiles'
#     act_column = 'exp'
#     # print(data.head())
#
#     res = MolCleaner.from_df(data, smiles_column=smiles_column, n_workers=3, pause=0, chunk_size=100)
#     print(res[[smiles_column, 'RDKIT_SMILES']].head())
