__all__ = ['mol_to_inchi', 'add_nitrogen_charges',
           'remove_unwanted', 'normalize_mol', 'get_stereo_info',
           'process_duplicates', 'MolCleaner']

from typing import List, Union
import re
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import Atom
from cheminftools.utils import MolBatcher, convert_smiles, get_delta_act
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


def check_stereo(smi):
    """
    Check if a SMILES
    has any stereo marks
    such as @ or double bonds.
    Parameters
    ----------
    smi
        a SMILES string
    Returns
    -------
        a boolean indicating whether `smi` has or not stereo marks.
    """
    smi = str(smi)
    double_bond_patt = r'[\\\/]'
    r_s_patt = '@'

    patt = re.compile('|'.join([double_bond_patt, r_s_patt]))
    out = bool(re.search(patt, smi))
    return out


def get_stereo_info(mol: Union[Chem.Mol, str]):
    """
    Return the stereo information on a given molecule.
    Stereo information includes CIS/TRANS, E/Z, R/S isomerism info
    for each bond in a molecule.

    Parameters
    ----------
    mol
        A rdkit.Chem.Mol object or a SMILES string

    Returns
    -------
    stereo_info
        A string representing the stereo info found
        in each bond of ´mol´.
    """

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
    """
    Converts a Chem.Mol object into InCHI keys

    Parameters
    ----------
    mol
        A rdkit.Chem.Mol object or a SMILES string
    """

    mol = convert_smiles(mol, sanitize=True)
    inchi = Chem.MolToInchiKey(mol)
    return inchi


def add_nitrogen_charges(mol):
    """
    Fixes charge on nitrogen if its valence raises an Exception on RDKit
    See the discussion on: https://github.com/rdkit/rdkit/issues/3310

    Parameters
    ----------
    mol
        A rdkit.Chem.Mol object.

    Returns
    -------
    mol
        A sanitized version of ´mol´
        with charges on nitrogen atoms corrected.
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


def remove_unwanted(mol,
                    allowed_atoms: List[Chem.Atom] = _allowed_atoms,
                    use_druglike: bool = True):
    """
    Remove molecules with unwanted elements (check the _unwanted definition) and isotopes.

    Parameters
    ----------
    mol
        A Chem.Mol object.
    allowed_atoms
        A list of Chem.Atom objects representing atoms that are allowed.
    use_druglike
        Whether to use atoms usually found in drugs (halogens, H, C, N, O)

    Returns
    -------
        A Chem.Mol object or None.

    """

    if use_druglike:
        allowed_atoms = [Atom(i) for i in [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]]

    remover = rdMolStandardize.AllowedAtomsValidation(allowed_atoms)

    if len(remover.validate(mol)) != 0:
        return None

    if sum([atom.GetIsotope() for atom in mol.GetAtoms()]) != 0:
        return None

    return mol


def normalize_mol(mol: Union[str, Chem.Mol]) -> Chem.Mol:
    """
    Standardize a rdchem.Mol object.

    Steps:

    1) Convert a SMILES to a rdchem.Mol object or return `mol` if it's already a rdchem.Mol

    2) Kekulize, check valencies, set aromaticity, conjugation
     and hybridization, remove hydrogens,
     disconnect metal atoms, normalize the molecule and reionize it.

    3) Get parent fragment if multiple molecules (e.g. mixtures) are present

    4) Neutralize parent molecule
    
    Parameters
    ----------
    
        mol : Chem.Mol

    Returns
    -------
        mol : Chem.Mol
            A standardized version of `mol`.
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


def process_duplicates(data: pd.DataFrame,
                       smiles_column: str,
                       act_column: str,
                       cols_to_check: List[str],
                       keep: str = 'first'):
    """
    Aggregates duplicates in a dataset.
    It is usually enough to look for duplicates by
    grouping on standardized SMILES or InChiKeys.
    However, in cases where there's a racemic mixture and
    defined isomers present, the standard deduplication
    procedure will return separate entries. When these
    entries are used to compute chemical features, such as
    Morgan fingerprints, MACCS  keys or 2D physicochemical descriptors,
    they will share the same values (unless stereochemistry is taken into
    consideration).
    For cases like the above, one can use ´process_duplicates´ function
    to merge and keep just one compound. This function looks for potential
    duplicates in your SMILES by grouping on the SMILES column and other
    relevant columns (´cols_to_check´).

    Three kinds of duplicates will be treated:
    to_keep: duplicates where the difference in ´act_column´ is larger
    than 1. In case of activity values, it considers 1 log unit
    to_merge: duplicates where the difference in ´act_column´ < 1.0
    no_duplicates: compounds without a duplicate flag.

    Parameters
    ----------
    data
        Input DataFrame.
    smiles_column
        Name of column with SMILES.
    act_column
        Name of column with activity data.
        It doesn't need to be activity - any
        numberical value is ok.
    cols_to_check
        Relevant columns to use for aggregation.
        See pandas groupby operation for detailed
        explanation.
    keep
        Which compound to keep
        first means the one with highest ´act_column´ value.
        last will keep the one with lowest ´act_column´ value.

    Returns
    -------
        Aggregated version of data.

    """
    assert keep in ['first', 'last']
    data = data.copy()

    # Generate inchikeys and stereo info
    logger.info(f'Converting {smiles_column} into inchikeys.')
    data['smiles_no_isomeric'] = data[smiles_column].progress_apply(
        lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=False))
    data['inchikey'] = data['smiles_no_isomeric'].progress_apply(mol_to_inchi)
    data['Stereo'] = data[smiles_column].progress_apply(lambda x: get_stereo_info(x))

    # Remove SMILES without stereo information IF a similar SMILES has stereo info
    # This is the case when unspecified and isolated isomers are present in the dataset

    data['has_stereo_mark'] = data[smiles_column].apply(check_stereo)  # Find
    data['group_size'] = data.groupby('inchikey')[smiles_column].transform('count')
    data = data.drop(data[(data['has_stereo_mark'] == False) & (data['group_size'] > 1)].index)
    data.reset_index(drop=True, inplace=True)

    c = ['inchikey'] + cols_to_check

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
        cols = to_merge.columns
        aggs = {c: lambda x: x[0] for c in cols}  # Aggregate and get first entry on each group.
        aggs[act_column] = 'median'  # Take median of activity column.
        merged = to_merge.groupby(c).agg(aggs)

        logger.info(f'Number of duplicates merged: {len(merged)} out of {len(to_merge)}.')
        results.append(merged)

    if not to_keep.empty and keep == 'first':
        logger.info(f'Check which duplicates should be kept -'
                    f' This usually happens with isomers with very different activities.')

        duplis_kept = to_keep.groupby('inchikey', group_keys=False).apply(
            lambda x: x.loc[x[act_column].idxmax()]).copy().reset_index(drop=True)
        duplis_kept.reset_index(drop=True, inplace=True)
        results.append(duplis_kept)

    elif not to_keep.empty and keep == 'last':
        duplis_kept = to_keep.groupby('inchikey', group_keys=False).apply(
            lambda x: x.loc[x[act_column].idxmin()]).copy().reset_index(drop=True)

        duplis_kept.reset_index(drop=True, inplace=True)
        results.append(duplis_kept)

    # Recombine data
    recombined_data = pd.concat([no_duplicates, *results], axis=0, ignore_index=True)
    logger.info(f'Duplicates removal reduced the number of rows from {len(data)} to {len(recombined_data)}')
    recombined_data.reset_index(drop=True, inplace=True)

    return recombined_data


class MolCleaner:
    """
    Use one of the factory methods (´from_df´, ´from_csv´ or ´from_list´) instead of using directly.

    1. Standardize unknown stereochemistry (Handled by the RDKit Mol file parser)
        i) Fix wiggly bonds on sp3 carbons - sets atoms and bonds marked as unknown stereo to no stereo
        ii) Fix wiggly bonds on double bonds – set double bond to crossed bond
    2. Clears S Group data from the mol file
    3. Kekulize the structure
    4. Remove H atoms (See the page on explicit Hs for more details)
    5. Normalization:
        Fix hypervalent nitro groups
        Fix KO to K+ O- and NaO to Na+ O- (Also add Li+ to this)
        Correct amides with N=COH
        Standardise sulphoxides to charge separated form
        Standardize diazonium N (atom :2 here: [*:1]-[N;X2:2]#[N;X1:3]>>[*:1]) to N+
        Ensure quaternary N is charged
        Ensure trivalent O ([*:1]=[O;X2;v3;+0:2]-[#6:3]) is charged
        Ensure trivalent S ([O:1]=[S;D2;+0:2]-[#6:3]) is charged
        Ensure halogen with no neighbors ([F,Cl,Br,I;X0;+0:1]) is charged
    6. The molecule is neutralized, if possible. See the page on neutralization rules for more details.
    7. Remove stereo from tartrate to simplify salt matching
    8. Normalise (straighten) triple bonds and allenes



    The curation steps in ChEMBL structure pipeline were augmented with additional steps to identify duplicated entries
    9. Find stereo centers
    10. Generate inchi keys
    11. Find duplicated SMILES. If the same SMILES is present multiple times, two outcomes are possible.
        i. The same compound (e.g. same ID and same SMILES)
        ii. Isomers with different SMILES, IDs and/or activities

        In case i), the compounds are merged by taking the median values of all numeric columns in the dataframe. For
        case ii), the compounds are further classified as 'to merge' or 'to keep' depending on the activity values.
        a) Compounds are considered for mergining (to merge) if the difference in acvitities is less than 1log unit.
        b) Compounds are considered for keeping as individual entries (to keep) if the difference in activities is
        larger than 1log unit. In this case, the user can select which compound to keep - the one with highest or
        lowest activity.
    """

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
    def from_list(cls,
                  smiles_list: List[str],
                  output_column: str = 'RDKIT_SMILES',
                  chunk_size: int = 64,
                  n_workers: int = 1,
                  pause: int = 0):

        df = pd.DataFrame({'ID': [f'mol_{x}' for x in range(len(smiles_list))],
                           'SMILES': smiles_list})

        return cls.from_df(df,
                           smiles_column='SMILES',
                           output_column=output_column,
                           chunk_size=chunk_size,
                           n_workers=n_workers,
                           pause=pause)

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

    @classmethod
    def from_csv(cls,
                 csv_path: str,
                 smiles_column: str,
                 output_column: str = 'RDKIT_SMILES',
                 chunk_size: int = 64,
                 n_workers: int = 1,
                 pause: int = 0):

        df = pd.read_csv(csv_path)

        return cls.from_df(df, smiles_column=smiles_column, output_column=output_column, chunk_size=chunk_size,
                           n_workers=n_workers, pause=pause)
