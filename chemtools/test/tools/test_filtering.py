import unittest
from chemtools.tools.sanitizer import MolCleaner, normalize_mol, get_stereo_info, get_delta_act, mol_to_inchi
import pandas as pd
from rdkit import Chem


class Tets(unittest.TestCase):
    def test_normalize_mol(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        self.assertIsInstance(smi, str)

    def test_get_stereo_inf(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        info = get_stereo_info(smi)
        self.assertEqual(info, '3?')

        smi = r'F/C=C\F'
        info = get_stereo_info(smi)
        self.assertEqual(info, '1_CIS_2')

    def test_MolCleaner(self):
        df = pd.DataFrame(
            {'ID': ['mol1', 'mol2'], 'SMILES': ['O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl', r'F/C=C\F']})
        res = MolCleaner.from_df(df, smiles_column='SMILES', n_workers=1, chunk_size=1, pause=0)
        self.assertEqual(res.empty, False)
        self.assertIn('RDKIT_SMILES', res.columns)
        self.assertIn('Stereo', res.columns)
        self.assertIn('inchikey', res.columns)

    def test_get_delta_act(self):
        keep_acts = [1.5, 3.5]
        merge_acts = [0.5, 0.2]
        ok_acts = [0.5, 0.5]

        deltas = [get_delta_act(x) for x in [ok_acts, merge_acts, keep_acts]]
        self.assertEqual(deltas, ['Not duplicate', 'to_merge', 'to_keep'])

    def test_normalize_mol(self):
        problematic_smi = 'C1=CC=CC=C1'
        normalized_mol = normalize_mol(problematic_smi)
        self.assertIsInstance(normalized_mol, Chem.Mol)
        self.assertEqual(Chem.MolToSmiles(normalized_mol), 'c1ccccc1')

    def test_mol_to_inchi(self):
        test_inchikeys = ['UHOVQNZJYSORNB-UHFFFAOYSA-N', 'KGFYHTZWPPHNLQ-AWEZNQCLSA-N']
        test_mols = ['c1ccccc1', 'C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl']
        result_inchikey = [mol_to_inchi(x) for x in test_mols]
        self.assertEqual(result_inchikey, test_inchikeys)


if __name__ == '__main__':
    unittest.main()
