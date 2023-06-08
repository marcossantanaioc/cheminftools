from rdkit import Chem

from cheminftools.utils import convert_smiles, get_delta_act


class TestUtils:
    """
    Pytests
    """

    def test_get_delta_act(self):
        keep_acts = [1.5, 3.5]
        merge_acts = [0.5, 0.2]
        ok_acts = [0.5, 0.5]

        deltas = [get_delta_act(x) for x in [ok_acts, merge_acts, keep_acts]]
        assert deltas == ['Not duplicate', 'to_merge', 'to_keep']

    def test_convert_smiles(self):
        smi = 'C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl'
        mol = convert_smiles(smi, sanitize=True)
        assert isinstance(mol, Chem.Mol)
