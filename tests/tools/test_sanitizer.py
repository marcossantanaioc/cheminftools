from cheminftools.tools import sanitizer as chem_sanitizer
import pandas as pd
from rdkit import Chem
import pytest
import warnings

class TestsSanitizer:
    def test_check_stereo(self):
        smi = "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"
        smi2 = "C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl"
        smi3_cis = (r"F/C=C\F",)
        smi3_trans = r"F\C=C\F"
        smi3_unspec = "FC=CF"

        assert not chem_sanitizer.check_stereo(smi)
        assert chem_sanitizer.check_stereo(smi2)
        assert chem_sanitizer.check_stereo(smi3_cis)
        assert chem_sanitizer.check_stereo(smi3_trans)
        assert not chem_sanitizer.check_stereo(smi3_unspec)

    def test_get_stereo_info(self):
        smi = "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"
        info = chem_sanitizer.get_stereo_info(smi)
        assert info == "3?"

        smi = r"F/C=C\F"
        info = chem_sanitizer.get_stereo_info(smi)
        assert info == "1_CIS_2"

    def test_MolCleaner(self):
        df = pd.DataFrame(
            {
                "ID": ["mol1", "mol2"],
                "SMILES": [
                    "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl",
                    r"F/C=C\F",
                ],
            }
        )
        res = chem_sanitizer.MolCleaner.from_df(
            df, smiles_column="SMILES", n_workers=1, chunk_size=1, pause=0
        )
        assert not res.empty
        assert "RDKIT_SMILES" in res.columns
        assert "Stereo" in res.columns
        assert "inchikey" in res.columns

    def test_normalize_mol(self):
        problematic_smi = "C1=CC=CC=C1"
        normalized_mol = chem_sanitizer.normalize_mol(problematic_smi)
        assert isinstance(normalized_mol, Chem.Mol)
        assert Chem.MolToSmiles(normalized_mol) == "c1ccccc1"

    def test_mol_to_inchi(self):
        test_inchikeys = ["UHOVQNZJYSORNB-UHFFFAOYSA-N", "KGFYHTZWPPHNLQ-AWEZNQCLSA-N"]
        test_mols = [
            "c1ccccc1",
            "C1COCC(=O)N1C2=CC=C(C=C2)N3C[C@@H](OC3=O)CNC(=O)C4=CC=C(S4)Cl",
        ]
        result_inchikey = [chem_sanitizer.mol_to_inchi(x) for x in test_mols]
        assert result_inchikey == test_inchikeys

    def test_process_duplicates(self):
        data = pd.DataFrame(
            {
                "SMILES": [
                    "O=C(NC[C@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",
                    "O=C(NC[C@@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",
                    "O=C(NCC1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",
                    "c1ccccc1",
                    r"F/C=C\F",
                    r"F\C=C\F",
                    r"FC=CF",
                ],
                "pIC50": [9.5, 11.0, 9.0, 1.0, 4.5, 3.78, 3.8],
                "ID": [
                    "riva_isomer1",
                    "riva_isomer2",
                    "riva_racemic",
                    "benzene",
                    "cisdifluoroethene",
                    "transdifluoroethene",
                    "difluoroethane",
                ],
            }
        )

        data_no_duplis = chem_sanitizer.process_duplicates(
            data,
            smiles_column="SMILES",
            act_column="pIC50",
            cols_to_check=[],
            keep="first",
        )
        unique_smiles = set(data_no_duplis["SMILES"])
        expected_smiles = set([
            "c1ccccc1",
            r"F/C=C\F",
            "O=C(NC[C@@H]1CN(c2ccc(N3CCOCC3=O)cc2)C(=O)O1)c1ccc(Cl)s1",]
        )
        warnings.warn(DeprecationWarning("Old"))
        assert unique_smiles.issubset(expected_smiles)
        assert data_no_duplis["pIC50"].max() == 11.0
        assert data_no_duplis["pIC50"].min() == 1.0
        assert len(data_no_duplis) == 3


if __name__ == "__main__":
    pytest.main([__file__])
