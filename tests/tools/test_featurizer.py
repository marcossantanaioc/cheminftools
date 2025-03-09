from rdkit.Chem import rdFingerprintGenerator
import pytest
from cheminftools.tools import featurizer as chem_featurizer


class TestMolFeaturizer:
    """
    Pytests
    """

    @pytest.fixture
    def smi(self):
        return "O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl"

    @pytest.fixture
    def morgan_featurizer(self):
        return chem_featurizer.MolFeaturizer(
            descriptor_type="morgan", params={"radius": 2, "fpSize": 2048}
        )

    def test_transform_one_morgan(self, smi, morgan_featurizer):
        X = morgan_featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert morgan_featurizer.descriptor_type == "morgan"
        assert isinstance(
            morgan_featurizer.generator, rdFingerprintGenerator.FingerprintGenerator64
        )
        assert X.max() == 1
        assert X.min() == 0

    def test_transform_one_morgan_count(self, smi, morgan_featurizer):
        X = morgan_featurizer.transform_one(smi, use_counts=True)
        assert X.shape == (1, 2048)
        assert morgan_featurizer.descriptor_type == "morgan"
        assert isinstance(
            morgan_featurizer.generator, rdFingerprintGenerator.FingerprintGenerator64
        )
        assert X.max() > 1

    def test_transform_one_maccs(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(descriptor_type="maccs")
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 167)
        assert featurizer.descriptor_type == "maccs"

    def test_transform_one_atom_pairs(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(
            descriptor_type="atom_pairs", params={"fpSize": 2048}
        )
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == "atom_pairs"

    def test_transform_one_erg(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(descriptor_type="erg")
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 315)
        assert featurizer.descriptor_type == "erg"

    def test_transform_one_rdkit(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(descriptor_type="rdkit")
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == "rdkit"

    def test_transform_one_rdkit2d(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(descriptor_type="rdkit2d")
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 217)
        assert featurizer.descriptor_type == "rdkit2d"

    def test_transform_one_torsion(self, smi):
        featurizer = chem_featurizer.MolFeaturizer(
            descriptor_type="torsion", params={"fpSize": 2048}
        )
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == "torsion"

    def test_transfom(self, smi, morgan_featurizer):
        X = morgan_featurizer.transform([smi])
        assert X.max() == 1
        assert X.min() == 0


if __name__ == "__main__":
    pytest.main([__file__])
