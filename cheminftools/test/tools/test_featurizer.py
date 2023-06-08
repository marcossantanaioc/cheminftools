from rdkit.Chem import rdFingerprintGenerator

from cheminftools.tools.featurizer import MolFeaturizer


class TestMolFeaturizer:
    """
    Pytests
    """

    def test_transform_one_morgan(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='morgan', params={'radius': 2, 'fpSize': 2048})
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == 'morgan'
        assert isinstance(featurizer.generator, rdFingerprintGenerator.FingerprintGenerator64)
        assert X.max() == 1
        assert X.min() == 0

    def test_transform_one_morgan_count(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='morgan', params={'radius': 2, 'fpSize': 2048})
        X = featurizer.transform_one(smi, use_counts=True)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == 'morgan'
        assert isinstance(featurizer.generator, rdFingerprintGenerator.FingerprintGenerator64)
        assert X.max() > 1

    def test_transform_one_maccs(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='maccs')
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 167)
        assert featurizer.descriptor_type == 'maccs'

    def test_transform_one_atom_pairs(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='atom_pairs', params={'fpSize': 2048})
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == 'atom_pairs'

    def test_transform_one_erg(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='erg')
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 315)
        assert featurizer.descriptor_type == 'erg'

    def test_transform_one_rdkit(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='rdkit')
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == 'rdkit'

    def test_transform_one_rdkit2d(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='rdkit2d')
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 208)
        assert featurizer.descriptor_type == 'rdkit2d'

    def test_transform_one_torsion(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer(descriptor_type='torsion', params={'fpSize': 2048})
        X = featurizer.transform_one(smi)
        assert X.shape == (1, 2048)
        assert featurizer.descriptor_type == 'torsion'

    def test_transfom(self):
        smi = 'O=C1OC(CN1c1ccc(cc1)N1CCOCC1=O)CNC(=O)c1ccc(s1)Cl'
        featurizer = MolFeaturizer('morgan')
        X = featurizer.transform([smi])
        assert X.max() == 1
        assert X.min() == 0
