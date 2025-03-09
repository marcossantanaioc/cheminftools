from rdkit.Chem.FilterCatalog import FilterCatalog
import pandas as pd
from cheminftools.tools.filtering import AlertMatcher, get_one_alert, MolFilter
import pytest


class TestFiltering:
    """
    Pytests
    """

    @pytest.fixture
    def smis(self):
        smis = ["OCc1ccccc1", "CCO"]
        return smis

    @pytest.fixture
    def smis_df(self, smis):
        return pd.DataFrame({"SMILES": smis})

    @pytest.fixture
    def alcohol_catalog(self):
        alerts_dict = {"has_primary_alcohol": {"SMARTS": "[CX4;H2][OH1]"}}
        catalog = AlertMatcher(alerts_dict).create_matcher()
        return catalog

    def test_alertmatcher(self):
        alerts_dict = {"check_aliphatic_carbon": {"SMARTS": "C"}}
        matcher = AlertMatcher(alerts_dict)
        catalog = matcher.create_matcher()
        assert (
            len(matcher.alert_names) == 1
        )  # Check if alerts_dict was added to matcher.
        assert isinstance(catalog, FilterCatalog)  # Check if FilterCatalog was created.

    def test_get_alerts(self, alcohol_catalog):
        smi = "OCc1ccccc1"
        alerts = get_one_alert(smi, alcohol_catalog)
        assert isinstance(alerts, pd.DataFrame)
        assert alerts["alert_name"].item() == "has_primary_alcohol"

    def test_molfilter_from_list(self, smis, alcohol_catalog):
        alerts = MolFilter.from_list(smiles_list=smis, catalog=alcohol_catalog)
        assert alerts["alert_name"].unique().item() == "has_primary_alcohol"

    def test_molfilter_from_df(self, smis_df, alcohol_catalog):
        alerts = MolFilter.from_df(
            df=smis_df, catalog=alcohol_catalog, smiles_column="SMILES"
        )
        assert alerts["alert_name"].unique().item() == "has_primary_alcohol"


if __name__ == "__main__":
    pytest.main([__file__])
