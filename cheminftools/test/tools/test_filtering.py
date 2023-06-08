from rdkit.Chem.FilterCatalog import FilterCatalog
import pandas as pd
from cheminftools.tools.filtering import AlertMatcher, get_one_alert, MolFilter


class TestFiltering:
    """
    Pytests
    """

    def test_alertmatcher(self):
        alerts_dict = {'check_aliphatic_carbon': {'SMARTS': 'C'}}
        matcher = AlertMatcher(alerts_dict)
        catalog = matcher.create_matcher()
        assert len(matcher.alert_names) == 1  # Check if alerts_dict was added to matcher.
        assert isinstance(catalog, FilterCatalog)  # Check if FilterCatalog was created.

    def test_get_alerts(self):
        alerts_dict = {'has_primary_alcohol': {'SMARTS': '[CX4;H2][OH1]'}}
        smi = 'OCc1ccccc1'
        catalog = AlertMatcher(alerts_dict).create_matcher()
        alerts = get_one_alert(smi, catalog)
        assert isinstance(alerts, pd.DataFrame)
        assert alerts['alert_name'].item() == 'has_primary_alcohol'

    def test_molfilter_from_list(self):
        alerts_dict = {'has_primary_alcohol': {'SMARTS': '[CX4;H2][OH1]'}}
        smis = ['OCc1ccccc1', 'CCO']
        catalog = AlertMatcher(alerts_dict).create_matcher()
        alerts = MolFilter.from_list(smiles_list=smis, catalog=catalog)
        assert alerts['alert_name'].unique().item() == 'has_primary_alcohol'

    def test_molfilter_from_df(self):
        alerts_dict = {'has_primary_alcohol': {'SMARTS': '[CX4;H2][OH1]'}}
        smis = ['OCc1ccccc1', 'CCO']
        df = pd.DataFrame({'SMILES': smis})
        catalog = AlertMatcher(alerts_dict).create_matcher()
        alerts = MolFilter.from_df(df=df, catalog=catalog, smiles_column='SMILES')
        assert alerts['alert_name'].unique().item() == 'has_primary_alcohol'
