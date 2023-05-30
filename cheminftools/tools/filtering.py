from typing import Dict, Any, List
import logging
import pandas as pd
from rdkit import Chem
import sys
from joblib import cpu_count
from rdkit.Chem import FilterCatalog
from cheminftools.utils import MolBatcher

logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt='%Y/%m/%d')
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def parse_conditions(condition_dict: Dict[str, Dict[str, Any]]) -> str:
    condition_dict = list(condition_dict.items())
    all_conditions = []
    for name, conditions in condition_dict:
        low, high = conditions['low'], conditions['high']

        if high and not low:
            all_conditions.append(f'{name} <= {high}')
        elif low and not high:
            all_conditions.append(f'{name} >= {low}')

        elif low and high:
            all_conditions.append(f'{high} <= {name} <= {high}')
    return ' & '.join(all_conditions)


class AlertMatcher:
    """
    Creates a matcher object from a dictionary of alerts.
    The dictionary must follow the format:
    {'ALERT_NAME': {'SMARTS': SMARTS string of the alert}}

    metadata can also be added in the internal dictionary, as follows:

    {'NO_DRUGLIKE': {'SMARTS' : ['Pb', 'As', 'Hg', 'Sn'], 'priority' : 0}},
    where `priority` sets how important the alert is.

    The `catalog` variable stores a rdkit FilterCatalog object that can be used
    to flag compounds.

    """

    def __init__(self, alert_dict: Dict[str, Dict[str, Any]]):
        self.alert_dict = alert_dict
        self.alert_names = list(alert_dict.keys())
        self.catalog = None

    @property
    def catalog(self):
        return self._catalog

    @catalog.setter
    def catalog(self, v):
        self._catalog = v

    def create_matcher(self) -> FilterCatalog.FilterCatalog:
        """
        Creates a FilterCatalog from a dictionary of alerts.

        Returns
        -------
        catalog
            A FilterCatalog.
        """

        catalog = FilterCatalog.FilterCatalog()

        for i, (patt_name, smarts_key) in enumerate(self.alert_dict.items()):

            info = self.alert_dict[patt_name]
            smarts = info['SMARTS']
            pname_final = []

            # info.pop('SMARTS', None) # Remove SMARTS and keep other metadata.

            if info:
                for key, value in info.items():
                    if key == list(info.keys())[-1]:
                        pname_final.append(f'key={key}__{str(value)}')
                    else:
                        pname_final.append(f'key={key}__{str(value)}__')

            pname_final = 'key=alert_name' + '__' + patt_name + '__' + ''.join(pname_final)

            fil = FilterCatalog.SmartsMatcher(pname_final, smarts, minCount=1)
            catalog.AddEntry(FilterCatalog.FilterCatalogEntry(pname_final, fil))
            catalog.GetEntry(i).SetProp('Scope', smarts)

        self.catalog = catalog
        return catalog


def get_alerts(smi: str, catalog: FilterCatalog.FilterCatalog) -> pd.DataFrame:
    results = []
    info_names = [x.split('key=')[1] for x in catalog.GetEntryWithIdx(0).GetDescription().split('__') if 'key=' in x]
    mol = Chem.MolFromSmiles(smi)

    entries = catalog.GetMatches(mol)

    if len(entries) == 0:
        return pd.DataFrame(results, columns=info_names)

    for entry in entries:
        description = entry.GetDescription().split('__')
        values = pd.DataFrame([x for x in description if 'key=' not in x]).T
        values.columns = info_names
        results.append(values)

    results = pd.concat(results).drop_duplicates()
    results['SMILES'] = smi
    results.reset_index(drop=True, inplace=True)
    return results


class MolFilter:
    @classmethod
    def filter(cls, df: pd.DataFrame, conditions: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
        conditions = parse_conditions(conditions)
        logger.info(f'Filter conditions: {conditions}')
        return df.query(conditions)

    @classmethod
    def get_alerts(cls, smiles_list: List[str], catalog: FilterCatalog.FilterCatalog, chunk_size: int = 64,
                   n_workers: int = None, pause: int = 1):

        if n_workers is None:
            n_workers = cpu_count()

        logger.info('Processing SMILES.')

        batcher = MolBatcher(get_alerts,
                             smiles_list=smiles_list,
                             catalog=catalog,
                             chunk_size=chunk_size,
                             n_workers=n_workers,
                             pause=pause)

        all_alerts = pd.concat(batcher)

        if all_alerts.empty:
            pass

        n_flagged = len(all_alerts['SMILES'].unique())
        most_common_alert = all_alerts['SMARTS'].mode().tolist()
        most_common_desc = all_alerts[all_alerts.columns[0]].mode().tolist()
        logger.info(f'Number of SMILES flagged: {n_flagged}')
        logger.info(f'Most common alert: {most_common_alert} - {most_common_desc}')
        return all_alerts

    @classmethod
    def from_df(cls,
                df: pd.DataFrame,
                catalog: FilterCatalog.FilterCatalog,
                smiles_column: str,
                chunk_size: int = 64,
                n_workers: int = 1,
                pause: int = 0,
                filters: Dict[str, Dict[str, Any]] = {}) -> pd.DataFrame:

        logger.info('Reading molecules from a DataFrame')
        data = df.copy()
        logger.info(f'Number of compounds: {len(data)}')

        if filters:
            logger.info(f'Filtering data using {filters}')
            data = cls.filter(data, filters)
            data.reset_index(drop=True, inplace=True)

        logger.info('Flagging compounds according.')
        flagged = cls.get_alerts(smiles_list=data[smiles_column].values,
                                 catalog=catalog,
                                 chunk_size=chunk_size,
                                 n_workers=n_workers,
                                 pause=pause)

        flagged_aggregated = flagged.groupby('SMILES').agg(lambda x: ','.join(x)).reset_index()
        flagged_aggregated.rename(columns={'SMILES': smiles_column}, inplace=True)

        return pd.merge(data, flagged_aggregated, on=smiles_column)

    @classmethod
    def from_csv(cls, csv_path: str,
                 catalog: FilterCatalog.FilterCatalog,
                 smiles_column: str,
                 chunk_size: int = 64,
                 n_workers: int = 1,
                 pause: int = 0,
                 filters: Dict[str, Dict[str, Any]] = {}):

        df = pd.read_csv(csv_path)
        return cls.from_df(df,
                           catalog=catalog,
                           smiles_column=smiles_column,
                           chunk_size=chunk_size,
                           n_workers=n_workers,
                           pause=pause,
                           filters=filters)

    @classmethod
    def from_list(cls, smiles_list: List[str],
                  catalog: FilterCatalog.FilterCatalog,
                  smiles_column: str,
                  chunk_size: int = 64,
                  n_workers: int = 1,
                  pause: int = 0,
                  filters: Dict[str, Dict[str, Any]] = {}):

        df = pd.DataFrame({'SMILES': smiles_list, 'ID': list(range(len(smiles_list)))})
        return cls.from_df(df,
                           catalog=catalog,
                           smiles_column=smiles_column,
                           chunk_size=chunk_size,
                           n_workers=n_workers,
                           pause=pause,
                           filters=filters)
