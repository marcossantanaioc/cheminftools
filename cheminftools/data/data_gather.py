from typing import List, Union
import pandas as pd
import psycopg2
import logging
import sys
from config import config

logging.basicConfig(level=logging.INFO, stream=sys.stdout, datefmt='%Y/%m/%d')
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ChemblFetcher:
    """
    Collects examples from a ChEMBL database
    The user must provide a valid instance from ChEMBL
    See https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/latest/
    for the latest relases.
    Attributes
    ----------
    version
        ChEMBL version to use.
        The latest version is 32.
    """

    def __init__(self, database_config_filename: str = 'database.ini', database_name: str = 'chembl',
                 version: str = '32'):
        self.database_name = database_name
        self.version = version
        self.params = config(filename=database_config_filename)

    def get_connection(self):
        """
        Opens a connection with a ChEMBL database.
        Returns
        -------
        connection
            A psycopg2 connection object.
        """

        logger.info(f'Connecting to the ChEMBL{self.version} database...')
        connection = psycopg2.connect(database=f'chembl{self.version}', **self.params)
        return connection

    def get_sql_query(self,
                      target_uniprot: Union[List[str], str],
                      activity_relation: Union[List[str], str] = ['<', '>', '=', '>=', '<='],
                      activity_type: Union[List[str], str] = ['IC50', 'Kd', 'Ki', 'EC50']):

        """
        Create a SQL query based on
        Uniprot accessions, activity type and relations.

        Parameters
        ----------
        target_uniprot
            A collection of uniprot accession codes.
        activity_relation
            Which activity relations to retrieve.
        activity_type
            Which activity types to retrieve.
        Returns
        -------
            A SQL string.

        """

        if isinstance(target_uniprot, List):
            target_uniprot = ', '.join(["'{}'".format(x) for x in target_uniprot])
        if isinstance(activity_type, List):
            activity_type = ', '.join(["'{}'".format(x) for x in activity_type])
        if isinstance(activity_relation, List):
            activity_relation = ', '.join(["'{}'".format(x) for x in activity_relation])

        sql_string = f"""
        SELECT compound_structures.canonical_smiles,
        activities.standard_value AS activity_value,
        activities.standard_relation AS activity_relation,
        activities.standard_units AS activity_units,
        activities.standard_type AS activity_type,
        target_dictionary.pref_name AS target_name,
        target_dictionary.chembl_id AS target_chembl_id,
        accession AS uniprot_id
        FROM component_sequences
        INNER JOIN target_components ON
        component_sequences.component_id = target_components.component_id
        INNER JOIN target_dictionary ON
        target_components.tid = target_dictionary.tid
        INNER JOIN assays ON 
        target_dictionary.tid = assays.tid
        INNER JOIN activities ON
        assays.assay_id = activities.assay_id
        INNER JOIN compound_structures ON
        activities.molregno = compound_structures.molregno
        WHERE accession IN ({target_uniprot}) AND
        standard_value IS NOT NULL AND
        standard_relation IN ({activity_relation}) AND
        standard_type IN ({activity_type}) AND
        activities.potential_duplicate = 0 AND
        assays.confidence_score >= 8 AND
        target_dictionary.target_type = 'SINGLE PROTEIN'
        """
        return sql_string

    def query_target_uniprot(self, target_uniprot: Union[List[str], str]):
        """
        Query ChEMBL for target_uniprot.
        Parameters
        ----------
        target_uniprot
            A collection of uniprot accession codes.

        Returns
        -------
        df
            Output pandas DataFrame.
        """

        connection = self.get_connection()
        sql_query_string = self.get_sql_query(target_uniprot=target_uniprot)
        with connection as conn:
            df = pd.read_sql(sql_query_string, con=conn)
        connection.close()
        return df
