# from cheminftools.utils import MolBatcher
# from cheminftools.tools.featurizer import MolFeaturizer
# import pandas as pd
# from typing import Dict, List
#
# similarity_threshold = 0.7
#
# class SimilaritySearch:
#     def __init__(self, smiles_column: str, fp_type: str='morgan', similarity_threshold: float=0.35, fp_params: Dict={}):
#         self.smiles_column = smiles_column
#         self.featurizer = MolFeaturizer(descriptor_type=fp_type, params=fp_params)
#         self.similarity_threshold = similarity_threshold
#
#     def process(self, df: pd.DataFrame):
#         df_dict = {x : self.featurizer.transform_one(x) for x in df[self.smiles_column]}
#         stack = {'target_SMILES':[], 'similarity':[], 'query_SMILES':[]}
#         for key2 in df_dict.keys():
#             if key != key2:
#                 similarity = 1 - distance.jaccard(df_dict[key].squeeze(), df_dict[key2].squeeze())
#                 if similarity >= similarity_threshold:
#                     stack['target_SMILES'].append(key2)
#                     stack['similarity'].append(similarity)
#                     stack['query_SMILES'].append(key)
#         return pd.DataFrame(stack)
#
# # def process_smiles_list(idxs, smiles_list):
# #
# #     res = [compare_keys(smiles_list[i]) for i in idxs]
# #     return res
# #
# #     batcher = MolBatcher(process_smiles_list,
# #                          smiles_list=data['canonical_smiles'].values,
# #                          chunk_size=256,
# #                          n_workers=5,
# #                          pause=0)
Phone = type('Phone',(),{'brand':'Apple'})
print(Phone)