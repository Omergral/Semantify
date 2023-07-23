import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from nltk.corpus import wordnet
from itertools import permutations
from typing import Tuple, Dict, Union, List, Any, Set
from semantify.utils.general import flatten_dict_of_dicts


class ChoosingDescriptorsUtils:
    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def choose_between_2_descriptors(
        self, df: pd.DataFrame, first_descriptor: str, second_descriptor: str
    ) -> Tuple[str, float]:
        first_descriptor_avg_iou = df[
            (df["descriptor_1"] == first_descriptor) | (df["descriptor_2"] == first_descriptor)
        ]["iou"].mean()
        second_descriptor_avg_iou = df[
            (df["descriptor_1"] == second_descriptor) | (df["descriptor_2"] == second_descriptor)
        ]["iou"].mean()
        if self.verbose:
            self.logger.info(f"{first_descriptor} iou: {first_descriptor_avg_iou}")
        if self.verbose:
            self.logger.info(f"{second_descriptor} iou: {second_descriptor_avg_iou}")
        if first_descriptor_avg_iou < second_descriptor_avg_iou:
            if self.verbose:
                self.logger.info(f"chose {first_descriptor} with iou {first_descriptor_avg_iou}")
            return first_descriptor, first_descriptor_avg_iou
        else:
            if self.verbose:
                self.logger.info(f"chose {second_descriptor} with iou {second_descriptor_avg_iou}")
            return second_descriptor, second_descriptor_avg_iou

    def _preprocess_2_words_descriptor(self, descriptor: str) -> str:
        splitted_to_words = descriptor.split(" ")
        if len(splitted_to_words) > 1:
            return True
        else:
            return False

    def _get_synonyms(self, descriptor: str) -> Set[str]:
        multiple_words_descriptor = self._preprocess_2_words_descriptor(descriptor)
        preprocessed_descriptor = descriptor.split(" ")[0] if multiple_words_descriptor else descriptor
        synonyms = []
        for syn in wordnet.synsets(preprocessed_descriptor):
            for l in syn.lemmas():
                synonyms.append(l.name())
        if multiple_words_descriptor:
            return set([f"{synonym} {descriptor.split(' ')[1]}" for synonym in synonyms])
        else:
            return set(synonyms)

    def _get_antonyms(self, descriptor: str) -> Set[str]:
        multiple_words_descriptor = self._preprocess_2_words_descriptor(descriptor)
        preprocessed_descriptor = descriptor.split(" ")[0] if multiple_words_descriptor else descriptor
        antonyms = []
        for syn in wordnet.synsets(preprocessed_descriptor):
            for l in syn.lemmas():
                if l.antonyms():
                    antonyms.append(l.antonyms()[0].name())
        if multiple_words_descriptor:
            return set([f"{antonym} {descriptor.split(' ')[1]}" for antonym in antonyms])
        else:
            return set(antonyms)

    def get_dfs(self, jsons_dir: Union[Path, str]):
        json_files = list(Path(jsons_dir).rglob("*_labels.json"))
        df = pd.DataFrame()
        for json_file in tqdm(json_files, desc="Loading json files", total=len(json_files)):
            with open(json_file, "r") as f:
                json_data = json.load(f)
                df = pd.concat([df, pd.DataFrame(json_data)], axis=0)
        if "broad shoulders" in df.columns:
            df = df.drop("broad shoulders", axis=1)
        if "built" in df.columns:
            df = df.drop("built", axis=1)
        df = df.apply(lambda x: [y[0] for y in x])

        # get variances
        variances = df.var(axis=0)
        variances.sort_values(ascending=False, inplace=True)
        variances = pd.DataFrame(zip(variances.index, variances.values), columns=["descriptor", "variance"])

        # get corrlation matrix between descriptors
        corr_df = pd.DataFrame(columns=["descriptor_1", "descriptor_2", "correlation"])
        permut_list = []
        for perm in permutations(df.columns, 2):
            permut_list.append(perm)
        for perm in tqdm(permut_list, desc="Calculating correlations", total=len(permut_list)):
            corr_df = pd.concat(
                [
                    corr_df,
                    pd.DataFrame(
                        {
                            "descriptor_1": [perm[0]],
                            "descriptor_2": [perm[1]],
                            "correlation": [np.corrcoef(df[perm[0]], df[perm[1]])[0, 1]],
                        }
                    ),
                ],
                axis=0,
            )
        return corr_df, variances

    def check_if_descriptor_is_correlated_with_chosen_descriptors(
        self, descriptor: str, chosen_descriptors: Dict[str, Dict[str, Any]], correlations_df: pd.DataFrame
    ) -> bool:
        for chosen_descriptor in chosen_descriptors:
            descriprtor_correlations = correlations_df[
                (
                    (correlations_df["descriptor_1"] == descriptor)
                    & (correlations_df["descriptor_2"] == chosen_descriptor)
                )
            ]
            if abs(descriprtor_correlations["correlation"][0]) > self.corr_threshold:
                return True, chosen_descriptor, descriprtor_correlations["correlation"][0]
        return False, None, None

    @staticmethod
    def get_number_of_unique_descriptors(df: Union[pd.DataFrame, str]) -> int:
        if isinstance(df, str):
            df = pd.read_csv(df)
        unique_descriptors = set(df["descriptor_1"].unique().tolist() + df["descriptor_2"].unique().tolist())
        return len(unique_descriptors)

    @staticmethod
    def get_descriptor_iou(descriptor: str, ious_df: pd.DataFrame) -> float:
        return ious_df[(ious_df["descriptor_1"] == descriptor) | (ious_df["descriptor_2"] == descriptor)]["iou"].mean()

    @staticmethod
    def get_clusters(descriptors_clusters_json: Path):
        with open(descriptors_clusters_json, "r") as f:
            clusters = json.load(f)
        return clusters

    def reduce_descriptor(self, dict_of_desctiptors: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Description
        -----------
        reduces the number of descriptors by removing the descriptor with the lowest variance

        Args
        ----
        dict_of_desctiptors (Dict[str, Dict[str, float]]) = a dictionary of descriptors and their variance

        Returns
        -------
        dict_of_desctiptors (Dict[str, Dict[str, float]]) = reduced dictionary of descriptors and their variance
        """
        # flatten the dict of dicts and sort it by variance
        flattened_dict = flatten_dict_of_dicts(dict_of_desctiptors)
        sorted_dict = sorted(flattened_dict, key=flattened_dict.get, reverse=False)

        # get the minimal variance descriptor, but verify that it is not in the descriptors to keep
        idx = 0
        keep_searching = True
        while keep_searching or idx < len(sorted_dict):
            minimal_var_descriptor = sorted_dict[idx]
            if minimal_var_descriptor not in self.descriptors_to_keep:
                keep_searching = False
            else:
                idx += 1
        cluster_of_descriptor = self.find_cluster_of_descriptor(minimal_var_descriptor, dict_of_desctiptors)

        del dict_of_desctiptors[cluster_of_descriptor][minimal_var_descriptor]
        if self.verbose:
            print(f"removing {minimal_var_descriptor} from cluster {cluster_of_descriptor}")

        clusters_to_delete = []
        for cluster in dict_of_desctiptors.keys():
            if dict_of_desctiptors[cluster] == {}:
                clusters_to_delete.append(cluster)
        for cluster in clusters_to_delete:
            del dict_of_desctiptors[cluster]
        return dict_of_desctiptors

    @staticmethod
    def find_cluster_of_descriptor(
        descriptor: str, dict_of_desctiptors: Dict[str, Union[Dict[str, float], List[str]]]
    ) -> int:
        for cluster, descriptors_dict in dict_of_desctiptors.items():
            if isinstance(descriptors_dict, dict):
                if descriptor in descriptors_dict.keys():
                    return cluster
            else:
                if descriptor in descriptors_dict:
                    return cluster

    @staticmethod
    def get_num_of_chosen_descriptors(dict_of_desctiptors: Dict[str, Dict[str, float]]) -> int:
        num_of_descriptors = 0
        for descriptors in dict_of_desctiptors.values():
            num_of_descriptors += len(descriptors)
        return num_of_descriptors
