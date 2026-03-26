from collections import Counter
from torch.utils.data import Dataset
import torch
from typing import Dict, Tuple, List, Optional
from utils.infra import get_files_in_dir
import copy
import numpy as np
import re
import random

class WSILevelFeatureDataset(Dataset):

    """
    Dataset which loads WSI-level features (e.g. features derived from some foundation model)
    """

    def __init__(
        self,
        feature_dir_path_list,
        targets_dict,
        sample_keys_set = None,
        filter_regex: str = '.*',
        allow_identical_targets: bool = True,
        no_targets: bool = False,
    ) -> None:
        
        # init arguments
        self.feature_dir_path_list = feature_dir_path_list
        self.targets_dict = copy.deepcopy(targets_dict)
        self.sample_keys_set = sample_keys_set
        self.filter_regex = filter_regex
        self.filter_regex_pattern = re.compile(self.filter_regex)
        self.allow_identical_targets = allow_identical_targets
        self.no_targets = no_targets
        
        # filtering dataset for the provided sample keys (e.g. only train samples)
        if self.sample_keys_set is not None:
            self.sample_keys_set = set(self.sample_keys_set)
            self.targets_dict = {
                k: v for k, v in self.targets_dict.items() if k in self.sample_keys_set
            }

        # list of all feature files
        self.feature_files = []
        for dir_path in self.feature_dir_path_list:
            self.feature_files.extend(get_files_in_dir(dir_path=dir_path))
            
        # if no targets are provided, create a dummy target for each sample
        if self.no_targets:
            self.targets_dict = {file.name: [0] for file in self.feature_files}

        # mapping sample targets, features, and names
        self.idx_sample_feature_path_map = {}
        self.idx_sample_target_map = {}
        self.idx_sample_id_map = {}
        self.sample_id_idx_map = {}

        sample_idx = 0
        for sample_id in self.targets_dict:

            # get a list of files associated with the sample
            sample_x_files = [file for file in self.feature_files if sample_id in str(file) and file.suffix == ".pt"]

            # filtering using regext expression
            if self.filter_regex_pattern is not None:
                sample_x_files = [
                    file for file in sample_x_files if self.filter_regex_pattern.match(str(file))
                ]

            # adding each file as a separate sample
            for feature_file in sample_x_files:
                self.idx_sample_target_map[sample_idx] = self.targets_dict[sample_id]
                self.idx_sample_feature_path_map[sample_idx] = feature_file
                self.idx_sample_id_map[sample_idx] = feature_file.name
                self.sample_id_idx_map[feature_file.name] = sample_idx
                sample_idx += 1

        # ensure consistent dimensions across the targets
        if not self.no_targets:
            target_dim_set = set([len(target) for target in self.idx_sample_target_map.values()])
            assert len(target_dim_set) == 1, f"All targets much have same dims. Got {target_dim_set}"
            self.target_dim = list(target_dim_set)[0]

            if not self.allow_identical_targets:
                print("Checking for identical targets...")
                unique_targets = set([str(v) for v in self.idx_sample_target_map.values()])
                print(f"Unique targets: {unique_targets}")
                if len(unique_targets) == 1:
                    raise RuntimeError(
                        "All targets are identical. Please set allow_identical_targets=True to proceed."
                    )

    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:

        # get the name of the sample
        sample_id = self.idx_sample_id_map[index]

        # get the sample label
        y = torch.Tensor(self.idx_sample_target_map[index]).float()

        # get location of features on disk
        x_path = self.idx_sample_feature_path_map[index]

        # load data tile features ot the cpu
        x = torch.load(x_path, map_location=torch.device('cpu'), weights_only=False)

        return x, y, sample_id

    def __len__(self):
        return len(self.idx_sample_id_map)
    

    def get_sample_weights(self, class_balance: bool = True, names_balance: List[str] = []):
        """ This function calculates sample weights for balancing by class labels, substrings in sample names, or both. """
        
        # Can only balance when the target is single-dimensional
        if self.target_dim > 1:
            raise RuntimeError("Cannot compute weights for multi-target samples.")
        
        # Collect all targets in order
        targets = []
        sample_names = []
        for idx in range(self.__len__()):
            targets.append(self.idx_sample_target_map[idx][0])
            sample_names.append(self.idx_sample_id_map[idx])  # Assuming a mapping of idx to sample name
        
        sample_weights = np.ones(len(targets))  # Default all weights to 1
        
        if class_balance:
            # Count occurrences of each class
            class_counts = Counter(targets)
            class_weights = {cls: 1.0 / count for cls, count in class_counts.items()}
            
            # Assign class-based weights
            sample_weights *= np.array([class_weights[t] for t in targets])
        
        if names_balance:
            # Count occurrences of each name substring
            name_counts = {name: 0 for name in names_balance}
            for sample in sample_names:
                for name in names_balance:
                    if name in sample:
                        name_counts[name] += 1
            
            # Compute weights for substrings
            name_weights = {name: 1.0 / count if count > 0 else 1.0 for name, count in name_counts.items()}
            print(name_weights)
            
            # Apply substring-based weights
            for i, sample in enumerate(sample_names):
                for name in names_balance:
                    if name in sample:
                        sample_weights[i] *= name_weights[name]
                        break  # Ensuring a single weight is applied per sample
        
        # Convert to tensor and return
        sample_weights = torch.tensor(sample_weights, dtype=torch.double)
        return sample_weights



    def _show_data_stats(self):
        targets_array = np.array(list(self.idx_sample_target_map.values()))
        print(f"Number of samples: {self.__len__()}")
        print(f"Target means: {np.mean(targets_array, axis=0)}")


class TileLevelFeaturesDataset(WSILevelFeatureDataset):

    """
    Dataset which loads tile-level features (e.g. features derived from some foundation model)
    """

    def __init__(
        self,
        feature_dir_path_list,
        targets_dict,
        sample_keys_set = None,
        filter_regex: str = '.*',
        n_tiles: Optional[int] = None,
        no_targets: bool = False,
    ) -> None:
        
        super().__init__(
            feature_dir_path_list=feature_dir_path_list,
            targets_dict=targets_dict,
            sample_keys_set=sample_keys_set,
            filter_regex=filter_regex,
            no_targets=no_targets,
        )
        self.n_tiles = n_tiles
    
    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:

        x, y, sample_id = super().__getitem__(index=index)

        # randomly sampling the tiles when there is a required tile count
        x = sample_tiles_from_tensor(all_tiles_tensor=x, target_tiles_count=self.n_tiles)

        return x, y, sample_id

class ViTPatchDataset(WSILevelFeatureDataset):
    """
    Dataset that loads ViT patch features / targets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:
        pass


class TileLevelFeaturesMetaDataFilterDataset(Dataset):

    def __init__(
        self,
        feature_dir_path_list,
        meta_data_dir_path_list,
        targets_dict,
        filter_field: str,
        filter_fraction: Optional[float] = None,
        absolute_threshold: Optional[float] = None,
        keep_top: bool = False,
        sample_keys_set = None,
        filter_regex: str =  '.*',
        apply_metadata_filter_regex: str = '.*',
        n_tiles: Optional[int] = None,
    ) -> None:
        
        assert (filter_fraction is not None or absolute_threshold is not None), \
            "filter_fraction and absolute_threshold cannot both be None"
        
        assert not (filter_fraction is not None and absolute_threshold is not None), \
            "filter_fraction and absolute_threshold are mutually exclusive"
        
        self.tile_feature_dataset = TileLevelFeaturesDataset(
            feature_dir_path_list=feature_dir_path_list,
            targets_dict=targets_dict,
            sample_keys_set=sample_keys_set,
            filter_regex=filter_regex,
            n_tiles=None,
        )
        self.metadata_dataset = WSILevelFeatureDataset(
            feature_dir_path_list=meta_data_dir_path_list,
            targets_dict=targets_dict,
            sample_keys_set=sample_keys_set,
            filter_regex=filter_regex,
        )
        self.n_tiles = n_tiles
        self.filter_field = filter_field
        self.filter_fraction = filter_fraction
        self.absolute_threshold = absolute_threshold
        self.keep_top = keep_top
        self.apply_metadata_filter_regex = apply_metadata_filter_regex
        self.apply_metadata_filter_regex_pattern = re.compile(self.apply_metadata_filter_regex)

        self.common_samples = set.intersection(
            set(self.tile_feature_dataset.sample_id_idx_map.keys()),
            set(self.metadata_dataset.sample_id_idx_map.keys())
        )

        self.idx_sample_id_map = {idx: sample_id for idx, sample_id in enumerate(self.common_samples)}

    def __len__(self): 
        return len(self.idx_sample_id_map)

    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:

        # get the sample id associated with the provided index
        sample_id = self.idx_sample_id_map[index]

        feature_path = self.tile_feature_dataset.idx_sample_feature_path_map[
            self.tile_feature_dataset.sample_id_idx_map[sample_id]
        ]

        # get the file features
        x, y, _ = self.tile_feature_dataset.__getitem__(
            index=self.tile_feature_dataset.sample_id_idx_map[sample_id]
        )

        # check if there is a need to apply a metadata tile filtering
        if self.apply_metadata_filter_regex_pattern.match(str(feature_path)):

            # get the relevant metadata
            meta, _, _ = self.metadata_dataset.__getitem__(
                index=self.metadata_dataset.sample_id_idx_map[sample_id]
            )
            meta = meta[self.filter_field]

            # filter using metadata
            if self.absolute_threshold is not None:
                x = self._filter_using_abs_threshold(x, meta)
            else:
                x = self._filter_using_fraction(x, meta)

        # randomly sampling the tiles when there is a required tile count
        x = sample_tiles_from_tensor(all_tiles_tensor=x, target_tiles_count=self.n_tiles)

        return x, y, sample_id
    
    def _show_data_stats(self):
        targets_array = np.array(list(self.tile_feature_dataset.idx_sample_target_map.values()))
        print(f"Number of samples: {self.__len__()}")
        print(f"Target means: {np.mean(targets_array, axis=0)}")

    def _filter_using_fraction(self, x, metadata):
        # get the number of tiles
        tile_count = x.shape[0]

        # order the tiles
        sorted_tile_idxs = torch.argsort(metadata)
        
        # counting number of tiles to keep
        n_to_keep = tile_count - int(tile_count * self.filter_fraction)

        # extract the tiles to keep
        if self.keep_top:
            tile_idxs = sorted_tile_idxs[tile_count - n_to_keep:]
        else:
            tile_idxs = sorted_tile_idxs[:n_to_keep]

        # filter tiles
        x = x[tile_idxs]

        return x 
    
    def _filter_using_abs_threshold(self, x, metadata):
        # get the mask
        mask = metadata <= self.absolute_threshold

        if self.keep_top:
            mask = ~mask

        # filter tiles
        x = x[mask]

        return x 


class CombTileLevelFeaturesDataset(Dataset):
    """
    Dataset which combines multiple feature sources.
    For example, we may have tile features derived from various foundation models.
    This class concatenates the data sources into a single feature tensor.
    The order of tiles across each data source are not preserved!
    """

    def __init__(
        self,
        feature_dir_path_list,
        targets_dict,
        features_source_list: List[str],
        n_tiles: Optional[int] = 1000,
        sample_keys_set = None,
        filter_regex: str = '.*',
    ) -> None:
        
        # init arguments
        self.feature_dir_path_list = feature_dir_path_list
        self.targets_dict = copy.deepcopy(targets_dict)
        self.features_source_list = features_source_list
        self.sample_keys_set = sample_keys_set
        self.n_tiles = n_tiles
        self.filter_regex = filter_regex

        # use features_source_list to separate the feature_dir_path_list into feature groups
        self.feature_source_dict= {}
        for feature_source in self.features_source_list:
            self.feature_source_dict[feature_source] = [
                feature_dir for feature_dir in feature_dir_path_list if feature_source in feature_dir
            ]
        
        # ensure no feature_dir is repeated/lost acorss the groups
        assert set(self.feature_dir_path_list) == set(
            [item for sublist in self.feature_source_dict.values() for item in sublist]
        ), "Init features_source_list must match that of features groups"
            
        # create a separate dataset for each feature group
        self.feature_dataset_dict = {}
        for feature_source, feature_dirs in self.feature_source_dict.items():
            feature_dataset = TileLevelFeaturesDataset(
                feature_dir_path_list=feature_dirs,
                targets_dict=self.targets_dict,
                sample_keys_set=self.sample_keys_set, 
                n_tiles=self.n_tiles,
                filter_regex=self.filter_regex
            )
            self.feature_dataset_dict[feature_source] = feature_dataset

        # getting common samples cross all feature sets
        common_samples = set.intersection(
            *[set(dataset.sample_id_idx_map.keys()) for dataset in self.feature_dataset_dict.values()]
        )

        # creating index to sample map
        self.idx_sample_id_map = {idx: sample for idx, sample in enumerate(common_samples)}

    def __len__(self):
        return len(self.idx_sample_id_map)
    
    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:

        # get the sample id associated with the provided index
        sample_id = self.idx_sample_id_map[index]

        # keeping track of features derived from each source
        sample_features_list = []

        # iterating through feature sources and calling getitem of each respective dataset
        for feature_source in self.features_source_list:
            dataset = self.feature_dataset_dict[feature_source]
            dataset_sample_idx = dataset.sample_id_idx_map[sample_id]
            x, y, _ = dataset.__getitem__(index=dataset_sample_idx)
            sample_features_list.append(x)

        # combining the features from all sources into a single tensor
        x = torch.cat(tuple(sample_features_list), dim=1)

        return x, y, sample_id

    def _show_data_stats(self):
        pass#TODO: FIX
        # targets_array = np.array(list(self.feature_dataset_dictvaluself.idx_sample_target_map.values()))
        # print(f"Number of samples: {self.__len__()}")
        # print(f"Target means: {np.mean(targets_array, axis=0)}")


class OrderedCombTileLevelFeaturesDataset(Dataset):
    """
    Dataset which combines multiple feature sources.
    For example, we may have tile features derived from various foundation models.
    This class concatenates the data sources into a single feature tensor.

    # NOTE: ASSUMPTION: feature sources can be uniquely identified via feature diretcory path
    # NOTE: ASSUMPTION: tiles order is presevred across the feature sources
    """

    def __init__(
        self,
        feature_dir_path_list,
        targets_dict,
        features_source_list: List[str],
        sample_keys_set = None,
        n_tiles: Optional[int] = None,
        filter_regex: str = '.*',
    ) -> None:
        
        # init arguments
        self.feature_dir_path_list = feature_dir_path_list
        self.targets_dict = copy.deepcopy(targets_dict)
        self.features_source_list = features_source_list
        self.sample_keys_set = sample_keys_set
        self.n_tiles = n_tiles
        self.filter_regex = filter_regex

        # use features_source_list to separate the feature_dir_path_list into feature groups
        self.feature_source_dict= {}
        for feature_source in self.features_source_list:
            self.feature_source_dict[feature_source] = [
                feature_dir for feature_dir in feature_dir_path_list if feature_source in feature_dir
            ]
        
        # ensure no feature_dir is repeated/lost acorss the groups
        assert set(self.feature_dir_path_list) == set(
            [item for sublist in self.feature_source_dict.values() for item in sublist]
        ), "Init features_source_list must match that of features groups"
            
        # create a separate dataset for each feature group
        self.feature_dataset_dict = {}
        for feature_source, feature_dirs in self.feature_source_dict.items():
            feature_dataset = TileLevelFeaturesDataset(
                feature_dir_path_list=feature_dirs,
                targets_dict=self.targets_dict,
                sample_keys_set=self.sample_keys_set, 
                n_tiles=None,
                filter_regex=self.filter_regex
            )
            self.feature_dataset_dict[feature_source] = feature_dataset

        # getting common samples cross all feature sets
        common_samples = set.intersection(
            *[set(dataset.sample_id_idx_map.keys()) for dataset in self.feature_dataset_dict.values()]
        )

        # filtering out samples with inconsistent tile counts
        dataset_samples = []
        for sample_id in common_samples:
            sample_features_list = []
            for feature_source in self.features_source_list:
                dataset = self.feature_dataset_dict[feature_source]
                x, _, _ = dataset.__getitem__(index=dataset.sample_id_idx_map[sample_id])
                sample_features_list.append(x)

            if len(set([x.shape[0] for x in sample_features_list])) == 1:
                dataset_samples.append(sample_id)

        # creating index to sample map
        self.idx_sample_id_map = {idx: sample for idx, sample in enumerate(dataset_samples)}

    def __len__(self):
        return len(self.idx_sample_id_map)
    
    def __getitem__(self, index) -> Tuple[torch.tensor, List[float], str]:

        # get the sample id associated with the provided index
        sample_id = self.idx_sample_id_map[index]

        # keeping track of features derived from each source
        sample_features_list = []

        # iterating through feature sources and calling getitem of each respective dataset
        for feature_source in self.features_source_list:
            dataset = self.feature_dataset_dict[feature_source]
            dataset_sample_idx = dataset.sample_id_idx_map[sample_id]
            x, y, _ = dataset.__getitem__(index=dataset_sample_idx)
            sample_features_list.append(x)

        # combining the features from all sources into a single tensor
        x = torch.cat(tuple(sample_features_list), dim=1)

        # randomly sampling the tiles when there is a required tile count
        x = sample_tiles_from_tensor(all_tiles_tensor=x, target_tiles_count=self.n_tiles)

        return x, y, sample_id

    def _show_data_stats(self):
        pass#TODO: FIX
        # targets_array = np.array(list(self.feature_dataset_dictvaluself.idx_sample_target_map.values()))
        # print(f"Number of samples: {self.__len__()}")
        # print(f"Target means: {np.mean(targets_array, axis=0)}")

class PtLevelMultiWSITileDataset(TileLevelFeaturesDataset):

    STAINS = ['_HE', "_PAS", "_Jones", "_Masson"]

    def __init__(
        self,
        feature_dir_path_list: List[str],
        targets_dict: Dict,
        sample_keys_set = None,
        n_tiles: Optional[int] = None,
        stain_balanced: bool = False,
        allow_identical_targets: bool = True,
        filter_regex: str = '.*',
    ) -> None:
        
        self.feature_dir_path_list = feature_dir_path_list
        self.targets_dict = targets_dict
        self.n_tiles = n_tiles
        self.stain_balanced = stain_balanced
        self.sample_keys_set = sample_keys_set
        self.allow_identical_targets = allow_identical_targets
        self.filter_regex = filter_regex
        self.filter_regex_pattern = re.compile(self.filter_regex)

        # filtering dataset for the provided sample keys (e.g. only train samples)
        if self.sample_keys_set is not None:
            self.sample_keys_set = set(self.sample_keys_set)
            self.targets_dict = {
                k: v for k, v in self.targets_dict.items() if k in self.sample_keys_set
            }

        # list of all feature files
        self.feature_files = []
        for dir_path in self.feature_dir_path_list:
            self.feature_files.extend(get_files_in_dir(dir_path=dir_path))

        # filtering using regext expression
        if self.filter_regex_pattern is not None:
            self.feature_files = [
                file for file in self.feature_files if self.filter_regex_pattern.match(str(file))
            ]

        # mapping sample targets, features, and names
        self.idx_sample_feature_paths_map = {} # int -> List[str]
        self.idx_sample_target_map = {} # int -> List[float]
        self.idx_sample_id_map = {} # int -> str
        self.sample_id_idx_map = {} # str -> int

        # populating the dataset
        sample_idx = 0
        for sample_id in self.targets_dict:
            feature_file_paths = [file for file in self.feature_files if sample_id in str(file)]
            if len(feature_file_paths) == 0:
                continue

            self.idx_sample_feature_paths_map[sample_idx] = feature_file_paths
            self.idx_sample_id_map[sample_idx] = sample_id
            self.sample_id_idx_map[sample_id] = sample_idx
            self.idx_sample_target_map[sample_idx] = self.targets_dict[sample_id]
            self.idx_sample_feature_paths_map[sample_idx] = [
                file for file in self.feature_files if sample_id in str(file)
            ]
            sample_idx += 1

        # ensure consistent dimensions across the targets
        target_dim_set = set([len(target) for target in self.idx_sample_target_map.values()])
        assert len(target_dim_set) == 1, f"All targets much have same dims. Got {target_dim_set}"
        self.target_dim = list(target_dim_set)[0]

        if not self.allow_identical_targets:
            print("Checking for identical targets...")
            unique_targets = set([str(v) for v in self.idx_sample_target_map.values()])
            print(f"Unique targets: {unique_targets}")
            if len(unique_targets) == 1:
                raise RuntimeError(
                    "All targets are identical. Please set allow_identical_targets=True to proceed."
                )

    def __len__(self):
        return len(self.idx_sample_id_map)
    
    def __getitem__(self, index):

        sample_id = self.idx_sample_id_map[index]
        feature_files = self.idx_sample_feature_paths_map[index]
        y = torch.Tensor(self.idx_sample_target_map[index]).float()

        if self.stain_balanced:
            stain_files = {}
            for stain in self.STAINS:
                stain_files[stain] = [file for file in feature_files if stain in str(file)]
            stain_counts = {stain: len(files) for stain, files in stain_files.items()}
            max_count = max(stain_counts.values())

            oversampled_stain_files = {}
            for stain, files in stain_files.items():
                if files and len(files) < max_count:
                    # Randomly sample with replacement to match max_count
                    oversampled_stain_files[stain] = files + random.choices(files, k=max_count - len(files), )
                else:
                    oversampled_stain_files[stain] = files  # No need to oversample if already at max

            feature_files = []
            for files in oversampled_stain_files.values():
                feature_files.extend(files)

        # load features from all files
        features_list = []
        for feature_file in feature_files:
            features_list.append(torch.load(feature_file, map_location="cpu", weights_only=True))
        x = torch.cat(tensors=features_list, dim=0)

        # sample tiles
        x = sample_tiles_from_tensor(all_tiles_tensor=x, target_tiles_count=self.n_tiles)
        
        return x, y, sample_id
    
    def _show_data_stats(self):
        print("Dataset statistics:")
        print(f"\tn samples: {len(self)}")
        print(f"\tmin instances: {min([len(v) for v in self.idx_sample_feature_paths_map.values()])}")
        print(f"\tmax instances: {max([len(v) for v in self.idx_sample_feature_paths_map.values()])}")
        print(f"\tavg instances: {np.mean([len(v) for v in self.idx_sample_feature_paths_map.values()])}\n")



def sample_tiles_from_tensor(all_tiles_tensor: torch.Tensor, target_tiles_count: Optional[int]):

    # return input tensor if there is no target tile count
    if target_tiles_count is None:
        return all_tiles_tensor
    
    # validate target_tiles_count
    assert target_tiles_count > 0, "The target tile count shoudl be positive."

    # get number of tiles within the tensor
    sample_n_tiles = all_tiles_tensor.size(0)

    # how many times does the tensor need to be entirely copied
    multiplier = target_tiles_count // sample_n_tiles

    # how many tiles need to be sampled
    n_random_sample = target_tiles_count % sample_n_tiles

    # sample remainder of the tiles without replacement
    sampled_tile_idxs = torch.randperm(sample_n_tiles, )[:n_random_sample]

    # constructing the tensor with sampled tiles
    if multiplier > 0:
        output = all_tiles_tensor.repeat(multiplier, 1)
        output = torch.cat((output, all_tiles_tensor[sampled_tile_idxs]), dim=0)
    else:
        output = all_tiles_tensor[sampled_tile_idxs]

    return output


# ______________________________________  Dataset Setup ____________________________________________

DATASET_MAP = {
    "WSILevelFeatureDataset": WSILevelFeatureDataset,
    "TileLevelFeaturesDataset": TileLevelFeaturesDataset,
    "CombTileLevelFeaturesDataset": CombTileLevelFeaturesDataset,
    "OrderedCombTileLevelFeaturesDataset": OrderedCombTileLevelFeaturesDataset,
    "TileLevelFeaturesMetaDataFilterDataset": TileLevelFeaturesMetaDataFilterDataset,

    "PtLevelMultiWSITileDataset": PtLevelMultiWSITileDataset,
}


def setup_dataset(
    dataset_type,
    feature_dir_path_list,
    targets_dict,
    sample_keys_set = None,
    dataset_init_args = {},
) -> Dataset:

    if dataset_type not in DATASET_MAP:
        raise ValueError(f"{dataset_type} dataset not founds. Select from {list(DATASET_MAP.keys())}")
    
    return DATASET_MAP[dataset_type](
        feature_dir_path_list=feature_dir_path_list,
        targets_dict=targets_dict,
        sample_keys_set=sample_keys_set,
        **dataset_init_args
    )