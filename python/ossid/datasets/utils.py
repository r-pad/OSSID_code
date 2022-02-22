import pickle
import torch
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler
from torch.utils.data.sampler import BatchSampler

def loadProcessZephyrResults(cfg):
    print("dtoid_bop_dataset: Loading zephyr results from:", cfg.dataset.zephyr_result_path)
    zephyr_results = pickle.load(open(cfg.dataset.zephyr_result_path, 'rb'))
    print("len(zephyr_results):", len(zephyr_results))

    if cfg.dataset.zephyr_filter_key is None or cfg.dataset.zephyr_filter_threshold is None:
        print("getDataloaders(): keeping all BOP testing targets")
    else:
        print("getDataloaders(): Re-selecting testing targets using", cfg.dataset.zephyr_filter_key, 
            "threshold =", cfg.dataset.zephyr_filter_threshold)
        # print(zephyr_results[0][cfg.dataset.zephyr_filter_key])
        zephyr_results = [r for r in zephyr_results if r[cfg.dataset.zephyr_filter_key] > cfg.dataset.zephyr_filter_threshold]
        print("getDataloaders(): remaining zephyr results =", len(zephyr_results))

    # Use only a portion of the zephyr results for finetuning if told so
    zephyr_results.sort(key = lambda x: (x['scene_id'], x['im_id']))
    if cfg.dataset.zephyr_results_percent < 1:
        n_keep = round(cfg.dataset.zephyr_results_percent * len(zephyr_results))
        zephyr_results = zephyr_results[:n_keep]
        print("getDataloaders(): Get only the first %d of the results. " % (len(zephyr_results)))

    # Split the zephyr results into training and validation set
    zephyr_results_train = [r for i, r in enumerate(zephyr_results) if i % 5 != 4]
    zephyr_results_valid = [r for i, r in enumerate(zephyr_results) if i % 5 == 4]
    print("zephyr results for training:", len(zephyr_results_train))
    print("zephyr results for validation:", len(zephyr_results_valid))

    return zephyr_results_train, zephyr_results_valid

def collate_fn(batch):
    keys = list(batch[0].keys())
    out = {}
    for k in keys:
        if k in ['qkpts', 'qkpts_warp']:
            out[k] = [torch.as_tensor(d[k]) for d in batch]
        elif batch[0][k] is None:
            out[k] = None
        else:
            out[k] = torch.stack([torch.as_tensor(d[k]) for d in batch], dim=0)

    return out

def getSampler(dataset, batch_size, shuffle=False, ttt_sampling=False):
    args = {}
    if ttt_sampling:
        print("BatchSampler: Using TTT sampler")
        if shuffle:
            batch_sampler = TTTBatchSampler(
                RandomSampler(dataset), batch_size=batch_size
            )
        else:
            batch_sampler = TTTBatchSampler(
                SequentialSampler(dataset), batch_size=batch_size
            )
        args['batch_sampler'] = batch_sampler
    else:
        print("BatchSampler: Using normal sampler")
        args['batch_size'] = batch_size
        args['shuffle'] = shuffle

    return args

class TTTBatchSampler(BatchSampler):
    '''
    This TTTBatchSampler will return the same index from each batch, 
    simply repeating that index and iterating over all the indices 
    returned by the provided Sampler class
    '''
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool = False) -> None:
        super().__init__(sampler, batch_size, drop_last)

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            for _ in range(self.batch_size):
                batch.append(idx)
            yield batch
            batch = []

    def __len__(self):
        return len(self.sampler)

def sortTargetByImage(targets):
    targets_sorted = {}
    for t in targets:
        obj_id = t['obj_id']
        scene_id = t['scene_id']
        im_id = t['im_id']
        
        k = (scene_id, im_id)
        if k not in targets_sorted:
            targets_sorted[k] = []

        targets_sorted[k].append(obj_id)
    
    return targets_sorted