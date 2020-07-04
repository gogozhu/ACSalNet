from collections import defaultdict
import numpy as np
import copy
import random

from torch.utils.data.sampler import Sampler


class RandomIdentitySampler(Sampler):

    def __init__(self, data_source, batch_size, num_instances):
        if batch_size % num_instances > 0:
            raise ValueError('batch_size={} must be K * num_instances={}'.format(batch_size, num_instances))
        assert len(data_source) % num_instances == 0
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.length = len(self.data_source) // self.num_instances

    def __iter__(self):
        final_idxs = []
        base_idxs = np.random.permutation(self.length) * self.num_instances
        for base_idx in base_idxs:
            # final_idxs.append(tuple((np.random.permutation(self.num_instances) + base_idx).tolist()))
            final_idxs.append(tuple((np.arange(self.num_instances) + base_idx).tolist()))

        return iter(final_idxs)

    def __len__(self):
        return self.length

# class RandomIdentitySampler(Sampler):

#     def __init__(self, data_source, batch_size, num_instances):
#         if batch_size % num_instances > 0:
#             raise ValueError('batch_size={} must be K * num_instances={}'.format(batch_size, num_instances))
#         assert len(data_source) % num_instances == 0
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.num_instances = num_instances
#         self.sort_index = np.argsort(self.data_source)
#         self.length = len(self.data_source)

#     def __iter__(self):
#         tmp_idxs = []
#         base_idxs = np.random.permutation(self.length // self.num_instances) * self.num_instances
#         for base_idx in base_idxs:
#             tmp_idxs += (np.random.permutation(self.num_instances) + base_idx).tolist()

#         final_idxs = np.array(copy.deepcopy(tmp_idxs))
#         final_idxs[self.sort_index] = tmp_idxs
#         final_idxs = final_idxs.tolist()

#         return iter([str(i) for i in final_idxs])

#     def __len__(self):
#         return self.length


