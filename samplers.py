# Author: Jiahao Li

import random
import torch
import torch.utils.data.sampler as sampler

class VanillaChunkSampler(sampler. Sampler):
  """ Samples elements sequentially from some offset.
  Args:
    num_samples: # of desired datapoints
    start: offset where we should start selecting from
  Source:
    https://github.com/pytorch/vision/issues/168

  Examples:
    NUM_TRAIN = 49000
    NUM_VAL = 1000
    cifar10_train = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=T.ToTensor())
    loader_train = DataLoader(cifar10_train, batch_size=64, sampler=ChunkSampler(NUM_TRAIN, 0))
    cifar10_val = dset.CIFAR10('./cs231n/datasets', train=True, download=True, transform=T.ToTensor())
    loader_val = DataLoader(cifar10_val, batch_size=64, sampler=ChunkSampler(NUM_VAL, NUM_TRAIN))
  """
  def __init__(self, num_samples, start = 0, shuffle = False):
    self.num_samples = num_samples
    self.start = start
    self.shuffle = shuffle

  def __iter__(self):
    if self.shuffle:
      return iter((torch.randperm(self.num_samples) + self.start).long())
    else:
      return iter(range(self.start, self.start + self.num_samples))

  def __len__(self):
    return self.num_samples

class SlaveChunkSampler(sampler.Sampler):
  """
    reference: https://github.com/pytorch/vision/issues/168
  """
  def __init__(self, parent_sampler, group_opt_points, post_shuffle_tag = False):
    self.parent_sampler = parent_sampler
    self.group_opt_points = group_opt_points
    self.post_shuffle_tag = post_shuffle_tag
    self.update_partial_indexes()

  def __len__(self):
    return len(self.partial_indexes)

  def __iter__(self):
    if self.post_shuffle_tag:
      random.shuffle(self.partial_indexes)
    return iter(self.partial_indexes)

  def update_partial_indexes(self):
    self.partial_indexes = []
    for index in self.group_opt_points:
      if index:
        start = self.parent_sampler.group_end_points[index - 1]
      else:
        start = 0
      end = self.parent_sampler.group_end_points[index]
      self.partial_indexes.extend(self.parent_sampler.global_indexes[start : end])
    if self.post_shuffle_tag:
      random.shuffle(self.partial_indexes)

class MasterChunkSampler(sampler.Sampler):
  """
    reference: https://github.com/pytorch/vision/issues/168
  """
  def __init__(self, group_end_points, group_opt_points = [0], prev_shuffle_tag = True, post_shuffle_tag = False):
    self.group_end_points = group_end_points
    self.group_opt_points = group_opt_points
    self.prev_shuffle_tag = prev_shuffle_tag
    self.post_shuffle_tag = post_shuffle_tag
    self.child_samplers = []
    self.update_global_indexes()
    self.update_partial_indexes()

  def __len__(self):
    return len(self.partial_indexes)

  def __iter__(self):
    if self.prev_shuffle_tag:
      self.global_indexes = torch.randperm(max(self.group_end_points)).long()
      self.partial_indexes = []
      for index in self.group_opt_points:
        if index:
          start = self.group_end_points[index - 1]
        else:
          start = 0
        end = self.group_end_points[index]
        self.partial_indexes.extend(self.global_indexes[start : end])
      for child in self.child_samplers:
        child.update_partial_indexes()
    if self.post_shuffle_tag:
      random.shuffle(self.partial_indexes)
    return iter(self.partial_indexes)

  def update_global_indexes(self):
    if self.prev_shuffle_tag:
      self.global_indexes = torch.randperm(max(self.group_end_points)).long()
    else:
      self.global_indexes = range(max(self.group_end_points))
    for child in self.child_samplers:
      child.update_partial_indexes()

  def update_partial_indexes(self):
    self.partial_indexes = []
    for index in self.group_opt_points:
      if index:
        start = self.group_end_points[index - 1]
      else:
        start = 0
      end = self.group_end_points[index]
      self.partial_indexes.extend(self.global_indexes[start : end])
    if self.post_shuffle_tag:
      random.shuffle(self.partial_indexes)

  def get_slave_sampler(self, group_opt_points, post_shuffle_tag = False):
    return SlaveChunkSampler(self, group_opt_points, post_shuffle_tag)