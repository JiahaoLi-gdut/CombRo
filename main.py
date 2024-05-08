# Author: Jiahao Li
import argparse

from node import Node
from model import Model
from datasets import *
from experiment import *
from visualization import *
from recorder import *

import torch.nn as tnn

class GumbelSoftmax():
  def __init__(self, tau = 0.5, hard = False, eps = 1e-10, dim = -1):
    self.tau = tau
    self.hard = hard
    self.eps = eps
    self.dim = dim

  def __call__(self, x):
    return tnf.gumbel_softmax(x, self.tau, self.hard, self.eps, self.dim)

class NeuralTree():
  def __init__(self,
    input_width,
    input_height,
    classes_number,
    router_channel_number = 128,
    transformer_channel_number = 256,
    gumbel_softmax_tau = 0.5,
    gumbel_softmax_hard = False,
    gumbel_softmax_eps = 1e-10,
    gumbel_softmax_milestones = None,
    gumbel_softmax_tau_decay = 1
  ):
    self.epoch = 0
    self.milestones_index = 0
    self.gumbel_softmax_tau = gumbel_softmax_tau
    self.gumbel_softmax_milestones = gumbel_softmax_milestones
    self.gumbel_softmax_tau_decay = gumbel_softmax_tau_decay
    self.gumbel_softmax = GumbelSoftmax(gumbel_softmax_tau, gumbel_softmax_hard, gumbel_softmax_eps)
    self.root = Node(content = {
      "prev_transform" : (
        True,
        tnn.Conv2d(3, transformer_channel_number // 4, kernel_size = (3, 3), padding = 1),
        tnn.BatchNorm2d(transformer_channel_number // 4),
        tnn.ReLU(True),
        tnn.Conv2d(transformer_channel_number // 4, transformer_channel_number // 4, kernel_size = (3, 3), padding = 1),
        tnn.BatchNorm2d(transformer_channel_number // 4),
        tnn.ReLU(True),
        tnn.MaxPool2d(2)
      ),
      "prev_router" : (
        True,
        tnn.Conv2d(transformer_channel_number // 4, router_channel_number, kernel_size = (3, 3), padding = 1),
        tnn.ReLU(True),
        tnn.Conv2d(router_channel_number, router_channel_number, kernel_size = (3, 3), padding = 1),
        tnn.ReLU(True),
        tnn.AdaptiveAvgPool2d((3, 3)),
        lambda x: torch.reshape(x, (-1, router_channel_number * 9)),
        tnn.Linear(router_channel_number * 9, 3),
        self.gumbel_softmax
      )
    })
    self.nodes_L1 = [
      Node(content = {
        "prev_transform" : (
          True,
          tnn.Conv2d(transformer_channel_number // 4, transformer_channel_number // 2, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number // 2),
          tnn.ReLU(True),
          tnn.Conv2d(transformer_channel_number // 2, transformer_channel_number // 2, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number // 2),
          tnn.ReLU(True),
          tnn.MaxPool2d(2)
        ),
        "prev_router" : (
          True,
          tnn.Conv2d(transformer_channel_number // 2, router_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.ReLU(True),
          tnn.Conv2d(router_channel_number, router_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.ReLU(True),
          tnn.AdaptiveAvgPool2d((3, 3)),
          lambda x: torch.reshape(x, (-1, router_channel_number * 9)),
          tnn.Linear(router_channel_number * 9, 3),
          self.gumbel_softmax
        )
      }) for _ in range(2)
    ]
    self.nodes_L2 = [
      Node(content = {
        "prev_transform" : (
          True,
          tnn.Conv2d(transformer_channel_number // 2, transformer_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number),
          tnn.ReLU(True),
          tnn.Conv2d(transformer_channel_number, transformer_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number),
          tnn.ReLU(True),
          tnn.MaxPool2d(2)
        ),
        "prev_router" : (
          True,
          tnn.Conv2d(transformer_channel_number, router_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.ReLU(True),
          tnn.Conv2d(router_channel_number, router_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.ReLU(True),
          tnn.AdaptiveAvgPool2d((3, 3)),
          lambda x: torch.reshape(x, (-1, router_channel_number * 9)),
          tnn.Linear(router_channel_number * 9, 3),
          self.gumbel_softmax
        )
      }) for _ in range(4)
    ]
    self.nodes_L3 = [
      Node(chr(i + 65), content = {
        "prev_transform" : (
          True,
          tnn.Conv2d(transformer_channel_number, transformer_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number),
          tnn.ReLU(True),
          tnn.Conv2d(transformer_channel_number, transformer_channel_number, kernel_size = (3, 3), padding = 1),
          tnn.BatchNorm2d(transformer_channel_number),
          tnn.ReLU(True),
          tnn.MaxPool2d(2)
        ),
        "solver" : (
          True,
          lambda x: torch.reshape(x, (-1, (input_width // 16) * (input_height // 16) * transformer_channel_number)),
          tnn.Linear((input_width // 16) * (input_height // 16) * transformer_channel_number, classes_number),
          tnn.LogSoftmax(dim = 1)
        )
      }) for i in range(8)
    ]
    self.root.attach_neighbor(self.nodes_L1[0], 1, 0) # left
    self.root.attach_neighbor(self.nodes_L1[1], 1, 1) # right
    self.nodes_L1[0].attach_neighbor(self.nodes_L2[0], 1, 0) # left
    self.nodes_L1[0].attach_neighbor(self.nodes_L2[1], 1, 1) # right
    self.nodes_L1[1].attach_neighbor(self.nodes_L2[2], 1, 0) # left
    self.nodes_L1[1].attach_neighbor(self.nodes_L2[3], 1, 1) # right
    self.nodes_L2[0].attach_neighbor(self.nodes_L3[0], 1, 0) # left
    self.nodes_L2[0].attach_neighbor(self.nodes_L3[1], 1, 1) # right
    self.nodes_L2[1].attach_neighbor(self.nodes_L3[2], 1, 0) # left
    self.nodes_L2[1].attach_neighbor(self.nodes_L3[3], 1, 1) # right
    self.nodes_L2[2].attach_neighbor(self.nodes_L3[4], 1, 0) # left
    self.nodes_L2[2].attach_neighbor(self.nodes_L3[5], 1, 1) # right
    self.nodes_L2[3].attach_neighbor(self.nodes_L3[6], 1, 0) # left
    self.nodes_L2[3].attach_neighbor(self.nodes_L3[7], 1, 1) # right

  def step(self):
    self.epoch += 1
    if  self.gumbel_softmax_milestones \
    and self.gumbel_softmax_milestones[self.milestones_index] == self.epoch:
      self.gumbel_softmax.tau *= self.gumbel_softmax_tau_decay
      self.milestones_index += 1

  def reset(self):
    self.epoch = 0
    self.milestones_index = 0
    self.gumbel_softmax.tau = self.gumbel_softmax_tau

  def get_root(self):
    return self.root

  def get_leaves(self):
    return self.nodes_L3

def set_device(cuda = None, gpu = 0):
  # get cpu or gpu device for training.
  if cuda:
    if torch.cuda.is_available():
      device = "cuda"
      torch.cuda.set_device(gpu)
    else:
      raise RuntimeError("CUDA is unavailable!")
  else:
    if cuda is None and torch.cuda.is_available():
      device = "cuda"
      torch.cuda.set_device(gpu)
    elif torch.backends.mps.is_available():
      device = "mps"
    else:
      device = "cpu"
  return device

def experiment(
  record_directory = "",
  dataset_root = "",
  dataset_name = "CIFAR100",
  router_filter_number = 128,
  transformer_filter_number = 256,
  gumbel_softmax_tau = 0.5,
  gumbel_softmax_hard = False,
  gumbel_softmax_eps = 1e-10,
  gumbel_softmax_milestones = None,
  gumbel_softmax_tau_decay = 1,
  datloader1_train_batsz = 128,
  datloader1_valid_batsz = 100,
  datloader1_test_batsz = 100,
  datloader1_train_num_workers = 0,
  datloader1_valid_num_workers = 0,
  datloader1_test_num_workers = 0,
  optimizer1_epochs = 100,
  optimizer1_record_last = True,
  optimizer1_sgd_or_adamw = False,
  optimizer1_lr = 1e-3,
  optimizer1_weight_decay = 1e-2,
  optimizer1_sgd_momentum = 0.9,
  optimizer1_sgd_dampening = 0,
  optimizer1_sgd_nesterov = True,
  optimizer1_adamw_betas = (0.9, 0.999),
  optimizer1_adamw_eps = 1e-8,
  optimizer1_adamw_amsgrad = True,
  scheduler1_milestones = [30, 50, 70, 90],
  scheduler1_gamma = 0.5,
  datloader2_train_batsz = 128,
  datloader2_valid_batsz = 100,
  datloader2_test_batsz = 100,
  datloader2_train_num_workers = 0,
  datloader2_valid_num_workers = 0,
  datloader2_test_num_workers = 0,
  optimizer2_epochs = 100,
  optimizer2_record_last = True,
  optimizer2_sgd_or_adamw = False,
  optimizer2_lr = 1e-3,
  optimizer2_weight_decay = 1e-2,
  optimizer2_sgd_momentum = 0.9,
  optimizer2_sgd_dampening = 0,
  optimizer2_sgd_nesterov = True,
  optimizer2_adamw_betas = (0.9, 0.999),
  optimizer2_adamw_eps = 1e-8,
  optimizer2_adamw_amsgrad = True,
  scheduler2_milestones = [30, 50, 70, 90],
  scheduler2_gamma = 0.5,
  search_combs_or_leaves = True,
  hard_or_soft_search = False,
  search_threshold = 0.0,
  reinitialization = True,
  count_params_or_not = True,
  count_flops_or_not = False,
  draw_heatmap_or_not = False,
  draw_status_or_not = True,
  draw_format = "pdf",
  cuda = None,
  gpu = 0,
  non_blocking = True
):
  # set device
  device = set_device(cuda, gpu)
  print(f"Using {device} device")

  # create dataset
  if dataset_name == "CIFAR10":
    train_dataset, valid_dataset, test_dataset = create_cifar10_datasets(dataset_root)
    input_channel = 3
    input_width = 32
    input_height = 32
    classes_number = 10
  elif dataset_name == "CIFAR100":
    train_dataset, valid_dataset, test_dataset = create_cifar100_datasets(dataset_root)
    input_channel = 3
    input_width = 32
    input_height = 32
    classes_number = 100
  elif dataset_name == "TinyImageNet":
    train_dataset, valid_dataset, test_dataset = create_tiny_image_net_datasets(dataset_root)
    input_channel = 3
    input_width = 64
    input_height = 64
    classes_number = 200
  else:
    raise ValueError("Unavailable dataset name!")

  # create neural tree
  neural_tree = NeuralTree(
    input_width,
    input_height,
    classes_number,
    router_filter_number,
    transformer_filter_number,
    gumbel_softmax_tau,
    gumbel_softmax_hard,
    gumbel_softmax_eps,
    gumbel_softmax_milestones,
    gumbel_softmax_tau_decay
  )

  # create model for phase 1
  nested_ensemble_model = Model(
    neural_tree.get_root(),
    neural_tree.get_leaves(),
    handle = 2, mulfun = False
  ).to(device, non_blocking = non_blocking)

  # create data loaders for phase 1
  train_loader, valid_loader, test_loader = create_data_loaders(
    train_dataset, valid_dataset, test_dataset,
    datloader1_train_batsz, datloader1_valid_batsz, datloader1_test_batsz,
    datloader1_train_num_workers, datloader1_valid_num_workers, datloader1_test_num_workers
  )

  # create optimizer for phase 1
  if optimizer1_sgd_or_adamw:
    optimizer = torch.optim.SGD(
      nested_ensemble_model.parameters(),
      optimizer1_lr,
      optimizer1_sgd_momentum,
      optimizer1_sgd_dampening,
      optimizer1_weight_decay,
      optimizer1_sgd_nesterov
    )
  else:
    optimizer = torch.optim.AdamW(
      nested_ensemble_model.parameters(),
      optimizer1_lr,
      optimizer1_adamw_betas,
      optimizer1_adamw_eps,
      optimizer1_weight_decay,
      optimizer1_adamw_amsgrad
    )

  # create scheduler for phase 1
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    scheduler1_milestones,
    scheduler1_gamma
  )

  # create echo indexes for phase 1
  echo_indexes = get_echo_indexes(len(train_loader), 4)

  # optimization of phase 1
  results_phase1 = []
  for t in range(optimizer1_epochs):
    print(f"Dataset: {dataset_name} - Nested Ensemble Phase - Epoch {t + 1}\n-------------------------------")
    train_results = nested_ensemble_train_on_nll_and_var(
      nested_ensemble_model, train_loader, optimizer,
      device, non_blocking, 1., 0., optimizer1_record_last, echo_indexes
    )
    valid_results = nested_ensemble_eval_on_nll_and_var(
      nested_ensemble_model, valid_loader,
      device, non_blocking, 1., 0., echo_end = "(Valid)\n"
    )
    test_results = nested_ensemble_eval_on_nll_and_var(
      nested_ensemble_model, test_loader,
      device, non_blocking, 1., 0., echo_end = "(Test)\n\n"
    )
    neural_tree.step()
    scheduler.step()
    # record
    if optimizer2_record_last:
      average_nllpr = train_results[0]
      average_envar = train_results[1]
      average_error = train_results[2]
    else:
      average_nllpr = None
      average_envar = None
      average_error = None
      for nllpr, envar, error in train_results:
        if average_nllpr is None:
          average_nllpr = nllpr
          average_envar = envar
          average_error = error
        else:
          average_nllpr += nllpr
          average_envar += envar
          average_error += error
      results_count = len(train_results)
      average_nllpr /= results_count
      average_envar /= results_count
      average_error /= results_count
    results_phase1.append((
      average_nllpr, average_envar, average_error,
      valid_results[0], valid_results[1], valid_results[2], valid_results[3],
       test_results[0],  test_results[1],  test_results[2],  test_results[3]
    ))

  # draw heatmap
  # if draw_heatmap_or_not and record_directory is not None:
  #   rdict = vote_counting_with_labels_for_leaf_combinations(nested_ensemble_model, train_loader, device, non_blocking)
  #   draw_heatmap(rdict, None, "label_architecture_heatmap", record_directory, draw_format)
  rdict, grbuf, _ = vote_counting_with_labels_for_leaf_combinations_and_node_routing(nested_ensemble_model, train_loader, device, non_blocking)
  lines = []
  for block in grbuf:
    if block[3]:
      for value, (sumoh, sumpr) in block[3].items():
        lines.append(f"{block[1].name} {value} {sumpr[0].item():>7f} {sumpr[1].item():>7f} {sumpr[2].item():>7f}\n")
  lines.append("\n")
  file_path = os.path.join(record_directory, "route.txt")
  directory = os.path.dirname(file_path)
  if not os.path.exists(directory):
    os.makedirs(directory)
  with open(file_path, 'a', encoding = "utf-8") as file:
    file.writelines(lines)

  # get architecture with most probability
  if search_combs_or_leaves:
    combs, total_prob = get_multiple_leaf_combinations(
      nested_ensemble_model, train_loader, device,
      non_blocking, True, hard_or_soft_search, search_threshold
    )
  else:
    leafs, total_prob = get_multiple_leaves(
      nested_ensemble_model, train_loader, device,
      non_blocking, True, hard_or_soft_search, search_threshold
    )
    combs = [(leaf,) for leaf in leafs]
  if 1 < len(combs):
    union = combs[0][0].union(combs[1][0])
    for nodes, _, _ in combs[2:]:
      union = union.union(nodes)
    print(f"{len(combs)} combinations are selected because they are the minimal set with a probability sum {total_prob:>7f} that is not less than {search_threshold}.")
  elif 0 < len(combs):
    union = combs[0][0]
    print(f"{len(combs)} combination is selected because it is the minimal set with a probability sum {total_prob:>7f} that is not less than {search_threshold}.")
  else:
    union = None
  for index, (nodes, sumoh, sumpr) in enumerate(combs):
    print(f"{index}. {leaves_to_string(nodes)} | Hard: {sumoh} | Soft: {sumpr} \n")
  if hard_or_soft_search:
    weights = [weight for _, weight, _ in combs]
  else:
    weights = [weight for _, _, weight in combs]
  wsum = sum(weights)
  weights = [weight / wsum for weight in weights]
  combs = [nodes for nodes, _, _ in combs]

  # create model for phase 2
  ensemble_model = Model(
    neural_tree.get_root(), union,
    handle = 3, mulfun = False
  ).to(device, non_blocking = non_blocking)

  # count params of model for phase 2
  if count_params_or_not:
    nparams = count_model_params(ensemble_model)
    print(f"There are(is) {nparams} parameters in the resulting model.")
  else:
    nparams = None

  # count FLOPs of model for phase 2
  if count_flops_or_not:
    input = torch.rand(1, input_channel, input_height, input_width).to(device)
    old_mode = ensemble_model.get_handle_mode()
    ensemble_model.set_mode_as_multiple_instances_without_graph()
    nflops = count_model_flops(ensemble_model, input)
    ensemble_model.set_handle_mode(old_mode)
    print(f"There are(is) {nflops} FLOps in the resulting model.")
  else:
    nflops = None

  # perform reinitialization on model
  if reinitialization:
    ensemble_model.reset_parameters()

  # create data loaders for phase 2
  train_loader, valid_loader, test_loader = create_data_loaders(
    train_dataset, valid_dataset, test_dataset,
    datloader2_train_batsz, datloader2_valid_batsz, datloader2_test_batsz,
    datloader2_train_num_workers, datloader2_valid_num_workers, datloader2_test_num_workers
  )

  # create optimizer for phase 2
  if optimizer2_sgd_or_adamw:
    optimizer = torch.optim.SGD(
      nested_ensemble_model.parameters(),
      optimizer2_lr,
      optimizer2_sgd_momentum,
      optimizer2_sgd_dampening,
      optimizer2_weight_decay,
      optimizer2_sgd_nesterov
    )
  else:
    optimizer = torch.optim.AdamW(
      nested_ensemble_model.parameters(),
      optimizer2_lr,
      optimizer2_adamw_betas,
      optimizer2_adamw_eps,
      optimizer2_weight_decay,
      optimizer2_adamw_amsgrad
    )

  # create scheduler for phase 2
  scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer,
    scheduler2_milestones,
    scheduler2_gamma
  )

  # create echo indexes for phase 2
  echo_indexes = get_echo_indexes(len(train_loader), 4)

  # optimization of phase 2
  results_phase2 = []
  for t in range(optimizer2_epochs):
    print(f"Dataset: {dataset_name} - Ensemble Phase - Epoch {t + 1}\n-------------------------------")
    train_results = partial_ensemble_train_on_nll(
      ensemble_model, train_loader, combs, optimizer,
      device, non_blocking, weights, 1., optimizer2_record_last, echo_indexes
    )
    valid_results = partial_ensemble_eval_on_nll(
      ensemble_model, valid_loader, combs,
      device, non_blocking, weights, 1., echo_end = "(Valid)\n"
    )
    test_results = partial_ensemble_eval_on_nll(
      ensemble_model, test_loader, combs,
      device, non_blocking, weights, 1., echo_end = "(Test)\n\n"
    )
    scheduler.step()
    # record
    if optimizer2_record_last:
      average_nllpr = train_results[0]
      average_error = train_results[1]
    else:
      average_nllpr = None
      average_error = None
      for nllpr, error in train_results:
        if average_nllpr is None:
          average_nllpr = nllpr
          average_error = error
        else:
          average_nllpr += nllpr
          average_error += error
      results_count = len(train_results)
      average_nllpr /= results_count
      average_error /= results_count
    results_phase2.append((
      average_nllpr, average_error,
      valid_results[0], valid_results[1], valid_results[2],
       test_results[0],  test_results[1],  test_results[2]
    ))

  if draw_status_or_not and record_directory is not None:
    draw_train_metrics(results_phase1, results_phase2, None, "train_phase_status", record_directory, draw_format)

  return nparams, nflops, results_phase1[-1][-1].item(), results_phase2[-1][-1].item(), total_prob, union

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description = 'Combinatorial Routing')
  parser.add_argument("--record_directory", type = str, default = "./")
  parser.add_argument("--dataset_root", type = str, default = "./")
  parser.add_argument("--dataset_name", type = str, default = "CIFAR100")
  parser.add_argument("--router_filter_number", type = int, default = 128)
  parser.add_argument("--transformer_filter_number", type = int, default = 256)
  parser.add_argument("--gumbel_softmax_tau", type = float, default = 0.5)
  parser.add_argument("--gumbel_softmax_hard", type = bool, default = False, action = argparse.BooleanOptionalAction)
  parser.add_argument("--gumbel_softmax_eps", type = float, default = 1e-10)
  parser.add_argument("--gumbel_softmax_milestones", type = int, nargs = "*", default = [])
  parser.add_argument("--gumbel_softmax_tau_decay", type = float, default = 1.0)
  parser.add_argument("--datloader1_train_batsz", type = int, default = 512)
  parser.add_argument("--datloader1_valid_batsz", type = int, default = 100)
  parser.add_argument("--datloader1_test_batsz", type = int, default = 100)
  parser.add_argument("--datloader1_train_num_workers", type = int, default = 0)
  parser.add_argument("--datloader1_valid_num_workers", type = int, default = 0)
  parser.add_argument("--datloader1_test_num_workers", type = int, default = 0)
  parser.add_argument("--optimizer1_epochs", type = int, default = 100)
  parser.add_argument("--optimizer1_record_last", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer1_sgd_or_adamw", type = bool, default = False, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer1_lr", type = float, default = 1e-3)
  parser.add_argument("--optimizer1_weight_decay", type = float, default = 1e-2)
  parser.add_argument("--optimizer1_sgd_momentum", type = float, default = 0.9)
  parser.add_argument("--optimizer1_sgd_dampening", type = float, default = 0.0)
  parser.add_argument("--optimizer1_sgd_nesterov", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer1_adamw_betas", type = float, nargs = "*", default = [0.9, 0.999])
  parser.add_argument("--optimizer1_adamw_eps", type = float, default = 1e-8)
  parser.add_argument("--optimizer1_adamw_amsgrad", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--scheduler1_milestones", type = int, nargs = "*", default = [30, 50, 70, 90])
  parser.add_argument("--scheduler1_gamma", type = float, default = 0.5)
  parser.add_argument("--datloader2_train_batsz", type = int, default = 512)
  parser.add_argument("--datloader2_valid_batsz", type = int, default = 100)
  parser.add_argument("--datloader2_test_batsz", type = int, default = 100)
  parser.add_argument("--datloader2_train_num_workers", type = int, default = 0)
  parser.add_argument("--datloader2_valid_num_workers", type = int, default = 0)
  parser.add_argument("--datloader2_test_num_workers", type = int, default = 0)
  parser.add_argument("--optimizer2_epochs", type = int, default = 100)
  parser.add_argument("--optimizer2_record_last", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer2_sgd_or_adamw", type = bool, default = False, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer2_lr", type = float, default = 1e-3)
  parser.add_argument("--optimizer2_weight_decay", type = float, default = 1e-2)
  parser.add_argument("--optimizer2_sgd_momentum", type = float, default = 0.9)
  parser.add_argument("--optimizer2_sgd_dampening", type = float, default = 0.0)
  parser.add_argument("--optimizer2_sgd_nesterov", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--optimizer2_adamw_betas", type = float, nargs = "*", default = [0.9, 0.999])
  parser.add_argument("--optimizer2_adamw_eps", type = float, default = 1e-8)
  parser.add_argument("--optimizer2_adamw_amsgrad", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--scheduler2_milestones", type = int, nargs = "*", default = [30, 50, 70, 90])
  parser.add_argument("--scheduler2_gamma", type = float, default = 0.5)
  parser.add_argument("--search_combs_or_leaves", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--hard_or_soft_search", type = bool, default = False, action = argparse.BooleanOptionalAction)
  parser.add_argument("--search_threshold", type = float, default = 0.0)
  parser.add_argument("--reinitialization", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--count_params_or_not", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--count_flops_or_not", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--draw_heatmap_or_not", type = bool, default = False, action = argparse.BooleanOptionalAction)
  parser.add_argument("--draw_status_or_not", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--draw_format", type = str, default = "pdf")
  parser.add_argument("--cuda", type = bool, default = True, action = argparse.BooleanOptionalAction)
  parser.add_argument("--gpu", type = int, default = 0)
  parser.add_argument("--non_blocking", type = bool, default = True, action = argparse.BooleanOptionalAction)
  args = parser.parse_args()
  try:
    results = experiment(
      args.record_directory,
      args.dataset_root,
      args.dataset_name,
      args.router_filter_number,
      args.transformer_filter_number,
      args.gumbel_softmax_tau,
      args.gumbel_softmax_hard,
      args.gumbel_softmax_eps,
      args.gumbel_softmax_milestones,
      args.gumbel_softmax_tau_decay,
      args.datloader1_train_batsz,
      args.datloader1_valid_batsz,
      args.datloader1_test_batsz,
      args.datloader1_train_num_workers,
      args.datloader1_valid_num_workers,
      args.datloader1_test_num_workers,
      args.optimizer1_epochs,
      args.optimizer1_record_last,
      args.optimizer1_sgd_or_adamw,
      args.optimizer1_lr,
      args.optimizer1_weight_decay,
      args.optimizer1_sgd_momentum,
      args.optimizer1_sgd_dampening,
      args.optimizer1_sgd_nesterov,
      args.optimizer1_adamw_betas,
      args.optimizer1_adamw_eps,
      args.optimizer1_adamw_amsgrad,
      args.scheduler1_milestones,
      args.scheduler1_gamma,
      args.datloader2_train_batsz,
      args.datloader2_valid_batsz,
      args.datloader2_test_batsz,
      args.datloader2_train_num_workers,
      args.datloader2_valid_num_workers,
      args.datloader2_test_num_workers,
      args.optimizer2_epochs,
      args.optimizer2_record_last,
      args.optimizer2_sgd_or_adamw,
      args.optimizer2_lr,
      args.optimizer2_weight_decay,
      args.optimizer2_sgd_momentum,
      args.optimizer2_sgd_dampening,
      args.optimizer2_sgd_nesterov,
      args.optimizer2_adamw_betas,
      args.optimizer2_adamw_eps,
      args.optimizer2_adamw_amsgrad,
      args.scheduler2_milestones,
      args.scheduler2_gamma,
      args.search_combs_or_leaves,
      args.hard_or_soft_search,
      args.search_threshold,
      args.reinitialization,
      args.count_params_or_not,
      args.count_flops_or_not,
      args.draw_heatmap_or_not,
      args.draw_status_or_not,
      args.draw_format,
      args.cuda,
      args.gpu,
      args.non_blocking
    )
    id = record_results(results, args.record_directory)
    record_arguments(id, args, args.record_directory)
    if args.draw_heatmap_or_not:
      src_path = os.path.join(args.record_directory,  "label_architecture_heatmap."      + args.draw_format)
      tar_path = os.path.join(args.record_directory, f"label_architecture_heatmap_{id}." + args.draw_format)
      os.rename(src_path, tar_path)
    if args.draw_status_or_not:
      src_path = os.path.join(args.record_directory,  "train_phase_status."      + args.draw_format)
      tar_path = os.path.join(args.record_directory, f"train_phase_status_{id}." + args.draw_format)
      os.rename(src_path, tar_path)
  except Exception as e:
    print("Exception: ", e)
    sys.exit(1)
  sys.exit(0)
