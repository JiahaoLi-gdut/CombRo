# Author: Jiahao Li
import os

def arguments_to_string(id, arguments):
  string  =    "experiment_id="                + str(id)
  string +=   " record_directory=\""           + str(arguments.record_directory)
  string += "\" dataset_root=\""               + str(arguments.dataset_root)
  string += "\" dataset_name=\""               + str(arguments.dataset_name)
  string += "\" router_filter_number="         + str(arguments.router_filter_number)
  string +=   " transformer_filter_number="    + str(arguments.transformer_filter_number)
  string +=   " gumbel_softmax_tau="           + str(arguments.gumbel_softmax_tau)
  string +=   " gumbel_softmax_hard="          + str(arguments.gumbel_softmax_hard)
  string +=   " gumbel_softmax_eps="           + str(arguments.gumbel_softmax_eps)
  string +=   " gumbel_softmax_milestones="    + str(arguments.gumbel_softmax_milestones)
  string +=   " gumbel_softmax_tau_decay="     + str(arguments.gumbel_softmax_tau_decay)
  string +=   " datloader1_train_batsz="       + str(arguments.datloader1_train_batsz)
  string +=   " datloader1_valid_batsz="       + str(arguments.datloader1_valid_batsz)
  string +=   " datloader1_test_batsz="        + str(arguments.datloader1_test_batsz)
  string +=   " datloader1_train_num_workers=" + str(arguments.datloader1_train_num_workers)
  string +=   " datloader1_valid_num_workers=" + str(arguments.datloader1_valid_num_workers)
  string +=   " datloader1_test_num_workers="  + str(arguments.datloader1_test_num_workers)
  string +=   " optimizer1_epochs="            + str(arguments.optimizer1_epochs)
  string +=   " optimizer1_record_last="       + str(arguments.optimizer1_record_last)
  string +=   " optimizer1_sgd_or_adamw="      + str(arguments.optimizer1_sgd_or_adamw)
  string +=   " optimizer1_lr="                + str(arguments.optimizer1_lr)
  string +=   " optimizer1_weight_decay="      + str(arguments.optimizer1_weight_decay)
  if arguments.optimizer1_sgd_or_adamw:
    string += " optimizer1_sgd_momentum="      + str(arguments.optimizer1_sgd_momentum)
    string += " optimizer1_sgd_dampening="     + str(arguments.optimizer1_sgd_dampening)
    string += " optimizer1_sgd_nesterov="      + str(arguments.optimizer1_sgd_nesterov)
  else:
    string += " optimizer1_adamw_betas="       + str(arguments.optimizer1_adamw_betas)
    string += " optimizer1_adamw_eps="         + str(arguments.optimizer1_adamw_eps)
    string += " optimizer1_adamw_amsgrad="     + str(arguments.optimizer1_adamw_amsgrad)
  string +=   " scheduler1_milestones="        + str(arguments.scheduler1_milestones)
  string +=   " scheduler1_gamma="             + str(arguments.scheduler1_gamma)
  string +=   " datloader2_train_batsz="       + str(arguments.datloader2_train_batsz)
  string +=   " datloader2_valid_batsz="       + str(arguments.datloader2_valid_batsz)
  string +=   " datloader2_test_batsz="        + str(arguments.datloader2_test_batsz)
  string +=   " datloader2_train_num_workers=" + str(arguments.datloader2_train_num_workers)
  string +=   " datloader2_valid_num_workers=" + str(arguments.datloader2_valid_num_workers)
  string +=   " datloader2_test_num_workers="  + str(arguments.datloader2_test_num_workers)
  string +=   " optimizer2_epochs="            + str(arguments.optimizer2_epochs)
  string +=   " optimizer2_record_last="       + str(arguments.optimizer2_record_last)
  string +=   " optimizer2_sgd_or_adamw="      + str(arguments.optimizer2_sgd_or_adamw)
  string +=   " optimizer2_lr="                + str(arguments.optimizer2_lr)
  string +=   " optimizer2_weight_decay="      + str(arguments.optimizer2_weight_decay)
  if arguments.optimizer2_sgd_or_adamw:
    string += " optimizer2_sgd_momentum="      + str(arguments.optimizer2_sgd_momentum)
    string += " optimizer2_sgd_dampening="     + str(arguments.optimizer2_sgd_dampening)
    string += " optimizer2_sgd_nesterov="      + str(arguments.optimizer2_sgd_nesterov)
  else:
    string += " optimizer2_adamw_betas="       + str(arguments.optimizer2_adamw_betas)
    string += " optimizer2_adamw_eps="         + str(arguments.optimizer2_adamw_eps)
    string += " optimizer2_adamw_amsgrad="     + str(arguments.optimizer2_adamw_amsgrad)
  string +=   " scheduler2_milestones="        + str(arguments.scheduler2_milestones)
  string +=   " scheduler2_gamma="             + str(arguments.scheduler2_gamma)
  string +=   " search_combs_or_leaves="       + str(arguments.search_combs_or_leaves)
  string +=   " hard_or_soft_search="          + str(arguments.hard_or_soft_search)
  string +=   " search_threshold="             + str(arguments.search_threshold)
  string +=   " reinitialization="             + str(arguments.reinitialization)
  string +=   " count_params_or_not="          + str(arguments.count_params_or_not)
  string +=   " count_flops_or_not="           + str(arguments.count_flops_or_not)
  string +=   " draw_heatmap_or_not="          + str(arguments.draw_heatmap_or_not)
  string +=   " draw_status_or_not="           + str(arguments.draw_status_or_not)
  string +=   " draw_format="                  + str(arguments.draw_format)
  string +=   " cuda="                         + str(arguments.cuda)
  string +=   " gpu="                          + str(arguments.gpu)
  string +=   " non_blocking="                 + str(arguments.non_blocking)
  return string

def leaves_to_string(leaves, sort = True):
  string = None
  if sort:
    leaves = sorted(list(leaves), key = lambda leaf: leaf.name)
  for leaf in leaves:
    if string is None:
      string  = str(leaf.name)
    else:
      string += " " + str(leaf.name)
  return string

def record_arguments(id, arguments, record_directory):
  arg_string = arguments_to_string(id, arguments)
  arg_path = os.path.join(record_directory, "arg.txt")
  with open(arg_path, "a", encoding = "utf-8") as f:
    f.write(arg_string + "\n")

def record_results(results, record_directory):
  if record_directory is None: return
  log_path = os.path.join(record_directory, "log.txt")
  if os.path.exists(log_path):
    # read
    with open(log_path, "r", encoding = "utf-8") as f:
      lines = f.readlines()
    id = len(lines) - 1
    references = lines.pop().split()
    references = [int(ref) for ref in references]
    references = [
      [references[           4  : references[0]],   int(lines[references[            4]].split()[1])],
      [references[references[0] : references[1]],   int(lines[references[references[0]]].split()[2])],
      [references[references[1] : references[2]], float(lines[references[references[1]]].split()[3])],
      [references[references[2] : references[3]], float(lines[references[references[2]]].split()[4])],
      [references[references[3] :              ], float(lines[references[references[3]]].split()[5])]
    ]
    # compare Params and nFLOPs
    for index in range(2):
      if references[index][1] < results[index]:
        pass
      elif results[index] < references[index][1]:
        references[index][0] = [len(lines)]
      else:
        references[index][0].append(len(lines))
    # compare accuracies and architecture probability
    for index in range(2, 5):
      if results[index] < references[index][1]:
        pass
      elif references[index][1] < results[index]:
        references[index][0] = [len(lines)]
      else:
        references[index][0].append(len(lines))
    split_end = 4
    last_string = ""
    for index in range(4):
      split_end += len(references[index][0])
      last_string += str(split_end) + " "
    for index in range(5):
      for value in references[index][0]:
        last_string += str(value) + " "
    lines.append(f"{id} {results[0]} {results[1]} {results[2]:>7f} {results[3]:>7f} {results[4]:>7f} {leaves_to_string(results[5])}\n")
    lines.append(last_string[:-1])
    # write
    with open(log_path, "w", encoding = "utf-8") as f:
      f.writelines(lines)
  else:
    id = 0
    lines = []
    lines.append(f"{id} {results[0]} {results[1]} {results[2]:>7f} {results[3]:>7f} {results[4]:>7f} {leaves_to_string(results[5])}\n")
    lines.append("5 6 7 8 0 0 0 0 0")
    with open(log_path, "w", encoding = "utf-8") as f:
      f.writelines(lines)
  return id