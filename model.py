# Author: Jiahao Li

# third-parties
import torch
from diverter import Diverter

################################## reset_parameters_recursively #################################

def reset_parameters_recursively(model):
  for child in model.children():
    try:
      child.reset_parameters()
    except AttributeError:
      pass
    reset_parameters_recursively(child)

################################## define Component #################################

class Component(torch.nn.Module):
  def __init__(self, oplist):
    super().__init__()
    self.__oplist = oplist
    self.__nnmods = torch.nn.ModuleList()
    for module in oplist:
      try:
        self.__nnmods.append(module)
      except:
        pass

  def forward(self, input):
    for opitem in self.__oplist:
      input = opitem(input)
    return input

################################## define Model #################################

class Model(torch.nn.Module):
  def __init__(self, source, targets, handle = 2, mulfun = True):
    super().__init__()
    self.__forest = source.get_forest_toward_multiple_targets(targets, True)
    self.__forest[0] = [source, self.__forest[0]]
    print(self.__forest)
    # register all modules
    self.__config = []
    self.__nnmods = torch.nn.ModuleList()
    for block in self.__forest:
      content = block[0].content
      oplib = {}
      for name, details in content.items():
        if not block[1] or len(block[1]) < 2:
          if name == "prev_router" \
          or name == "logit" \
          or name == "prev_router":
            continue
        if isinstance(details, (list, tuple)):
          if 1 < len(details) and details[0]:
            details = details[1:]
            if len(details) < 2:
              module = details[0]
              try:
                self.__nnmods.append(module)
              except:
                pass
            else:
              module = Component(details)
              self.__nnmods.append(module)
            oplib[name] = module
        else:
          try:
            self.__nnmods.append(details)
          except:
            pass
          oplib[name] = details
      self.__config.append(oplib)
    self.__diverter = Diverter(self.__forest, self.__config, mulfun)
    self.set_handle_mode(handle)

  def reset_parameters(self):
    reset_parameters_recursively(self)

  def get_probabilty_multiplier(self):
    return self.__diverter.get_probabilty_multiplier()

  def get_probabilty_multiplier_type(self):
    return self.__diverter.get_probabilty_multiplier_type()

  def get_handle_mode(self):
    if self.__handler == self.__diverter.handle_single_instance_with_one_policy:
      return 0
    elif self.__handler == self.__diverter.handle_multiple_instances_with_one_policy:
      return 1
    elif self.__handler == self.__diverter.handle_multiple_instances_with_all_policies:
      return 2
    elif self.__handler == self.__diverter.handle_multiple_instances_without_policies:
      return 3
    elif self.__handler == self.__diverter.handle_multiple_instances_without_graph:
      return 4
    elif self.__handler == self.__diverter.handle_multiple_instances_with_all_policies_given_leaf_nodes:
      return 5
    elif self.__handler == self.__diverter.handle_multiple_instances_without_policies_given_leaf_nodes:
      return 6
    elif self.__handler == self.__diverter.handle_multiple_instances_without_graph_given_leaf_nodes:
      return 7
    else:
      raise ValueError("Unavailable mode!")

  def set_handle_mode(self, handle):
    if handle == 0:
      self.__handler = self.__diverter.handle_single_instance_with_one_policy
    elif handle == 1:
      self.__handler = self.__diverter.handle_multiple_instances_with_one_policy
    elif handle == 2:
      self.__handler = self.__diverter.handle_multiple_instances_with_all_policies
    elif handle == 3:
      self.__handler = self.__diverter.handle_multiple_instances_without_policies
    elif handle == 4:
      self.__handler = self.__diverter.handle_multiple_instances_without_graph
    elif handle == 5:
      self.__handler = self.__diverter.handle_multiple_instances_with_all_policies_given_leaf_nodes
    elif handle == 6:
      self.__handler = self.__diverter.handle_multiple_instances_without_policies_given_leaf_nodes
    elif handle == 7:
      self.__handler = self.__diverter.handle_multiple_instances_without_graph_given_leaf_nodes
    else:
      raise ValueError("Unavailable mode!")

  def set_mode_as_single_instance_with_one_policy(self):
    self.__handler = self.__diverter.handle_single_instance_with_one_policy

  def set_mode_as_multiple_instances_with_one_policy(self):
    self.__handler = self.__diverter.handle_multiple_instances_with_one_policy

  def set_mode_as_multiple_instances_with_all_policies(self):
    self.__handler = self.__diverter.handle_multiple_instances_with_all_policies

  def set_mode_as_multiple_instances_without_policies(self):
    self.__handler = self.__diverter.handle_multiple_instances_without_policies

  def set_mode_as_multiple_instances_without_graph(self):
    self.__handler = self.__diverter.handle_multiple_instances_without_graph

  def set_mode_as_multiple_instances_with_all_policies_given_leaf_nodes(self):
    self.__handler = self.__diverter.handle_multiple_instances_with_all_policies_given_leaf_nodes

  def set_mode_as_multiple_instances_without_policies_given_leaf_nodes(self):
    self.__handler = self.__diverter.handle_multiple_instances_without_policies_given_leaf_nodes

  def set_mode_as_multiple_instances_without_graph_given_leaf_nodes(self):
    self.__handler = self.__diverter.handle_multiple_instances_without_graph_given_leaf_nodes

  def forward(self, input, *args):
    return self.__handler(input, *args)
