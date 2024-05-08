# Author: Jiahao Li

# https://discuss.pytorch.org/t/different-model-branches-for-different-samples-in-batch/151978/2
# https://github.com/pytorch/pytorch/issues/23653

import random
from functools import cmp_to_key

def leaves_sort_proxy(path0, path1):
  if path0[1] is None:
    if path1[1] is None:
      if path0[0] < path1[0]:
        return -1
      elif path1[0] < path0[0]:
        return 1
      else:
        return 0
    else:
      return -1
  elif path1[1] is None:
    return 1
  else:
    for desc0, desc1 in zip(path0[1], path1[1]):
      if desc0 < desc1:
        return -1
      elif desc1 < desc0:
        return 1
    if len(path0[1]) < len(path1[1]):
      return -1
    elif len(path1[1]) < len(path0[1]):
      return 1
    else:
      return 0

########################### define probability multiplier ##########################

def additive_multiplier(prob1, prob2):
  if prob1 is None:
    if prob2 is None:
      return None
    else:
      return prob2
  else:
    if prob2 is None:
      return prob1
    else:
      return prob1 + prob2

def original_multiplier(prob1, prob2):
  if prob1 is None:
    if prob2 is None:
      return None
    else:
      return prob2
  else:
    if prob2 is None:
      return prob1
    else:
      return prob1 * prob2

################################## define sampler ##################################

def distribution_sampler(route, logit, shift):
  if shift:
    idrng = range(len(logit))
    chose = random.choices(idrng, logit)[0]
    return chose, route[chose]
  else:
    return None, None

def maximization_sampler(route, logit, shift):
  if shift:
    idlst = [0]
    idlen = len(logit)
    for index in range(1, idlen):
      curlg = logit[index]
      maxlg = logit[idlst[0]]
      if maxlg < curlg:
        idlst.clear()
        idlst.append(index)
      elif not curlg < maxlg:
        idlst.append(index)
    chose = random.choice(idlst)
    return chose, route[chose]
  else:
    return None, None

################################### define hitter ##################################

class DigitHitter:
  def __init__(self):
    pass

  def __call__(self, clink, chose, shift):
    if shift:
      lkinf = clink[1]
      eport = lkinf[0]
      edesc = eport[1]
      return chose + 1 >> edesc & 1
    else:
      return True

  def __getitem__(self, param):
    clnks = param[0]
    shift = param[1]
    if shift:
      chose = 0
      for clink in clnks:
        lkinf = clink[1]
        eport = lkinf[0]
        edesc = eport[1]
        chose += 1 << edesc
      chose -= 1
    else:
      chose = None
    return chose

class EqualHitter:
  def __init__(self):
    pass

  def __call__(clink, chose, shift):
    if shift:
      lkinf = clink[1]
      eport = lkinf[0]
      edesc = eport[1]
      return chose == edesc
    else:
      return True

  def __getitem__(self, param):
    clnks = param[0]
    shift = param[1]
    clink = clnks[0]
    if shift:
      lkinf = clink[1]
      eport = lkinf[0]
      edesc = eport[1]
    else:
      edesc = None
    return edesc

########################### define GraphExtracter ###########################

class GraphExtracter:
  def __init__(self, graph, leafs, mulop, lsort = True, pbuff = True, fbuff = True, wbuff = True):
    self.__graph = graph
    self.__leafs = leafs
    self.__mulop = mulop
    self.__lsort = False
    self.__pbuff = pbuff
    self.__fbuff = fbuff
    self.__wbuff = wbuff
    self.__paths = {}
    self.__fmaps = {}
    self.__pways = {}
    if lsort:
      self.leaves_sort(True)

  def get_probabilty_multiplier(self):
    return self.__mulop

  def get_probabilty_multiplier_type(self):
    return self.__mulop is additive_multiplier

  def get_graph(self):
    return self.__graph

  def get_leaves(self):
    return self.__leafs

  def set_graph(self, graph):
    self.__graph = graph

  def set_leaves(self, leafs):
    self.__leafs = leafs

  def path_buffer_on(self):
    self.__pbuff = True

  def path_buffer_off(self):
    self.__pbuff = False

  def fmap_buffer_on(self):
    self.__fbuff = True

  def fmap_buffer_off(self):
    self.__fbuff = False

  def pway_buffer_on(self):
    self.__wbuff = True

  def pway_buffer_off(self):
    self.__wbuff = False

  def clear_path_buffer(self):
    self.__pbuff.clear()

  def clear_fmap_buffer(self):
    self.__fbuff.clear()

  def clear_pway_buffer(self):
    self.__wbuff.clear()

  def clear_all_buffers(self):
    self.__pbuff.clear()
    self.__fbuff.clear()
    self.__wbuff.clear()

  def transfer_block_indexes_to_nodes(self, blktp):
    return [self.__graph[blkid][1] for blkid in blktp]

  def transfer_block_id_combinations_to_node_combinations(self, blkts):
    return [
      [self.__graph[blkid][1] for blkid in blktp]
      for blktp in blkts
    ]

  def transfer_nodes_to_block_indexes(self, nodes):
    table = {}
    for blkid in self.__leafs:
      node = self.__graph[blkid][1]
      table[node] = blkid
    return [table[node] for node in nodes]

  def transfer_node_combinations_to_block_id_combinations(self, combs):
    table = {}
    for blkid in self.__leafs:
      node = self.__graph[blkid][1]
      table[node] = blkid
    return [
      [table[node] for node in nodes]
      for nodes in combs
    ]

  def create_path(self, blkid):
    blkls = []
    while blkid is not None:
      blkls.append(blkid)
      blkid = self.__graph[blkid][0]
    return blkls

  def create_paths(self, blktp):
    paths = []
    for blkid in blktp:
      blkls = self.create_path(blkid)
      paths.append(blkls)
    return paths

  def buffer_path(self, blkid):
    try:
      blkls = self.__paths[blkid]
    except KeyError:
      blkls = self.create_path(blkid)
      if blkls:
        self.__paths[blkid] = blkls
    return blkls

  def buffer_paths(self, blktp):
    paths = []
    for blkid in blktp:
      blkls = self.buffer_path(blkid)
      paths.append(blkls)
    return paths

  def get_path(self, blkid, pbuff = None):
    if pbuff is None:
      pbuff = self.__pbuff
    if pbuff:
      blkls = self.buffer_path(blkid)
    else:
      blkls = self.create_path(blkid)
    return blkls

  def get_paths(self, blktp, pbuff = None):
    if pbuff is None:
      pbuff = self.__pbuff
    if pbuff:
      paths = self.buffer_paths(blktp)
    else:
      paths = self.create_paths(blktp)
    return paths

  def get_path_vnum(self, blkid, pbuff = None):
    # get number of vertexs
    blkls = self.get_path(blkid, pbuff)
    return len(blkls)

  def get_paths_vnum(self, blktp, pbuff = None):
    # get number of vertexs
    paths = self.get_paths(blktp, pbuff)
    return [len(blkls) for blkls in paths]

  def get_path_enum(self, blkid, pbuff = None):
    # get number of edges
    blkls = self.get_path(blkid, pbuff)
    return len(blkls) - 1

  def get_paths_enum(self, blktp, pbuff = None):
    # get number of edges
    paths = self.get_paths(blktp, pbuff)
    return [len(blkls) - 1 for blkls in paths]

  def create_fmap(self, blktp, pbuff = None):
    paths = self.get_paths(blktp, pbuff)
    forks = {}
    for blkls in paths:
      blkid = blkls[-1]
      try:
        forks[blkid].append(blkls)
      except KeyError:
        forks[blkid] = [blkls]
    graph = []
    leafs = []
    stack = []
    for blkid, paths in forks.items():
      paths = [p for p in paths if 1 < len(p)]
      if paths:
        gradr = len(graph)
        graph.append(None)
        stack.append([paths, blkid, gradr, 2])
      else:
        leafs.append([None, blkid])
    while stack:
      entry = stack.pop()
      paths = entry[0]
      blkid = entry[1]
      gradr = entry[2]
      depth = entry[3]
      forks = {}
      for blkls in paths:
        kidid = blkls[-depth]
        try:
          forks[kidid].append(blkls)
        except KeyError:
          forks[kidid] = [blkls]
      block = self.__graph[blkid]
      # pnode = block[1]
      cdict = block[2]
      # input = block[3]
      route = block[4]
      # logit = block[5]
      shift = block[6]
      # smplr = block[7]
      hiter = block[8]
      kdict = {}
      clnks = []
      for kidid, paths in forks.items():
        lkinf = cdict[kidid]
        clnks.append((kidid, lkinf))
        paths = [p for p in paths if depth < len(p)]
        if paths:
          grlen = len(graph)
          kdict[grlen] = lkinf
          graph.append(gradr)
          stack.append([paths, kidid, grlen, depth + 1])
        else:
          leafs.append([gradr, kidid, kdict, lkinf])
      chose = hiter[clnks, shift]
      if chose is None or 0 <= chose < route.shape[1]:
        graph[gradr] = [graph[gradr], blkid, kdict, chose]
      else:
        return None
    for block in leafs:
      if len(block) == 4:
        kdict = block[2]
        lkinf = block[3]
        grlen = len(graph)
        kdict[grlen] = lkinf
        del block[2:]
      graph.append(block)
    return graph

  def create_fmaps(self, blkts, pbuff = None):
    fmaps = []
    for blktp in blkts:
      graph = self.create_fmap(blktp, pbuff)
      fmaps.append(graph)
    return fmaps

  def buffer_fmap(self, blktp, pbuff = None):
    try:
      graph = self.__fmaps[blktp]
    except KeyError:
      graph = self.create_fmap(blktp, pbuff)
      if graph:
        self.__fmaps[blktp] = graph
    return graph

  def buffer_fmaps(self, blkts, pbuff = None):
    fmaps = []
    for blktp in blkts:
      graph = self.buffer_fmap(blktp, pbuff)
      fmaps.append(graph)
    return fmaps

  def get_fmap(self, blktp, pbuff = None, fbuff = None):
    if fbuff is None:
      fbuff = self.__fbuff
    if fbuff:
      graph = self.buffer_fmap(blktp, pbuff)
    else:
      graph = self.create_fmap(blktp, pbuff)
    return graph

  def get_fmaps(self, blkts, pbuff = None, fbuff = None):
    if fbuff is None:
      fbuff = self.__fbuff
    if fbuff:
      fmaps = self.buffer_fmaps(blkts, pbuff)
    else:
      fmaps = self.create_fmaps(blkts, pbuff)
    return fmaps

  def get_fmap_vnum(self, blktp, pbuff = None, fbuff = None):
    # get number of vertexs
    graph = self.get_fmap(blktp, pbuff, fbuff)
    return len(graph)

  def get_fmaps_vnum(self, blkts, pbuff = None, fbuff = None):
    # get number of vertexs
    fmaps = self.get_fmaps(blkts, pbuff, fbuff)
    return [len(graph) for graph in fmaps]

  def get_fmap_enum(self, blktp, pbuff = None, fbuff = None):
    # get number of edges
    enumb = 0
    graph = self.get_fmap(blktp, pbuff, fbuff)
    for block in graph:
      if len(block) == 4:
        enumb += len(block[2])
    return enumb

  def get_fmaps_enum(self, blkts, pbuff = None, fbuff = None):
    # get number of edges
    enums = []
    fmaps = self.get_fmaps(blkts, pbuff, fbuff)
    for graph in fmaps:
      enumb = 0
      for block in graph:
        if len(block) == 4:
          enumb += len(block[2])
      enums.append(enumb)
    return enums

  def create_pway(self, blkid, pbuff = None):
    blkls = self.get_path(blkid, pbuff)
    lstid = len(blkls) - 1
    proad = []
    while lstid:
      curid = blkls[lstid]
      lstid = lstid - 1
      sucid = blkls[lstid]
      block = self.__graph[curid]
      # pnode = block[1]
      cdict = block[2]
      # input = block[3]
      route = block[4]
      # logit = block[5]
      shift = block[6]
      # smplr = block[7]
      hiter = block[8]
      lkinf = cdict[sucid]
      clink = (sucid, lkinf)
      picks = []
      for chose in range(route.shape[1]):
        if hiter(clink, chose, shift):
          picks.append(chose)
      if picks:
        proad.append([curid, lkinf, picks])
      else:
        return None
    proad.append([blkid])
    return proad

  def create_pways(self, blktp, pbuff = None):
    pways = []
    for blkid in blktp:
      proad = self.create_pway(blkid, pbuff)
      pways.append(proad)
    return pways

  def buffer_pway(self, blkid, pbuff = None):
    try:
      proad = self.__pways[blkid]
    except KeyError:
      proad = self.create_pway(blkid, pbuff)
      if proad:
        self.__pways[blkid] = proad
    return proad

  def buffer_pways(self, blktp, pbuff = None):
    pways = []
    for blkid in blktp:
      proad = self.buffer_pway(blkid, pbuff)
      pways.append(proad)
    return pways

  def get_pway(self, blkid, pbuff = None, wbuff = None):
    if wbuff is None:
      wbuff = self.__wbuff
    if wbuff:
      proad = self.buffer_pway(blkid, pbuff)
    else:
      proad = self.create_pway(blkid, pbuff)
    return proad

  def get_pways(self, blktp, pbuff = None, wbuff = None):
    if wbuff is None:
      wbuff = self.__wbuff
    if wbuff:
      pways = self.buffer_pways(blktp, pbuff)
    else:
      pways = self.create_pways(blktp, pbuff)
    return pways

  def get_pway_vnum(self, blktp, pbuff = None, fbuff = None):
    # get number of vertexs
    proad = self.get_pway(blktp, pbuff, fbuff)
    return len(proad)

  def get_pways_vnum(self, blkts, pbuff = None, fbuff = None):
    # get number of vertexs
    pways = self.get_pways(blkts, pbuff, fbuff)
    return [len(proad) for proad in pways]

  def get_pway_enum(self, blktp, pbuff = None, fbuff = None):
    # get number of edges
    proad = self.get_pway(blktp, pbuff, fbuff)
    return len(proad) - 1

  def get_pways_enum(self, blkts, pbuff = None, fbuff = None):
    # get number of edges
    pways = self.get_pways(blkts, pbuff, fbuff)
    return [len(proad) - 1 for proad in pways]

  def verify_combination(self, blktp, pbuff = None, fbuff = None):
    if self.get_fmap(blktp, pbuff, fbuff):
      return True
    else:
      return False

  def get_prob_of_combination(self, blktp, pbuff = None, fbuff = None):
    graph = self.get_fmap(blktp, pbuff, fbuff)
    if graph:
      prmul = None
      for block in graph:
        if len(block) == 2:
          break
        chose = block[3]
        if chose is not None:
          blkid = block[1]
          route = self.__graph[blkid][4]
          prmul = self.__mulop(prmul, route[:, chose])
      return prmul
    else:
      return False

  def get_graph_of_combination(self, blktp, pbuff = None, fbuff = None):
    graph = self.get_fmap(blktp, pbuff, fbuff)
    if graph:
      grbuf = []
      for block in graph:
        gradr = block[0]
        blkid = block[1]
        if len(block) == 2:
          block = self.__graph[blkid]
          grbuf.append([gradr, block[1], None] + block[3:])
        else:
          kdict = block[2]
          block = self.__graph[blkid]
          grbuf.append([gradr, block[1], kdict] + block[3:])
      return grbuf
    else:
      return None

  def get_prob_and_graph_of_combination(self, blktp, pbuff = None, fbuff = None):
    graph = self.get_fmap(blktp, pbuff, fbuff)
    if graph:
      grbuf = []
      prmul = None
      for block in graph:
        gradr = block[0]
        blkid = block[1]
        if len(block) == 2:
          block = self.__graph[blkid]
          grbuf.append([gradr, block[1], None] + block[3:])
        else:
          kdict = block[2]
          chose = block[3]
          block = self.__graph[blkid]
          route = block[4]
          prmul = self.__mulop(prmul, route[:, chose])
          grbuf.append([gradr, block[1], kdict] + block[3:])
      return prmul, grbuf
    else:
      return None

  def verify_child(self, blkid, pbuff = None, wbuff = None):
    if self.get_pway(blkid, pbuff, wbuff):
      return True
    else:
      return False

  def get_prob_of_child(self, blkid, pbuff = None, wbuff = None):
    proad = self.get_pway(blkid, pbuff, wbuff)
    if proad:
      prmul = None
      if self.__mulop is additive_multiplier:
        for block in proad:
          if len(block) == 1:
            break
          blkid = block[0]
          picks = block[2]
          route = self.__graph[blkid][4]
          if len(picks) == 1:
            prsum = route[:, chose]
          else:
            prsum = None
            for chose in picks:
              if prsum is None:
                prsum = route[:, chose].exp()
              else:
                prsum = prsum + route[:, chose].exp()
            prsum = prsum.log()
          prmul = self.__mulop(prmul, prsum)
      else:
        for block in proad:
          if len(block) == 1:
            break
          blkid = block[0]
          picks = block[2]
          route = self.__graph[blkid][4]
          prsum = None
          for chose in picks:
            if prsum is None:
              prsum = route[:, chose]
            else:
              prsum = prsum + route[:, chose]
          prmul = self.__mulop(prmul, prsum)
      return prmul
    else:
      return False

  def get_path_of_child(self, blkid, pbuff = None, wbuff = None):
    proad = self.get_pway(blkid, pbuff, wbuff)
    if proad:
      graph = []
      gradr = None
      for block in proad:
        blkid = block[0]
        if len(block) == 1:
          block = self.__graph[blkid]
          graph.append([gradr, block[1], None] + block[3:])
        else:
          lkinf = block[1]
          block = self.__graph[blkid]
          grlen = len(graph)
          graph.append([gradr, block[1], {grlen + 1 : lkinf}] + block[3:])
          gradr = grlen
      return graph
    else:
      return None

  def get_prob_and_path_of_child(self, blkid, pbuff = None, wbuff = None):
    proad = self.get_pway(blkid, pbuff, wbuff)
    if proad:
      prmul = None
      gradr = None
      graph = []
      if self.__mulop is additive_multiplier:
        for block in proad:
          blkid = block[0]
          if len(block) == 1:
            block = self.__graph[blkid]
            graph.append([gradr, block[1], None] + block[3:])
          else:
            lkinf = block[1]
            picks = block[2]
            block = self.__graph[blkid]
            route = block[4]
            if len(picks) == 1:
              prsum = route[:, chose]
            else:
              prsum = None
              for chose in picks:
                if prsum is None:
                  prsum = route[:, chose].exp()
                else:
                  prsum = prsum + route[:, chose].exp()
              prsum = prsum.log()
            prmul = self.__mulop(prmul, prsum)
            grlen = len(graph)
            graph.append([gradr, block[1], {grlen + 1 : lkinf}] + block[3:])
            gradr = grlen
      else:
        for block in proad:
          blkid = block[0]
          if len(block) == 1:
            block = self.__graph[blkid]
            graph.append([gradr, block[1], None] + block[3:])
          else:
            lkinf = block[1]
            picks = block[2]
            block = self.__graph[blkid]
            route = block[4]
            prsum = None
            for chose in picks:
              if prsum is None:
                prsum = route[:, chose]
              else:
                prsum = prsum + route[:, chose]
            prmul = self.__mulop(prmul, prsum)
            grlen = len(graph)
            graph.append([gradr, block[1], {grlen + 1 : lkinf}] + block[3:])
            gradr = grlen
      return prmul, graph
    else:
      return None

  def leaves_sort(self, store = False, pbuff = None):
    if self.__lsort:
      return self.__leafs
    else:
      paths = self.get_paths(self.__leafs, pbuff)
      for index, blkls in enumerate(paths):
        lstid = len(blkls) - 1
        descs = []
        while lstid:
          curid = blkls[lstid]
          lstid = lstid - 1
          sucid = blkls[lstid]
          block = self.__graph[curid]
          clink = block[2]
          lkinf = clink[sucid]
          eport = lkinf[0]
          edesc = eport[1]
          descs.append(edesc)
        if descs:
          paths[index] = (sucid, descs)
        else:
          paths[index] = (blkls[0], None)
      paths = sorted(paths, key = cmp_to_key(lambda x, y: leaves_sort_proxy(x, y)))
      leafs = [entry[0] for entry in paths]
      if store:
        self.__leafs = leafs
        self.__lsort = True
      return leafs

  def __call__(self, blktp, prext = True, grext = False, pbuff = None, fbuff = None):
    if prext:
      if grext:
        return self.get_prob_and_graph_of_combination(blktp, pbuff, fbuff)
      else:
        return self.get_prob_of_combination(blktp, pbuff, fbuff)
    else:
      if grext:
        return self.get_graph_of_combination(blktp, pbuff, fbuff)
      else:
        return self.verify_combination(blktp, pbuff, fbuff)

  def __getitem__(self, param):
    try:
      blkid, prext, grext, pbuff, wbuff = param
    except TypeError:
      try:
        blkid, prext, grext, pbuff = param
      except TypeError:
        try:
          blkid, prext, grext = param
        except TypeError:
          try:
            blkid, prext = param
          except TypeError:
            blkid = param
            prext = True
          grext = False
        pbuff = None
      wbuff = None
    if prext:
      if grext:
        return self.get_prob_and_path_of_child(blkid, pbuff, wbuff)
      else:
        return self.get_prob_of_child(blkid, pbuff, wbuff)
    else:
      if grext:
        return self.get_path_of_child(blkid, pbuff, wbuff)
      else:
        return self.verify_child(blkid, pbuff, wbuff)

################################## define diverter #################################

class Diverter:
  def __init__(self, forest, config, mulfun = True):
    self.__forest = forest
    self.__config = config
    self.__gextra = None
    digit_hitter = DigitHitter()
    equal_hitter = EqualHitter()
    for oplib in config:
      try:
        smplr = oplib["sampler"]
        if smplr is True:
          oplib["sampler"] = distribution_sampler
        elif smplr is False:
          oplib["sampler"] = maximization_sampler
      except KeyError:
        oplib["sampler"] = distribution_sampler
      try:
        hiter = oplib["hitter"]
        if hiter is True:
          oplib["hitter"] = digit_hitter
        elif hiter is False:
          oplib["hitter"] = equal_hitter
      except KeyError:
        oplib["hitter"] = digit_hitter
    if mulfun:
      self.__mulfun = additive_multiplier
    else:
      self.__mulfun = original_multiplier

  def get_probabilty_multiplier(self):
    return self.__mulfun

  def get_probabilty_multiplier_type(self):
    return self.__mulfun is additive_multiplier

  def handle_single_instance_with_one_policy(self, input, *_):
    prmap = None
    edarr = [    ]
    graph = [None]
    stack = [[0, 0, input, None]]
    mulop = self.__mulfun
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      prmul = entry[3]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      smplr = oplib["sampler"]
      hiter = oplib["hitter"]
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        route = oplib["prev_router"](input)
        shift = True
      except KeyError:
        route = input
        shift = False
      try:
        logit = oplib["logit"](route)
        shift = True
      except KeyError:
        logit = route
        shift = shift | False
      try:
        route = oplib["post_router"](route)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      # sampling
      smprs = smplr(route, logit, shift)
      chose, propt = smprs
      prmap = mulop(prmap, propt)
      # routing
      sdict = {}
      if cdict:
        for clink in cdict.items():
          if hiter(clink, chose, shift):
            blkid, lkinf = clink
            grlen = len(graph)
            graph.append(gradr)
            sdict[grlen] = lkinf
            prout = mulop(prmul, propt)
            stack.append([blkid, grlen, input, prout])
      # update graph
      if sdict:
        graph[gradr] = [graph[gradr], pnode, sdict, chose, propt, prmul]
      else:
        try:
          input = oplib["solver"](input)
        except KeyError:
          pass
        edarr.append(gradr)
        graph[gradr] = [graph[gradr], pnode, input, chose, propt, prmul]
    return graph, edarr, prmap

  def handle_multiple_instances_with_one_policy(self, input, *_):
    # grbuf(global) : max graph instances can reach
    # edbuf(global) : ending outputs of max graph
    # grlst(global) : graphs of n instances
    # edlst(global) : ending outputs of n graphs
    # gplst(global) : probabilities of n graphs
    # snlst(local)  : series numbers of n samples
    # adlst(local)  : addresses of n blocks
    # prlst(local)  : probabilities of n instances toward node
    grbuf = [None]
    edbuf = [    ]
    snlst = range(len(input))
    grlst = [[None] for _ in snlst]
    edlst = [[    ] for _ in snlst]
    gplst = [ None  for _ in snlst]
    adlst = [  0    for _ in snlst]
    prlst = [ None  for _ in snlst]
    stack = [[0, 0, input, snlst, adlst, prlst]]
    mulop = self.__mulfun
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      snlst = entry[3]
      adlst = entry[4]
      prlst = entry[5]
      inrng = range(len(input))
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      smplr = oplib["sampler"]
      hiter = oplib["hitter"]
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        route = oplib["prev_router"](input)
        shift = True
      except KeyError:
        route = input
        shift = False
      try:
        logit = oplib["logit"](route)
        shift = True
      except KeyError:
        logit = route
        shift = shift | False
      try:
        route = oplib["post_router"](route)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      # initialization and sampling
      picks, probs = [], []
      pracs, dicts = [], []
      for curid in inrng:
        lgdat = logit[curid]
        rtdat = route[curid]
        smprs = smplr(rtdat, lgdat, shift)
        chose, propt = smprs
        prmul = prlst[curid]
        prmul = mulop(prmul, propt)
        picks.append(chose)
        probs.append(propt)
        pracs.append(prmul)
        dicts.append({})
      # routing
      kdict = {}
      if cdict:
        for clink in cdict.items():
          fragm = False
          blkid, lkinf = clink
          idarr, snarr = [], []
          adarr, prarr = [], []
          for curid in inrng:
            chose = picks[curid]
            if hiter(clink, chose, shift):
              prmul = pracs[curid]
              sdict = dicts[curid]
              cursn = snlst[curid]
              curad = adlst[curid]
              graph = grlst[cursn]
              grlen = len(graph)
              idarr.append(curid)
              snarr.append(cursn)
              adarr.append(grlen)
              prarr.append(prmul)
              graph.append(curad)
              sdict[grlen] = lkinf
            else:
              fragm = True
          if idarr:
            indat = input[idarr] if fragm else input
            grlen = len(grbuf)
            grbuf.append(gradr)
            kdict[grlen] = lkinf
            stack.append([blkid, grlen, indat, snarr, adarr, prarr])
      # update graph for each instance and collect instances without being routed
      idarr = []
      fragm = False
      for curid in inrng:
        sdict = dicts[curid]
        if sdict:
          fragm = True
          chose = picks[curid]
          propt = probs[curid]
          prmul = prlst[curid]
          cursn = snlst[curid]
          curad = adlst[curid]
          graph = grlst[cursn]
          gplst[cursn] = mulop(gplst[cursn], propt)
          graph[curad] = [graph[curad], pnode, sdict, chose, propt, prmul, gradr]
        else:
          idarr.append(curid)
      # handle instances without being routed
      if idarr:
        if fragm:
          input = input[idarr]
        try:
          input = oplib["solver"](input)
        except AttributeError:
          pass
      for curid in idarr:
        chose = picks[curid]
        propt = probs[curid]
        prmul = prlst[curid]
        edout = input[curid]
        cursn = snlst[curid]
        curad = adlst[curid]
        graph = grlst[cursn]
        edlst[cursn].append(curad)
        gplst[cursn] = mulop(gplst[cursn], propt)
        graph[curad] = [graph[curad], pnode, edout, chose, propt, prmul, gradr]
      # update max graph samples can reach
      if kdict:
        grbuf[gradr] = [grbuf[gradr], pnode, kdict, snlst, adlst]
      else:
        edbuf.append(gradr)
        grbuf[gradr] = [grbuf[gradr], pnode, None, snlst, adlst]
    return grbuf, edbuf, grlst, edlst, gplst

  def handle_multiple_instances_with_all_policies(self, input, *_):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    leafs = [    ]
    graph = [None]
    stack = [[0, 0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      smplr = oplib["sampler"]
      hiter = oplib["hitter"]
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        route = oplib["prev_router"](input)
        shift = True
      except KeyError:
        route = input
        shift = False
      try:
        logit = oplib["logit"](route)
        shift = True
      except KeyError:
        logit = route
        shift = shift | False
      try:
        route = oplib["post_router"](route)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        kdict = {}
        for blkid, lkinf in cdict.items():
          grlen = len(graph)
          graph.append(gradr)
          kdict[grlen] = lkinf
          stack.append([blkid, grlen, input])
        graph[gradr] = [graph[gradr], pnode, kdict, input, route, logit, shift, smplr, hiter]
      else:
        leafs.append(gradr)
        graph[gradr] = [graph[gradr], pnode, None, input, route, logit, shift, smplr, hiter]
    if self.__gextra is None:
      self.__gextra = GraphExtracter(graph, leafs, self.__mulfun)
    else:
      self.__gextra.set_graph(graph)
    return self.__gextra

  def handle_multiple_instances_without_policies(self, input, *_):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    leafs = [    ]
    graph = [None]
    stack = [[0, 0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        kdict = {}
        for blkid, lkinf in cdict.items():
          grlen = len(graph)
          graph.append(gradr)
          kdict[grlen] = lkinf
          stack.append([blkid, grlen, input])
        graph[gradr] = [graph[gradr], pnode, kdict, input]
      else:
        leafs.append(gradr)
        graph[gradr] = [graph[gradr], pnode, None, input]
    return graph, leafs

  def handle_multiple_instances_without_graph(self, input, *_):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    leafs = [    ]
    stack = [[0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      input = entry[1]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        for blkid, _ in cdict.items():
          stack.append([blkid, input])
      else:
        leafs.append((pnode, input))
    leafs = sorted(leafs, key = lambda entry: id(entry[0]))
    leafs = [entry[1] for entry in leafs]
    return leafs

  def handle_multiple_instances_with_all_policies_given_leaf_nodes(self, input, *args):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    nodes = args[0]
    leafs = [    ]
    graph = [None]
    stack = [[0, 0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      smplr = oplib["sampler"]
      hiter = oplib["hitter"]
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        route = oplib["prev_router"](input)
        shift = True
      except KeyError:
        route = input
        shift = False
      try:
        logit = oplib["logit"](route)
        shift = True
      except KeyError:
        logit = route
        shift = shift | False
      try:
        route = oplib["post_router"](route)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        kdict = {}
        for blkid, lkinf in cdict.items():
          if self.__forest[blkid][0] in nodes \
          or self.__forest[blkid][0].check_targets_reachablity(nodes):
            grlen = len(graph)
            graph.append(gradr)
            kdict[grlen] = lkinf
            stack.append([blkid, grlen, input])
        if kdict:
          graph[gradr] = [graph[gradr], pnode, kdict, input, route, logit, shift, smplr, hiter]
        else:
          leafs.append(gradr)
          graph[gradr] = [graph[gradr], pnode, None, input, route, logit, shift, smplr, hiter]
      else:
        leafs.append(gradr)
        graph[gradr] = [graph[gradr], pnode, None, input, route, logit, shift, smplr, hiter]
    if self.__gextra is None:
      self.__gextra = GraphExtracter(graph, leafs, self.__mulfun)
    else:
      self.__gextra.set_graph(graph)
    return self.__gextra

  def handle_multiple_instances_without_policies_given_leaf_nodes(self, input, *args):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    nodes = args[0]
    leafs = [    ]
    graph = [None]
    stack = [[0, 0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      gradr = entry[1]
      input = entry[2]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        kdict = {}
        for blkid, lkinf in cdict.items():
          if self.__forest[blkid][0] in nodes \
          or self.__forest[blkid][0].check_targets_reachablity(nodes):
            grlen = len(graph)
            graph.append(gradr)
            kdict[grlen] = lkinf
            stack.append([blkid, grlen, input])
        if kdict:
          graph[gradr] = [graph[gradr], pnode, kdict, input]
        else:
          leafs.append(gradr)
          graph[gradr] = [graph[gradr], pnode, None, input]
      else:
        leafs.append(gradr)
        graph[gradr] = [graph[gradr], pnode, None, input]
    return graph, leafs

  def handle_multiple_instances_without_graph_given_leaf_nodes(self, input, *args):
    # leafs(global) : leaves shared by all instances
    # graph(global) : graph  shared by all instances
    nodes = args[0]
    leafs = [    ]
    stack = [[0, input]]
    while stack:
      entry = stack.pop()
      blkid = entry[0]
      input = entry[1]
      block = self.__forest[blkid]
      oplib = self.__config[blkid]
      pnode, cdict = block
      try:
        input = oplib["prev_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["post_transform"](input)
      except KeyError:
        pass
      try:
        input = oplib["solver"](input)
      except KeyError:
        pass
      if cdict:
        kdict = {}
        for blkid, _ in cdict.items():
          if self.__forest[blkid][0] in nodes \
          or self.__forest[blkid][0].check_targets_reachablity(nodes):
            stack.append([blkid, input])
        if not kdict:
          leafs.append((pnode, input))
      else:
        leafs.append((pnode, input))
    leafs = sorted(leafs, key = lambda entry: id(entry[0]))
    leafs = [entry[1] for entry in leafs]
    return leafs