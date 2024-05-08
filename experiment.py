# Author: Jiahao Li
import sys, os

import torch.nn.functional as tnf
from torch.multiprocessing import Manager
from fvcore.nn import FlopCountAnalysis
from prediction import *

################################# get_echo_indexes ##################################

def get_echo_indexes(num_batch, echo_freq, head_first = True):
  if echo_freq == 0:
    return []
  elif echo_freq == 1:
    return [0] if head_first else [num_batch - 1]
  elif echo_freq == 2:
    return [0, num_batch - 1]
  elif num_batch <= echo_freq:
    return range(num_batch)
  else:
    idlst = [0]
    intvl = (num_batch - 2) // (echo_freq - 1)
    for _ in range(echo_freq - 2):
      idlst.append(intvl)
    remsz = (num_batch - 2) % (echo_freq - 1)
    if remsz:
      intvl = (echo_freq - 2) // remsz
      index = intvl
      for _ in range(remsz):
        idlst[index] += 1
        index += intvl
    idsum = idlst[1]
    for index in range(2, echo_freq - 1):
      idlst[index] += idsum
      idsum = idlst[index]
    idlst.append(num_batch - 1)
    return idlst

########################### global_ensemble_train / eval ###########################

def global_ensemble_train_on_nll(
  model, loader, optim, device,
  non_blocking = False,
  likelihood_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  for batid, (input, label) in enumerate(loader):
    input = input.to(device, non_blocking = non_blocking)
    label = label.to(device, non_blocking = non_blocking)
    eprob = prediction_on_global_ensemble_label(model, input)
    nllpr = tnf.nll_loss(eprob, label)
    error = likelihood_weight * nllpr
    optim.zero_grad()
    error.requires_grad_(True)
    error.backward()
    optim.step()
    watch += len(input)
    if only_record_last:
      errls = (nllpr, error)
    else:
      errls.append((nllpr, error))
    if echo_indexes is None or batid == echo_indexes[echos]:
      echos += 1
      print(f"NLL: {nllpr:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

def global_ensemble_eval_on_nll(
  model, loader, device,
  non_blocking = False,
  likelihood_weight = 1.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      eprob = prediction_on_global_ensemble_label(model, input)
      nllac += tnf.nll_loss(eprob, label, reduction = "none")
      right += (eprob.argmax(1) == label).type(torch.float).sum()
    ldrsz = len(loader)
    nllac = nllac / ldrsz
    error = likelihood_weight * nllac
    datsz = len(loader.sampler)
    right = right / datsz
    print(f"NLL: {nllac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, error, right

########################### partial_ensemble_train / eval ###########################

def partial_ensemble_train_on_nll(
  model, loader, combs, optim, device,
  non_blocking = False,
  combination_weights = None,
  likelihood_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  if combination_weights is None:
    combination_weights = [1 / len(combs)] * len(combs)
  for batid, (input, label) in enumerate(loader):
    input = input.to(device, non_blocking = non_blocking)
    label = label.to(device, non_blocking = non_blocking)
    rslst = prediction_on_partial_ensemble_label(model, input, combs)
    eprob = combination_weights[0] * rslst[0]
    for weight, yprob in zip(combination_weights[1:], rslst[1:]):
      eprob = eprob + weight * yprob
    nllpr = tnf.nll_loss(eprob, label)
    error = likelihood_weight * nllpr
    optim.zero_grad()
    error.requires_grad_(True)
    error.backward()
    optim.step()
    watch += len(input)
    if only_record_last:
      errls = (nllpr, error)
    else:
      errls.append((nllpr, error))
    if echo_indexes is None or batid == echo_indexes[echos]:
      echos += 1
      print(f"NLL: {nllpr:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

def partial_ensemble_eval_on_nll(
  model, loader, combs, device,
  non_blocking = False,
  combination_weights = None,
  likelihood_weight = 1.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  if combination_weights is None:
    combination_weights = [1 / len(combs)] * len(combs)
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      rslst = prediction_on_partial_ensemble_label(model, input, combs)
      eprob = combination_weights[0] * rslst[0]
      for weigh, yprob in zip(combination_weights[1:], rslst[1:]):
        eprob = eprob + weigh * yprob
      nllac += tnf.nll_loss(eprob, label)
      right += (eprob.argmax(1) == label).type(torch.float).sum()
    ldrsz = len(loader)
    nllac = nllac / ldrsz
    error = likelihood_weight * nllac
    datsz = len(loader.sampler)
    right = right / datsz
    print(f"NLL: {nllac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, error, right

########################### nested_ensemble_train_xxxx_on_nll ###########################

def nested_ensemble_train_on_nll(
  model, loader, optim, device,
  non_blocking = False,
  likelihood_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  for batid, (input, label) in enumerate(loader):
    input = input.to(device, non_blocking = non_blocking)
    label = label.to(device, non_blocking = non_blocking)
    eprob, _ = prediction_on_label(model, input)
    nllpr = tnf.nll_loss(eprob, label)
    error = likelihood_weight * nllpr
    optim.zero_grad()
    error.requires_grad_(True)
    error.backward()
    optim.step()
    watch += len(input)
    if only_record_last:
      errls = (nllpr, error)
    else:
      errls.append((nllpr, error))
    if echo_indexes is None or batid == echo_indexes[echos]:
      echos += 1
      print(f"NLL: {nllpr:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

def nested_ensemble_train_parallel_on_nll(
  model, loader, optim, device, mpool,
  non_blocking = False,
  likelihood_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  # only with torch.no_grad(), what!
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  with Manager() as manager:
    queue = manager.Queue()
    event = manager.Event()
    for batid, (input, label) in enumerate(loader):
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      eprob, _ = prediction_on_label_with_event(model, input, mpool, queue, event)
      nllpr = tnf.nll_loss(eprob, label)
      error = likelihood_weight * nllpr
      optim.zero_grad()
      error.requires_grad_(True)
      error.backward()
      optim.step()
      watch += len(input)
      if only_record_last:
        errls = (nllpr, error)
      else:
        errls.append((nllpr, error))
      if echo_indexes is None or batid == echo_indexes[echos]:
        echos += 1
        print(f"NLL: {nllpr:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

########################### nested_ensemble_train_xxxx_on_var ###########################

def nested_ensemble_train_on_var(
  model, loader, optim, device,
  non_blocking = False,
  variance_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  for batid, (input, label) in enumerate(loader):
    input = input.to(device, non_blocking = non_blocking)
    label = label.to(device, non_blocking = non_blocking)
    probs, _ = prediction_on_subgraph(model, input, False)
    prvar = torch.var(probs, dim = -1)
    nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
    error = variance_weight * nlvar
    optim.zero_grad()
    error.requires_grad_(True)
    error.backward()
    optim.step()
    watch += len(input)
    if only_record_last:
      errls = (nlvar, error)
    else:
      errls.append((nlvar, error))
    if echo_indexes is None or batid == echo_indexes[echos]:
      echos += 1
      print(f"NLV: {nlvar:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

def nested_ensemble_train_parallel_on_var(
  model, loader, optim, device, mpool,
  non_blocking = False,
  variance_weight = 1.,
  only_record_last = True,
  echo_indexes = None
):
  # only with torch.no_grad(), what!
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  with Manager() as manager:
    queue = manager.Queue()
    event = manager.Event()
    for batid, (input, label) in enumerate(loader):
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      probs, _ = prediction_on_subgraph_with_event(model, input, mpool, queue, event, False)
      prvar = torch.var(probs, dim = -1)
      nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
      error = variance_weight * nlvar
      optim.zero_grad()
      error.requires_grad_(True)
      error.backward()
      optim.step()
      watch += len(input)
      if only_record_last:
        errls = (nlvar, error)
      else:
        errls.append((nlvar, error))
      if echo_indexes is None or batid == echo_indexes[echos]:
        echos += 1
        print(f"NLV: {nlvar:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

########################### nested_ensemble_train_xxxx_on_nll_and_var ###########################

def nested_ensemble_train_on_nll_and_var(
  model, loader, optim, device,
  non_blocking = False,
  likelihood_weight = 1.,
  variance_weight = 0.,
  only_record_last = True,
  echo_indexes = None
):
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  for batid, (input, label) in enumerate(loader):
    input = input.to(device, non_blocking = non_blocking)
    label = label.to(device, non_blocking = non_blocking)
    eprob, probs, _ = prediction_on_label_and_subgraph(model, input, True, False)
    prvar = torch.var(probs, dim = -1)
    nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
    nllpr = tnf.nll_loss(eprob, label)
    error = likelihood_weight * nllpr + variance_weight * nlvar
    optim.zero_grad()
    error.requires_grad_(True)
    error.backward()
    optim.step()
    watch += len(input)
    if only_record_last:
      errls = (nllpr, nlvar, error)
    else:
      errls.append((nllpr, nlvar, error))
    if echo_indexes is None or batid == echo_indexes[echos]:
      echos += 1
      print(f"NLL: {nllpr:>7f} | NLV: {nlvar:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

def nested_ensemble_train_parallel_on_nll_and_var(
  model, loader, optim, device, mpool,
  non_blocking = False,
  likelihood_weight = 1.,
  variance_weight = 0.,
  only_record_last = True,
  echo_indexes = None
):
  # only with torch.no_grad(), what!
  model.train()
  watch = 0
  echos = 0
  errls = []
  datsz = len(loader.sampler)
  with Manager() as manager:
    queue = manager.Queue()
    event = manager.Event()
    for batid, (input, label) in enumerate(loader):
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      eprob, probs, _ = prediction_on_label_and_subgraph_with_event(model, input, mpool, queue, event, True, False)
      prvar = torch.var(probs, dim = -1)
      nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
      nllpr = tnf.nll_loss(eprob, label)
      error = likelihood_weight * nllpr + variance_weight * nlvar
      optim.zero_grad()
      error.requires_grad_(True)
      error.backward()
      optim.step()
      watch += len(input)
      if only_record_last:
        errls = (nllpr, nlvar, error)
      else:
        errls.append((nllpr, nlvar, error))
      if echo_indexes is None or batid == echo_indexes[echos]:
        echos += 1
        print(f"NLL: {nllpr:>7f} | NLV: {nlvar:>7f} | ERR: {error:>7f} [{watch:>5d} / {datsz:>5d}]")
  return errls

########################### nested_ensemble_eval_xxxx_on_nll ###########################

def nested_ensemble_eval_on_nll(
  model, loader, device,
  non_blocking = False,
  likelihood_weight = 1.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      eprob, _ = prediction_on_label(model, input)
      nllpr = tnf.nll_loss(eprob, label)
      nllac += nllpr
      right += (eprob.argmax(1) == label).type(torch.float).sum()
    ldrsz = len(loader)
    nllac = nllac / ldrsz
    error = likelihood_weight * nllac
    datsz = len(loader.sampler)
    right = right / datsz
    print(f"NLL: {nllac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, error, right

def nested_ensemble_eval_parallel_on_nll(
  model, loader, device, mpool,
  non_blocking = False,
  likelihood_weight = 1.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, label in loader:
        input = input.to(device, non_blocking = non_blocking)
        label = label.to(device, non_blocking = non_blocking)
        eprob, _ = prediction_on_label_with_queue(model, input, mpool, queue)
        nllpr = tnf.nll_loss(eprob, label)
        nllac += nllpr
        right += (eprob.argmax(1) == label).type(torch.float).sum()
      ldrsz = len(loader)
      nllac = nllac / ldrsz
      error = likelihood_weight * nllac
      datsz = len(loader.sampler)
      right = right / datsz
      print(f"NLL: {nllac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, error, right

########################### nested_ensemble_eval_xxxx_on_var ###########################

def nested_ensemble_eval_on_var(
  model, loader, device,
  non_blocking = False,
  variance_weight = 1.,
  echo_end = ""
):
  model.eval()
  nlvac = 0
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      probs, _ = prediction_on_subgraph(model, input, False)
      prvar = torch.var(probs, dim = -1)
      nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
      nlvac += nlvar
    ldrsz = len(loader)
    nlvac = nlvac / ldrsz
    error = variance_weight * nlvac
    print(f"NLV: {nlvac:>7f} | ERR: {error:>7f}", end = echo_end)
  return nlvac, error

def nested_ensemble_eval_parallel_on_var(
  model, loader, device, mpool,
  non_blocking = False,
  variance_weight = 1.,
  echo_end = ""
):
  model.eval()
  nlvac = 0
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, label in loader:
        input = input.to(device, non_blocking = non_blocking)
        label = label.to(device, non_blocking = non_blocking)
        probs, _ = prediction_on_subgraph_with_queue(model, input, mpool, queue, False)
        prvar = torch.var(probs, dim = -1)
        nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
        nlvac += nlvar
      ldrsz = len(loader)
      nlvac = nlvac / ldrsz
      error = variance_weight * nlvac
      print(f"NLV: {nlvac:>7f} | ERR: {error:>7f}", end = echo_end)
  return nlvac, error

########################### nested_ensemble_eval_xxxx_on_nll_and_var ###########################

def nested_ensemble_eval_on_nll_and_var(
  model, loader, device,
  non_blocking = False,
  likelihood_weight = 1.,
  variance_weight = 0.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  nlvac = 0
  with torch.no_grad():
    for input, label in loader:
      # inlen = len(input)
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      eprob, probs, _ = prediction_on_label_and_subgraph(model, input, logcp = False)
      prvar = torch.var(probs, dim = -1)
      nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
      nllpr = tnf.nll_loss(eprob, label)
      nlvac += nlvar
      nllac += nllpr
      right += (eprob.argmax(1) == label).type(torch.float).sum()
    ldrsz = len(loader)
    nllac = nllac / ldrsz
    nlvac = nlvac / ldrsz
    error = likelihood_weight * nllac + variance_weight * nlvac
    datsz = len(loader.sampler)
    right = right / datsz
    print(f"NLL: {nllac:>7f} | NLV: {nlvac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, nlvac, error, right

def nested_ensemble_eval_parallel_on_nll_and_var(
  model, loader, device, mpool,
  non_blocking = False,
  likelihood_weight = 1.,
  variance_weight = 0.,
  echo_end = ""
):
  model.eval()
  right = 0
  nllac = 0
  nlvac = 0
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, label in loader:
        # inlen = len(input)
        input = input.to(device, non_blocking = non_blocking)
        label = label.to(device, non_blocking = non_blocking)
        eprob, probs, _ = prediction_on_label_and_subgraph_with_queue(model, input, mpool, queue, logcp = False)
        prvar = torch.var(probs, dim = -1)
        nlvar = torch.mean(-torch.log(tnf.relu(prvar * probs.shape[1])))
        nllpr = tnf.nll_loss(eprob, label)
        nlvac += nlvar
        nllac += nllpr
        right += (eprob.argmax(1) == label).type(torch.float).sum()
      ldrsz = len(loader)
      nllac = nllac / ldrsz
      nlvac = nlvac / ldrsz
      error = likelihood_weight * nllac + variance_weight * nlvac
      datsz = len(loader.sampler)
      right = right / datsz
      print(f"NLL: {nllac:>7f} | NLV: {nlvac:>7f} | ERR: {error:>7f} | Accuracy: {(100 * right):>3f}%", end = echo_end)
  return nllac, nlvac, error, right

########################### vote_counting_for_node_routing ###########################

def vote_counting_for_node_routing(model, loader, device, non_blocking = False):
  model.eval()
  grbuf = None
  leafs = None
  with torch.no_grad():
    for input, _ in loader:
      input = input.to(device, non_blocking = non_blocking)
      extra = model(input)
      # for every node
      graph = extra.get_graph()
      if grbuf is None:
        leafs = extra.get_leaves()
        grbuf = [
          block[0:3] + [None] + block[6:]
          for block in graph
        ]
      for blkid, block in enumerate(graph):
        if block[2]:
          prmat = block[4]
          maxid = torch.argmax(prmat, dim = -1)
          maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
          sumoh = torch.sum(maxid, dim = 0)
          sumpr = torch.sum(prmat, dim = 0)
          rslst = grbuf[blkid][3]
          if rslst is None:
            grbuf[blkid][3] = [sumoh, sumpr]
          else:
            rslst[0] = rslst[0] + sumoh
            rslst[1] = rslst[1] + sumpr
  return grbuf, leafs

########################### vote_counting_with_labels_for_node_routing ###########################

def vote_counting_with_labels_for_node_routing(model, loader, device, non_blocking = False):
  model.eval()
  grbuf = None
  leafs = None
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      extra = model(input)
      graph = extra.get_graph()
      if grbuf is None:
        leafs = extra.get_leaves()
        grbuf = [
          block[0:3] + [{} if block[2] else None] + block[6:]
          for block in graph
        ]
      # for different labels
      unilb = torch.unique(label, return_inverse = True)
      for index, value in enumerate(unilb[0]):
        value = value.item()
        idxes = (unilb[1] == index).nonzero(as_tuple = True)[0]
        # for every node
        for blkid, block in enumerate(graph):
          if block[2]:
            prmat = block[4][idxes]
            maxid = torch.argmax(prmat, dim = -1)
            maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
            sumoh = torch.sum(maxid, dim = 0)
            sumpr = torch.sum(prmat, dim = 0)
            rsmap = grbuf[blkid][3]
            try:
              oldrs = rsmap[value]
              sumoh = sumoh + oldrs[0]
              sumpr = sumpr + oldrs[1]
              rsmap[value] = (sumoh, sumpr)
            except KeyError:
              rsmap[value] = (sumoh, sumpr)
  return grbuf, leafs

########################### vote_counting_xxx_for_leaf_combinations ###########################

def vote_counting_for_leaf_combinations(model, loader, device, non_blocking = False):
  model.eval()
  rslst = None
  with torch.no_grad():
    for input, _ in loader:
      input = input.to(device, non_blocking = non_blocking)
      probs, _ = prediction_on_subgraph(model, input, logpr = False)
      maxid = torch.argmax(probs, dim = -1)
      maxid = tnf.one_hot(maxid, num_classes = probs.shape[-1])
      sumoh = torch.sum(maxid, dim = 0)
      sumpr = torch.sum(probs, dim = 0)
      if rslst is None:
        rslst = [sumoh, sumpr]
      else:
        rslst[0] = rslst[0] + sumoh
        rslst[1] = rslst[1] + sumpr
  return rslst

def vote_counting_parallel_for_leaf_combinations(model, loader, device, mpool, non_blocking = False):
  model.eval()
  rslst = None
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, _ in loader:
        input = input.to(device, non_blocking = non_blocking)
        probs, _ = prediction_on_subgraph_with_queue(model, input, mpool, queue, logpr = False)
        maxid = torch.argmax(probs, dim = -1)
        maxid = tnf.one_hot(maxid, num_classes = probs.shape[-1])
        sumoh = torch.sum(maxid, dim = 0)
        sumpr = torch.sum(probs, dim = 0)
        if rslst is None:
          rslst = [sumoh, sumpr]
        else:
          rslst[0] = rslst[0] + sumoh
          rslst[1] = rslst[1] + sumpr
  return rslst

########################### vote_counting_xxx_with_labels_for_leaf_combinations ###########################

def vote_counting_with_labels_for_leaf_combinations(model, loader, device, non_blocking = False):
  model.eval()
  rdict = None
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      probs, _ = prediction_on_subgraph(model, input, logpr = False)
      # for different labels
      unilb = torch.unique(label, return_inverse = True)
      for index, value in enumerate(unilb[0]):
        value = value.item()
        idxes = (unilb[1] == index).nonzero(as_tuple = True)[0]
        prmat = probs[idxes]
        maxid = torch.argmax(prmat, dim = -1)
        maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
        sumoh = torch.sum(maxid, dim = 0)
        sumpr = torch.sum(prmat, dim = 0)
        try:
          oldrs = rdict[value]
          sumoh = sumoh + oldrs[0]
          sumpr = sumpr + oldrs[1]
          rdict[value] = (sumoh, sumpr)
        except KeyError:
          rdict[value] = (sumoh, sumpr)
        except TypeError:
          rdict = {value : (sumoh, sumpr)}
  return rdict

def vote_counting_parallel_with_labels_for_leaf_combinations(model, loader, device, mpool, non_blocking = False):
  model.eval()
  rdict = None
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, label in loader:
        input = input.to(device, non_blocking = non_blocking)
        label = label.to(device, non_blocking = non_blocking)
        probs, _ = prediction_on_subgraph_with_queue(model, input, mpool, queue, logpr = False)
        # for different labels
        unilb = torch.unique(label, return_inverse = True)
        for index, value in enumerate(unilb[0]):
          value = value.item()
          idxes = (unilb[1] == index).nonzero(as_tuple = True)[0]
          prmat = probs[idxes]
          maxid = torch.argmax(prmat, dim = -1)
          maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
          sumoh = torch.sum(maxid, dim = 0)
          sumpr = torch.sum(prmat, dim = 0)
          try:
            oldrs = rdict[value]
            sumoh = sumoh + oldrs[0]
            sumpr = sumpr + oldrs[1]
            rdict[value] = (sumoh, sumpr)
          except KeyError:
            rdict[value] = (sumoh, sumpr)
          except TypeError:
            rdict = {value : (sumoh, sumpr)}
  return rdict

########################### vote_counting_xxx_for_leaf_combinations_and_node_routing ###########################

def vote_counting_for_leaf_combinations_and_node_routing(model, loader, device, non_blocking = False):
  model.eval()
  rslst = None
  grbuf = None
  leafs = None
  with torch.no_grad():
    for input, _ in loader:
      input = input.to(device, non_blocking = non_blocking)
      probs, extra = prediction_on_subgraph(model, input, logpr = False)
      maxid = torch.argmax(probs, dim = -1)
      maxid = tnf.one_hot(maxid, num_classes = probs.shape[-1])
      sumoh = torch.sum(maxid, dim = 0)
      sumpr = torch.sum(probs, dim = 0)
      if rslst is None:
        rslst = [sumoh, sumpr]
      else:
        rslst[0] = rslst[0] + sumoh
        rslst[1] = rslst[1] + sumpr
      # for every node
      graph = extra.get_graph()
      if grbuf is None:
        leafs = extra.get_leaves()
        grbuf = [
          block[0:3] + [None] + block[6:]
          for block in graph
        ]
      for blkid, block in enumerate(graph):
        if block[2]:
          prmat = block[5]
          maxid = torch.argmax(prmat, dim = -1)
          maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
          sumoh = torch.sum(maxid, dim = 0)
          sumpr = torch.sum(prmat, dim = 0)
          rsarr = grbuf[blkid][3]
          if rsarr is None:
            grbuf[blkid][3] = [sumoh, sumpr]
          else:
            rsarr[0] = rsarr[0] + sumoh
            rsarr[1] = rsarr[1] + sumpr
  return rslst, grbuf, leafs

def vote_counting_parallel_for_leaf_combinations_and_node_routing(model, loader, device, mpool, non_blocking = False):
  model.eval()
  rslst = None
  grbuf = None
  leafs = None
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, _ in loader:
        input = input.to(device, non_blocking = non_blocking)
        probs, extra = prediction_on_subgraph_with_queue(model, input, mpool, queue, logpr = False)
        maxid = torch.argmax(probs, dim = -1)
        maxid = tnf.one_hot(maxid, num_classes = probs.shape[-1])
        sumoh = torch.sum(maxid, dim = 0)
        sumpr = torch.sum(probs, dim = 0)
        if rslst is None:
          rslst = [sumoh, sumpr]
        else:
          rslst[0] = rslst[0] + sumoh
          rslst[1] = rslst[1] + sumpr
        # for every node
        graph = extra.get_graph()
        if grbuf is None:
          leafs = extra.get_leaves()
          grbuf = [
            block[0:3] + [None] + block[6:]
            for block in graph
          ]
        for blkid, block in enumerate(graph):
          if block[2]:
            prmat = block[5]
            maxid = torch.argmax(prmat, dim = -1)
            maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
            sumoh = torch.sum(maxid, dim = 0)
            sumpr = torch.sum(prmat, dim = 0)
            rsarr = grbuf[blkid][3]
            if rsarr is None:
              grbuf[blkid][3] = [sumoh, sumpr]
            else:
              rsarr[0] = rsarr[0] + sumoh
              rsarr[1] = rsarr[1] + sumpr
  return rslst, grbuf, leafs

########################### vote_counting_xxx_with_labels_for_leaf_combinations_and_node_routing ###########################

def vote_counting_with_labels_for_leaf_combinations_and_node_routing(model, loader, device, non_blocking = False):
  model.eval()
  rdict = None
  grbuf = None
  leafs = None
  with torch.no_grad():
    for input, label in loader:
      input = input.to(device, non_blocking = non_blocking)
      label = label.to(device, non_blocking = non_blocking)
      probs, extra = prediction_on_subgraph(model, input, logpr = False)
      graph = extra.get_graph()
      if grbuf is None:
        leafs = extra.get_leaves()
        grbuf = [
          block[0:3] + [{} if block[2] else None] + block[6:]
          for block in graph
        ]
      # for different labels
      unilb = torch.unique(label, return_inverse = True)
      for index, value in enumerate(unilb[0]):
        value = value.item()
        idxes = (unilb[1] == index).nonzero(as_tuple = True)[0]
        prmat = probs[idxes]
        maxid = torch.argmax(prmat, dim = -1)
        maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
        sumoh = torch.sum(maxid, dim = 0)
        sumpr = torch.sum(prmat, dim = 0)
        try:
          oldrs = rdict[value]
          sumoh = sumoh + oldrs[0]
          sumpr = sumpr + oldrs[1]
          rdict[value] = (sumoh, sumpr)
        except KeyError:
          rdict[value] = (sumoh, sumpr)
        except TypeError:
          rdict = {value : (sumoh, sumpr)}
        # for every node
        for blkid, block in enumerate(graph):
          if block[2]:
            prmat = block[5][idxes]
            maxid = torch.argmax(prmat, dim = -1)
            maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
            sumoh = torch.sum(maxid, dim = 0)
            sumpr = torch.sum(prmat, dim = 0)
            rsmap = grbuf[blkid][3]
            try:
              oldrs = rsmap[value]
              sumoh = sumoh + oldrs[0]
              sumpr = sumpr + oldrs[1]
              rsmap[value] = (sumoh, sumpr)
            except KeyError:
              rsmap[value] = (sumoh, sumpr)
  return rdict, grbuf, leafs

def vote_counting_parallel_with_labels_for_leaf_combinations_and_node_routing(model, loader, device, mpool, non_blocking = False):
  model.eval()
  rdict = None
  grbuf = None
  leafs = None
  with Manager() as manager:
    queue = manager.Queue()
    with torch.no_grad():
      for input, label in loader:
        input = input.to(device, non_blocking = non_blocking)
        label = label.to(device, non_blocking = non_blocking)
        probs, extra = prediction_on_subgraph_with_queue(model, input, mpool, queue, logpr = False)
        graph = extra.get_graph()
        if grbuf is None:
          leafs = extra.get_leaves()
          grbuf = [
            block[0:3] + [{} if block[2] else None] + block[6:]
            for block in graph
          ]
        # for different labels
        unilb = torch.unique(label, return_inverse = True)
        for index, value in enumerate(unilb[0]):
          value = value.item()
          idxes = (unilb[1] == index).nonzero(as_tuple = True)[0]
          prmat = probs[idxes]
          maxid = torch.argmax(prmat, dim = -1)
          maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
          sumoh = torch.sum(maxid, dim = 0)
          sumpr = torch.sum(prmat, dim = 0)
          try:
            oldrs = rdict[value]
            sumoh = sumoh + oldrs[0]
            sumpr = sumpr + oldrs[1]
            rdict[value] = (sumoh, sumpr)
          except KeyError:
            rdict[value] = (sumoh, sumpr)
          except TypeError:
            rdict = {value : (sumoh, sumpr)}
          # for every node
          for blkid, block in enumerate(graph):
            if block[2]:
              prmat = block[5][idxes]
              maxid = torch.argmax(prmat, dim = -1)
              maxid = tnf.one_hot(maxid, num_classes = prmat.shape[-1])
              sumoh = torch.sum(maxid, dim = 0)
              sumpr = torch.sum(prmat, dim = 0)
              rsmap = grbuf[blkid][3]
              try:
                oldrs = rsmap[value]
                sumoh = sumoh + oldrs[0]
                sumpr = sumpr + oldrs[1]
                rsmap[value] = (sumoh, sumpr)
              except KeyError:
                rsmap[value] = (sumoh, sumpr)
  return rdict, grbuf, leafs

################################# some method for getting multiple leaves ###################################

def get_top_k_leaves(model, loader, device, non_blocking = False, normalization = True, hard = False, k = 0):
  rslst, grbuf, leafs = vote_counting_for_leaf_combinations_and_node_routing(model, loader, device, non_blocking)
  leafm = {}
  index = 0
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      sumoh = rslst[0][index] / count
      sumpr = rslst[1][index] / count
      index += 1
      for blkid in blktp:
        try:
          entry = leafm[blkid]
          entry[0] += sumoh
          entry[1] += sumpr
        except KeyError:
          entry = [sumoh, sumpr]
          leafm[blkid] = entry
  leafm = [(blkid, entry[0], entry[1]) for blkid, entry in leafm.items()]
  if hard:
    leafm.sort(key = lambda entry: entry[1], reverse = True)
  else:
    leafm.sort(key = lambda entry: entry[2], reverse = True)
  rslst = []
  if normalization:
    datsz = len(loader.sampler)
    if 0 < k and k < len(leafs):
      for index in range(k):
        blkid, sumoh, sumpr = leafm[index]
        rslst.append((grbuf[blkid][1], sumoh / datsz, sumpr / datsz))
    else:
      for blkid, sumoh, sumpr in leafm:
        rslst.append((grbuf[blkid][1], sumoh / datsz, sumpr / datsz))
  else:
    if 0 < k and k < len(leafs):
      for index in range(k):
        blkid, sumoh, sumpr = leafm[index]
        rslst.append((grbuf[blkid][1], sumoh, sumpr))
    else:
      for blkid, sumoh, sumpr in leafm:
        rslst.append((grbuf[blkid][1], sumoh, sumpr))
  return rslst

def get_multiple_leaves(model, loader, device, non_blocking = False, normalization = True, hard = False, threshold = 0.8):
  rslst = get_top_k_leaves(model, loader, device, non_blocking, normalization, hard, 0)
  total = None
  leafs = None
  if hard:
    for rsdat in rslst:
      if leafs is None:
        leafs = [rsdat]
        total = rsdat[1]
      else:
        leafs.append(rsdat)
        total = total + rsdat[1]
      if threshold <= total:
        break
  else:
    for rsdat in rslst:
      if leafs is None:
        leafs = [rsdat]
        total = rsdat[2]
      else:
        leafs.append(rsdat)
        total = total + rsdat[2]
      if threshold <= total:
        break
  return leafs, total

########################### some method for getting multiple optimal combinations ###########################

def get_top_k_leaf_combinations(model, loader, device, non_blocking = False, normalization = True, hard = False, k = 0):
  rslst, grbuf, leafs = vote_counting_for_leaf_combinations_and_node_routing(model, loader, device, non_blocking)
  blkts = []
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      index = len(blkts)
      sumoh = rslst[0][index]
      sumpr = rslst[1][index]
      blkts.append((blktp, sumoh, sumpr))
  if hard:
    blkts.sort(key = lambda entry: entry[1], reverse = True)
  else:
    blkts.sort(key = lambda entry: entry[2], reverse = True)
  rslst = []
  if normalization:
    datsz = len(loader.sampler)
    if 0 < k and k < len(blkts):
      for index in range(k):
        blktp, sumoh, sumpr = blkts[index]
        nodes = {grbuf[blkid][1] for blkid in blktp}
        rslst.append((nodes, sumoh / datsz, sumpr / datsz))
    else:
      for blktp, sumoh, sumpr in blkts:
        nodes = {grbuf[blkid][1] for blkid in blktp}
        rslst.append((nodes, sumoh / datsz, sumpr / datsz))
  else:
    if 0 < k and k < len(blkts):
      for index in range(k):
        blktp, sumoh, sumpr = blkts[index]
        nodes = {grbuf[blkid][1] for blkid in blktp}
        rslst.append((nodes, sumoh, sumpr))
    else:
      for blktp, sumoh, sumpr in blkts:
        nodes = {grbuf[blkid][1] for blkid in blktp}
        rslst.append((nodes, sumoh, sumpr))
  return rslst

def get_multiple_leaf_combinations(model, loader, device, non_blocking = False, normalization = True, hard = False, threshold = 0.8):
  rslst = get_top_k_leaf_combinations(model, loader, device, non_blocking, normalization, hard, 0)
  total = None
  combs = None
  if hard:
    for rsdat in rslst:
      if combs is None:
        combs = [rsdat]
        total = rsdat[1]
      else:
        combs.append(rsdat)
        total = total + rsdat[1]
      if threshold <= total:
        break
  else:
    for rsdat in rslst:
      if combs is None:
        combs = [rsdat]
        total = rsdat[2]
      else:
        combs.append(rsdat)
        total = total + rsdat[2]
      if threshold <= total:
        break
  return combs, total

def get_union_of_leaf_combinations(model, loader, device, non_blocking = False, hard = False, threshold = 0.8):
  rslst = get_top_k_leaf_combinations(model, loader, device, non_blocking, True, hard, 0)
  total = None
  union = None
  if hard:
    for nodes, sumoh, _ in rslst:
      if union is None:
        union = nodes
        total = sumoh
      else:
        union = union.union(nodes)
        total = total + sumoh
      if threshold <= total:
        break
  else:
    for nodes, _, sumpr in rslst:
      if union is None:
        union = nodes
        total = sumpr
      else:
        union = union.union(nodes)
        total = total + sumpr
      if threshold <= total:
        break
  return union, total

def count_model_params(model):
  total = 0
  for params in model.parameters():
    params_number = 1
    for dim_size in params.size():
      params_number *= dim_size
    total += params_number
  return total

def count_model_flops(model, input):
  model.eval()
  with torch.no_grad():
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    old_stdout.flush()
    old_stderr.flush()
    sys.stdout = open(os.devnull,'w')
    sys.stderr = sys.stdout

    flops = FlopCountAnalysis(model, input)
    flops = flops.total()

    sys.stdout.flush()
    sys.stdout.close()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    return flops