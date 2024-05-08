# Author: Jiahao Li

# https://github.com/pytorch/pytorch/issues/67887
# https://pytorch.org/docs/stable/multiprocessing.html
# https://superfastpython.com/multiprocessing-pool-share-queue/
# https://superfastpython.com/multiprocessing-pool-event/
# https://discuss.pytorch.org/t/using-torch-tensor-over-multiprocessing-queue-process-fails/2847/2
# https://stackoverflow.com/questions/3827065/can-i-use-a-multiprocessing-queue-in-a-function-called-by-pool-imap

import torch
from itertools import combinations

def share_graph_tensors(graph, style = 0):
  if style == 0:
    for block in graph:
      if block[2] is None:
        block[3].share_memory_()
      block[4].share_memory_()
  else:
    for block in graph:
      block[3].share_memory_()
      block[4].share_memory_()
      block[5].share_memory_()

########################### prediction_on_global_ensemble_label ###########################

def prediction_on_global_ensemble_label(model, input, logpr = True, delta = 1e-10):
  graph, leafs = model(input)
  yprob = None
  for blkid in leafs:
    if yprob is None:
      yprob = torch.exp(graph[blkid][3])
    else:
      yprob = yprob + torch.exp(graph[blkid][3])
  yprob = yprob / len(leafs)
  if logpr:
    yprob = torch.log(yprob + delta)
  return yprob

########################### prediction_on_partial_ensemble_label ###########################

def prediction_on_partial_ensemble_label(model, input, combs, logpr = True, delta = 1e-10):
  if 1 < len(combs):
    union = combs[0].union(combs[1], *combs[2:])
  elif 0 < len(combs):
    union = combs[0]
  else:
    union = None
  hmode = model.get_handle_mode()
  model.set_mode_as_multiple_instances_without_policies_given_leaf_nodes()
  graph, leafs = model(input, union)
  model.set_handle_mode(hmode)
  leafs = {graph[blkid][1] : blkid for blkid in leafs}
  rslst = []
  if logpr:
    for nodes in combs:
      yprob = None
      for node in nodes:
        blkid = leafs[node]
        if yprob is None:
          yprob = torch.exp(graph[blkid][3])
        else:
          yprob = yprob + torch.exp(graph[blkid][3])
      yprob = yprob / len(nodes)
      yprob = torch.log(yprob + delta)
      rslst.append(yprob)
  else:
    for nodes in combs:
      yprob = None
      for node in nodes:
        blkid = leafs[node]
        if yprob is None:
          yprob = torch.exp(graph[blkid][3])
        else:
          yprob = yprob + torch.exp(graph[blkid][3])
      yprob = yprob / len(nodes)
      rslst.append(yprob)
  return rslst

########################### prediction_on_subgraph_xxxx ###########################

def prediction_on_subgraph(model, input, logpr = True, delta = 1e-10):
  probs = []
  extra = model(input)
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      cprob = extra(blktp)
      if cprob is not False:
        probs.append(cprob)
  if probs:
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logpr:
        probs = torch.exp(probs)
    else:
      if logpr:
        probs = torch.log(probs + delta)
  return probs, extra

def prediction_on_subgraph_with_queue_proxy(index, blktp, extra, queue):
  queue.put((index, extra(blktp)))

def prediction_on_subgraph_with_queue(model, input, mpool, queue, logpr = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  # get all block tuples
  total = 0
  items = []
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      items.append((total, blktp, extra, queue))
      total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_subgraph_with_queue_proxy,
    items
  )
  # get result
  probs = []
  while total:
    total -= 1
    entry = queue.get()
    index, cprob = entry
    if cprob is not False:
      probs.append((index, cprob))
    # del entry
  if probs:
    probs.sort(key = lambda entry: entry[0])
    probs = [entry[1] for entry in probs]
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logpr:
        probs = torch.exp(probs)
    else:
      if logpr:
        probs = torch.log(probs + delta)
  return probs, extra

def prediction_on_subgraph_with_event_proxy(tplst, extra, queue, event):
  for index, blktp in tplst:
    queue.put((index, extra(blktp)))
  event.wait()

def prediction_on_subgraph_with_event(model, input, mpool, queue, event, logpr = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  procs = mpool._processes
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  total = 2 ** len(leafs) - 1
  if procs < total:
    # get all block tuples
    total = 0
    tplst = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        tplst.append((total, blktp))
        total += 1
    # arrange block tuples
    tparr = []
    nhalf = total >> 1
    index = nhalf
    while index:
      tparr.append(tplst[total - index])
      index -= 1
      tparr.append(tplst[index])
    if total & 1:
      tparr.append(tplst[nhalf])
    # slice tuple array
    index = 0
    items = []
    tplst = []
    tasks = min(1, total // procs)
    for blktp in tparr:
      tplst.append(blktp)
      index += 1
      if len(items) < procs and index % tasks == 0:
        items.append((tplst, extra, queue, event))
        tplst = []
    index = 0
    for blktp in tplst:
      items[index][0].append(blktp)
      index += 1
  else:
    total = 0
    items = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        items.append(([(total, blktp)], extra, queue, event))
        total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_subgraph_with_event_proxy,
    items
  )
  # get result
  probs = []
  while total:
    total -= 1
    entry = queue.get()
    index, cprob = entry
    if cprob is not False:
      probs.append((index, cprob))
    # del entry
  if probs:
    probs.sort(key = lambda entry: entry[0])
    probs = [entry[1] for entry in probs]
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logpr:
        probs = torch.exp(probs)
    else:
      if logpr:
        probs = torch.log(probs + delta)
  return probs, extra

########################### prediction_on_label_xxxx ###########################

def prediction_on_label_proxy(graph, blktp, extra):
  cprob = extra(blktp)
  if cprob is False:
    return False
  yprob = None
  for blkid in blktp:
    if yprob is None:
      yprob = torch.exp(graph[blkid][3])
    else:
      yprob = yprob + torch.exp(graph[blkid][3])
  count = len(blktp)
  if extra.get_probabilty_multiplier_type():
    if yprob is None:
      jprob = torch.exp(cprob.unsqueeze(1))
    else:
      jprob = yprob / count * torch.exp(cprob.unsqueeze(1))
  else:
    if yprob is None:
      jprob = cprob.unsqueeze(1)
    else:
      jprob = yprob / count * cprob.unsqueeze(1)
  return jprob

def prediction_on_label(model, input, logpr = True, delta = 1e-10):
  eprob = None
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      jprob = prediction_on_label_proxy(graph, blktp, extra)
      if jprob is not False:
        if eprob is None:
          eprob = jprob
        else:
          eprob = eprob + jprob
  if eprob is not None and logpr:
    eprob = torch.log(eprob + delta)
  return eprob, extra

def prediction_on_label_with_queue_proxy(graph, blktp, extra, queue):
  queue.put(prediction_on_label_proxy(graph, blktp, extra))

def prediction_on_label_with_queue(model, input, mpool, queue, logpr = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  total = 2 ** len(leafs) - 1
  items = []
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      items.append((graph, blktp, extra, queue))
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_label_with_queue_proxy,
    items
  )
  # get result
  eprob = None
  while total:
    total -= 1
    jprob = queue.get()
    if jprob is not False:
      if eprob is None:
        eprob = jprob
      else:
        eprob = eprob + jprob
    # del jprob
  if eprob is not None and logpr:
    eprob = torch.log(eprob + delta)
  return eprob, extra

def prediction_on_label_with_event_proxy(graph, tplst, extra, queue, event):
  for blktp in tplst:
    queue.put(prediction_on_label_proxy(graph, blktp, extra))
  event.wait()

def prediction_on_label_with_event(model, input, mpool, queue, event, logpr = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  procs = mpool._processes
  total = 2 ** len(leafs) - 1
  if procs < total:
    # get all block tuples
    tplst = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        tplst.append(blktp)
    # arrange block tuples
    tparr = []
    nhalf = total >> 1
    index = nhalf
    while index:
      tparr.append(tplst[total - index])
      index -= 1
      tparr.append(tplst[index])
    if total & 1:
      tparr.append(tplst[nhalf])
    # slice tuple array
    index = 0
    items = []
    tplst = []
    tasks = min(1, total // procs)
    for blktp in tparr:
      tplst.append(blktp)
      index += 1
      if len(items) < procs and index % tasks == 0:
        items.append((graph, tplst, extra, queue, event))
        tplst = []
    index = 0
    for blktp in tplst:
      items[index][1].append(blktp)
      index += 1
  else:
    items = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        items.append((graph, [blktp], extra, queue, event))
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_label_with_event_proxy,
    items
  )
  # get result
  eprob = None
  while total:
    total -= 1
    jprob = queue.get()
    if jprob is not False:
      if eprob is None:
        eprob = jprob
      else:
        eprob = eprob + jprob
    # del jprob
  if eprob is not None and logpr:
    eprob = torch.log(eprob + delta)
  return eprob, extra

########################### prediction_on_label_and_subgraph_xxxx ###########################

def prediction_on_label_and_subgraph_proxy(graph, blktp, extra):
  cprob = extra(blktp)
  if cprob is False:
    return False
  yprob = None
  for blkid in blktp:
    if yprob is None:
      yprob = torch.exp(graph[blkid][3])
    else:
      yprob = yprob + torch.exp(graph[blkid][3])
  count = len(blktp)
  if extra.get_probabilty_multiplier_type():
    if yprob is None:
      jprob = torch.exp(cprob.unsqueeze(1))
    else:
      jprob = yprob / count * torch.exp(cprob.unsqueeze(1))
  else:
    if yprob is None:
      jprob = cprob.unsqueeze(1)
    else:
      jprob = yprob / count * cprob.unsqueeze(1)
  return jprob, cprob

def prediction_on_label_and_subgraph(model, input, loglp = True, logcp = True, delta = 1e-10):
  probs = []
  eprob = None
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      entry = prediction_on_label_and_subgraph_proxy(graph, blktp, extra)
      if entry is not False:
        jprob, cprob = entry
        probs.append(cprob)
        if eprob is None:
          eprob = jprob
        else:
          eprob = eprob + jprob
  if probs:
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logcp:
        probs = torch.exp(probs)
    else:
      if logcp:
        probs = torch.log(probs + delta)
    if loglp:
      eprob = torch.log(eprob + delta)
  return eprob, probs, extra

def prediction_on_label_and_subgraph_with_queue_proxy(index, graph, blktp, extra, queue):
  queue.put((index, prediction_on_label_and_subgraph_proxy(graph, blktp, extra)))

def prediction_on_label_and_subgraph_with_queue(model, input, mpool, queue, loglp = True, logcp = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  total = 0
  items = []
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      items.append((total, graph, blktp, extra, queue))
      total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_label_and_subgraph_with_queue_proxy,
    items
  )
  # get result
  eprob = None
  probs = []
  while total:
    total -= 1
    entry = queue.get()
    index, entry = entry
    if entry is not False:
      jprob, cprob = entry
      probs.append((index, cprob))
      if eprob is None:
        eprob = jprob
      else:
        eprob = eprob + jprob
    # del entry
  if probs:
    probs.sort(key = lambda entry: entry[0])
    probs = [entry[1] for entry in probs]
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logcp:
        probs = torch.exp(probs)
    else:
      if logcp:
        probs = torch.log(probs + delta)
    if loglp:
      eprob = torch.log(eprob + delta)
  return eprob, probs, extra

def prediction_on_label_and_subgraph_with_event_proxy(graph, tplst, extra, queue, event):
  for index, blktp in tplst:
    queue.put((index, prediction_on_label_and_subgraph_proxy(graph, blktp, extra)))
  event.wait()

def prediction_on_label_and_subgraph_with_event(model, input, mpool, queue, event, loglp = True, logcp = True, delta = 1e-10):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  procs = mpool._processes
  total = 2 ** len(leafs) - 1
  if procs < total:
    # get all block tuples
    total = 0
    tplst = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        tplst.append((total, blktp))
        total += 1
    # arrange block tuples
    tparr = []
    nhalf = total >> 1
    index = nhalf
    while index:
      tparr.append(tplst[total - index])
      index -= 1
      tparr.append(tplst[index])
    if total & 1:
      tparr.append(tplst[nhalf])
    # slice tuple array
    index = 0
    items = []
    tplst = []
    tasks = min(1, total // procs)
    for blktp in tparr:
      tplst.append(blktp)
      index += 1
      if len(items) < procs and index % tasks == 0:
        items.append((graph, tplst, extra, queue, event))
        tplst = []
    index = 0
    for blktp in tplst:
      items[index][1].append(blktp)
      index += 1
  else:
    total = 0
    items = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        items.append((graph, [(total, blktp)], extra, queue, event))
        total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_label_and_subgraph_with_event_proxy,
    items
  )
  # get result
  eprob = None
  probs = []
  while total:
    total -= 1
    entry = queue.get()
    index, entry = entry
    if entry is not False:
      jprob, cprob = entry
      probs.append((index, cprob))
      if eprob is None:
        eprob = jprob
      else:
        eprob = eprob + jprob
    # del entry
  if probs:
    probs.sort(key = lambda entry: entry[0])
    probs = [entry[1] for entry in probs]
    probs = torch.stack(probs, dim = -1)
    if extra.get_probabilty_multiplier_type():
      if not logcp:
        probs = torch.exp(probs)
    else:
      if logcp:
        probs = torch.log(probs + delta)
    if loglp:
      eprob = torch.log(eprob + delta)
  return eprob, probs, extra

########################### prediction_on_output_xxxx ###########################

def prediction_on_output_proxy(graph, blktp):
  resls = []
  for blkid in blktp:
    resls.append(graph[blkid][3])
  return resls

def prediction_on_output(model, input):
  outls = []
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      resls = prediction_on_output_proxy(graph, blktp)
      outls.append(resls)
  return outls, extra

def prediction_on_output_with_queue_proxy(index, graph, blktp, queue):
  resls = prediction_on_output_proxy(graph, blktp)
  queue.put((index, resls))

def prediction_on_output_with_queue(model, input, mpool, queue):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  total = 0
  items = []
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      items.append((total, graph, blktp, queue))
      total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_output_with_queue_proxy,
    items
  )
  # get result
  items = []
  while total:
    total -= 1
    entry = queue.get()
    items.append(entry)
    # del entry
  items.sort(key = lambda entry: entry[0])
  items = [entry[1] for entry in items]
  return items, extra

def prediction_on_output_with_event_proxy(graph, tplst, queue, event):
  for index, blktp in tplst:
    resls = prediction_on_output_proxy(graph, blktp)
    queue.put((index, resls))
  event.wait()

def prediction_on_output_with_event(model, input, mpool, queue, event):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  procs = mpool._processes
  total = 2 ** len(leafs) - 1
  if procs < total:
    # get all block tuples
    total = 0
    tplst = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        tplst.append((total, blktp))
        total += 1
    # arrange block tuples
    tparr = []
    nhalf = total >> 1
    index = nhalf
    while index:
      tparr.append(tplst[total - index])
      index -= 1
      tparr.append(tplst[index])
    if total & 1:
      tparr.append(tplst[nhalf])
    # slice tuple array
    index = 0
    items = []
    tplst = []
    tasks = min(1, total // procs)
    for blktp in tparr:
      tplst.append(blktp)
      index += 1
      if len(items) < procs and index % tasks == 0:
        items.append((graph, tplst, queue, event))
        tplst = []
    index = 0
    for blktp in tplst:
      items[index][1].append(blktp)
      index += 1
  else:
    total = 0
    items = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        items.append((graph, [(total, blktp)], queue, event))
        total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_output_with_event_proxy,
    items
  )
  # get result
  items = []
  while total:
    total -= 1
    entry = queue.get()
    items.append(entry)
    # del entry
  items.sort(key = lambda entry: entry[0])
  items = [entry[1] for entry in items]
  return items, extra

########################### prediction_on_output_and_subgraph_xxxx ###########################

def prediction_on_output_and_subgraph_proxy(graph, blktp, extra):
  cprob = extra(blktp)
  resls = []
  if cprob is not False:
    for blkid in blktp:
      resls.append(graph[blkid][3])
    resls.append(cprob)
  return resls

def prediction_on_output_and_subgraph(model, input):
  outls = []
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      resls = prediction_on_output_and_subgraph_proxy(graph, blktp)
      if resls:
        outls.append(resls)
  return outls, extra

def prediction_on_output_and_subgraph_with_queue_proxy(index, graph, blktp, extra, queue):
  resls = prediction_on_output_and_subgraph_proxy(graph, blktp, extra)
  queue.put((index, resls))

def prediction_on_output_and_subgraph_with_queue(model, input, mpool, queue):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  total = 0
  items = []
  for count in range(1, len(leafs) + 1):
    for blktp in combinations(leafs, count):
      items.append((total, graph, blktp, extra, queue))
      total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_output_and_subgraph_with_queue_proxy,
    items
  )
  # get result
  items = []
  while total:
    total -= 1
    entry = queue.get()
    resls = entry[1]
    if resls:
      items.append(entry)
    # del entry
  items.sort(key = lambda entry: entry[0])
  items = [entry[1] for entry in items]
  return items, extra

def prediction_on_output_and_subgraph_with_event_proxy(graph, tplst, extra, queue, event):
  for index, blktp in tplst:
    resls = prediction_on_output_and_subgraph_proxy(graph, blktp, extra)
    queue.put((index, resls))
  event.wait()

def prediction_on_output_and_subgraph_with_event(model, input, mpool, queue, event):
  # Cowardly refusing to serialize non-leaf tensor which requires_grad,
  # since autograd does not support crossing process boundaries.
  extra = model(input)
  graph = extra.get_graph()
  leafs = extra.get_leaves()
  procs = mpool._processes
  total = 2 ** len(leafs) - 1
  if procs < total:
    # get all block tuples
    total = 0
    tplst = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        tplst.append((total, blktp))
        total += 1
    # arrange block tuples
    tparr = []
    nhalf = total >> 1
    index = nhalf
    while index:
      tparr.append(tplst[total - index])
      index -= 1
      tparr.append(tplst[index])
    if total & 1:
      tparr.append(tplst[nhalf])
    # slice tuple array
    index = 0
    items = []
    tplst = []
    tasks = min(1, total // procs)
    for blktp in tparr:
      tplst.append(blktp)
      index += 1
      if len(items) < procs and index % tasks == 0:
        items.append((graph, tplst, extra, queue, event))
        tplst = []
    index = 0
    for blktp in tplst:
      items[index][1].append(blktp)
      index += 1
  else:
    total = 0
    items = []
    for count in range(1, len(leafs) + 1):
      for blktp in combinations(leafs, count):
        items.append((graph, [(total, blktp)], extra, queue, event))
        total += 1
  # execute async
  share_graph_tensors(graph)
  extra.buffer_off()
  mpool.starmap_async(
    prediction_on_output_and_subgraph_with_event_proxy,
    items
  )
  # get result
  items = []
  while total:
    total -= 1
    entry = queue.get()
    resls = entry[1]
    if resls:
      items.append(entry)
    # del entry
  items.sort(key = lambda entry: entry[0])
  items = [entry[1] for entry in items]
  return items, extra