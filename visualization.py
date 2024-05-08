# Author: Jiahao Li

# Ref: https://www.python-graph-gallery.com/dendrogram/

import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from graphviz import Digraph

def draw_graph(graph, title, filename = None, directory = None, format = 'pdf'):
  dot = Digraph(title)
  dot.format = format
  stack = []
  for kidid, block in enumerate(graph):
    blkid = block[0]
    if blkid is None:
      stack.append(kidid)
    else:
      break
  while stack:
    blkid = stack.pop()
    block = graph[blkid]
    blkid = block[0]
    pnode = block[1]
    kdict = block[2]
    if kdict:
      for kidid in kdict:
        stack.append(kidid)
    pname = pnode.name
    dot.node(pname, pname)
    if blkid is not None:
      block = graph[blkid]
      gnode = block[1]
      gname = gnode.name
      dot.node(gname, gname)
    dot.edge(gname, pname)
  dot.render(filename, directory)

def draw_dendrogram(graph, blktp, title, filename = None, directory = None, format = 'pdf'):
  hrloc = 0
  lines = []
  ptbuf = []
  stack = [[0, ptbuf, []]]
  while stack:
    entry = stack[-1]
    blkid = entry[0]
    block = graph[blkid]
    if 3 < len(entry):
      count, hravg = 0, 0
      vmax0, vmax1 = 0, 0
      sclst, clnks = [], []
      for (hrpos, vrpos, state, activ), scbuf, clink in zip(entry[3], entry[4], entry[5]):
        count += 1
        hravg += hrpos
        scale = [1]
        scbuf.append(scale)
        if state | activ:
          clnks.append(clink)
          vmax1 = max(vmax1, vrpos)
        else:
          sclst.extend(scbuf)
          vmax0 = max(vmax0, vrpos)
        lines.append([[None, hrpos], [None, vrpos], state, activ, scale])
      if clnks:
        activ = True
        rsmap = block[3]
        shift = block[4]
        hiter = block[6]
        chose = hiter[clnks, shift]
        vrinc = []
        for _, (sumoh, _) in rsmap.items():
          vrinc.append(sumoh)
        vrinc = torch.stack(vrinc).sum(dim = 0)
        vrinc = vrinc[chose] / vrinc.sum()
        if sclst and vmax0:
          vrsca = vmax1 / vmax0
          for scale in sclst:
            scale[0] = vrsca
        vmax1 = vmax1 + vrinc
      else:
        activ = False
        entry[2].extend(sclst)
        vmax1 = vmax1 + 1
      hravg /= count
      state = blkid in blktp
      ptlst = entry[1]
      ptlst.append([hravg, vmax1, state, activ])
      while count:
        entry = lines[-count]
        entry[0][0] = hravg
        entry[1][0] = vmax1
        count -= 1
      del stack[-1]
    elif block[2]:
      ptlst = []
      sclst = []
      lklst = []
      klist = [k for k in block[2].items()]
      klist.sort(key = lambda k:k[1][0][1])
      while klist:
        scbuf = []
        clink = klist.pop()
        kidid = clink[0]
        stack.append([kidid, ptlst, scbuf])
        sclst.append(scbuf)
        lklst.append(clink)
      entry.append(ptlst)
      entry.append(sclst)
      entry.append(lklst)
    else:
      ptlst = entry[1]
      scbuf = entry[2]
      scbuf.append([1])
      state = blkid in blktp
      ptlst.append([hrloc, 0, state, state])
      hrloc += 1
      del stack[-1]
  # draw lines
  plt.clf()
  lines.sort(key = lambda k:k[2] | k[3])
  for xdata, ydata, state, activ, scale in lines:
    ydata[0] *= scale[0]
    ydata[1] *= scale[0]
    color = 'r' if state or activ else 'b'
    hdata = [xdata[0], xdata[1], xdata[1]]
    vdata = [ydata[0], ydata[0], ydata[1]]
    plt.plot(hdata, vdata, color = color)
    marker = 'o' if state else '.' if activ else ','
    plt.plot([xdata[1]], [ydata[1]], color = color, marker = marker)
  color  = 'r' if ptbuf[0][2] or ptbuf[0][3] else 'b'
  marker = 'o' if ptbuf[0][2] else '.' if ptbuf[0][3] else ','
  plt.plot([ptbuf[0][0]], [ptbuf[0][1]], color = color, marker = marker)
  plt.title(title)
  # store or show
  if filename is None:
    filename = title
  path = os.path.join(directory, filename + '.' + format)
  plt.savefig(path, format = format, dpi = 300, pad_inches = 0)
  # plt.show()

def draw_heatmap(rdict, title, filename = None, directory = None, format = 'pdf'):
  odata = []
  pdata = []
  for _, (sumoh, sumpr) in rdict.items():
    odata.append(sumoh)
    pdata.append(sumpr)
  odata = torch.stack(odata).detach().numpy()
  pdata = torch.stack(pdata).detach().numpy()
  odata = pd.DataFrame(odata)
  pdata = pd.DataFrame(pdata)
  fig, axes = plt.subplots(nrows = 2, figsize = (8, 4))
  ax1, ax2 = axes
  if title is not None:
    ax1.set_title(title + "-Hard")
    ax2.set_title(title + "-Soft")
  sns.heatmap(odata, ax = ax1)
  sns.heatmap(pdata, ax = ax2)
  # store or show
  if filename is None:
    filename = title
  path = os.path.join(directory, filename + '.' + format)
  plt.savefig(path, format = format, dpi = 300, pad_inches = 0)
  plt.cla()
  plt.close("all")
  # plt.show()

def draw_train_metrics(results_phase1, results_phase2, title, filename = None, directory = None, format = 'pdf'):
  train_average_nllpr = []
  valid_average_nllpr = []
  test_average_nllpr = []
  valid_accuracy = []
  test_accuracy = []
  for entry in results_phase1:
    train_average_nllpr.append(entry[0].detach().cpu().numpy())
    valid_average_nllpr.append(entry[3].detach().cpu().numpy())
    test_average_nllpr.append(entry[7].detach().cpu().numpy())
    valid_accuracy.append(entry[6].detach().cpu().numpy() * 100)
    test_accuracy.append(entry[10].detach().cpu().numpy() * 100)
  for entry in results_phase2:
    train_average_nllpr.append(entry[0].detach().cpu().numpy())
    valid_average_nllpr.append(entry[2].detach().cpu().numpy())
    test_average_nllpr.append(entry[5].detach().cpu().numpy())
    valid_accuracy.append(entry[4].detach().cpu().numpy() * 100)
    test_accuracy.append(entry[7].detach().cpu().numpy() * 100)

  xlabel_font = {
    # 'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 10
  }
  ylabel_font = {
    # 'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 10
  }
  legend_font = {
    # 'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 10
  }
  _, ax1 = plt.subplots(figsize=(4, 3))
  ax1.grid(color = 'k', linestyle = '--', linewidth = 1,alpha = 0.1)
  ax1.tick_params(labelsize = 10)
  ax1.plot(
    range(len(valid_accuracy)), valid_accuracy, label = 'Valid Acc', color = 'g',
    linestyle = '-', marker = 's', markevery = 50, lw = 1, ms = 5
  )
  ax1.plot(
    range(len(test_accuracy)), test_accuracy, label = 'Test Acc', color = 'b',
    linestyle = '-', marker = 'o', markevery = 50, lw = 1, ms = 5
  )

  ax1.set_xlabel('Epoch', fontdict = xlabel_font)
  ax1.set_ylim([0, 100])
  ax1.set_ylabel('Accuracy', fontdict = ylabel_font)
  handles1, labels1 = ax1.get_legend_handles_labels()
  ax1.legend(handles1, labels1, loc = 'center left', frameon = False, prop = legend_font)

  ax2 = ax1.twinx()
  ax2.tick_params(labelsize = 10)
  ax2.plot(
    range(len(train_average_nllpr)), train_average_nllpr, label = 'Train Loss', color = 'r',
    linestyle = '-', marker = 'v', markevery = 50, lw = 1, ms = 5
  )
  ax2.plot(
    range(len(valid_average_nllpr)), valid_average_nllpr, label = 'Valid Loss', color = 'g',
    linestyle = '-', marker = 'D', markevery = 50, lw = 1, ms = 5
  )
  ax2.plot(
    range(len(test_average_nllpr)), test_average_nllpr, label = 'Test Loss', color = 'b',
    linestyle = '-', marker = 'p', markevery = 50, lw = 1, ms = 5
  )

  ax2.set_ylim([0, 6])
  ax2.set_ylabel('Loss', fontdict = ylabel_font)
  ax2.legend(loc = 'center right', frameon = False, prop = legend_font)
  handles2, labels2 = ax2.get_legend_handles_labels()
  handles1.extend(handles2)
  labels1.extend(labels2)

  if title:
    plt.title(title)
  # plt.xticks(range(0, 10, len(test_accuracy)), range(0, 10, len(test_accuracy)))
  plt.subplots_adjust(top = 0.96, bottom = 0.24, left = 0.18, right = 0.83, hspace = 0, wspace = 0)

  if filename is None:
    filename = title
  path = os.path.join(directory, filename + '.' + format)
  plt.savefig(path, format = format, dpi = 300, pad_inches = 0)
  plt.cla()
  plt.close("all")
  # plt.show()