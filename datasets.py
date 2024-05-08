# Author: Jiahao Li
import os
import os.path

from samplers import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import check_integrity, verify_str_arg, download_url, extract_archive

def find_classes(class_file):
  with open(class_file) as r:
    classes = list(map(lambda s: s.strip(), r.readlines()))
  classes.sort()
  class_to_idx = {classes[i]: i for i in range(len(classes))}
  return classes, class_to_idx

def make_dataset(root, base_folder, dirname, class_to_idx):
  images = []
  dir_path = os.path.join(root, base_folder, dirname)
  if dirname == 'train':
    for fname in sorted(os.listdir(dir_path)):
      cls_fpath = os.path.join(dir_path, fname)
      if os.path.isdir(cls_fpath):
        cls_imgs_path = os.path.join(cls_fpath, 'images')
        for imgname in sorted(os.listdir(cls_imgs_path)):
          path = os.path.join(cls_imgs_path, imgname)
          item = (path, class_to_idx[fname])
          images.append(item)
  else:
    imgs_path = os.path.join(dir_path, 'images')
    imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')
    with open(imgs_annotations) as r:
      data_info = map(lambda s: s.split('\t'), r.readlines())
    cls_map = {line_data[0]: line_data[1] for line_data in data_info}
    for imgname in sorted(os.listdir(imgs_path)):
      path = os.path.join(imgs_path, imgname)
      item = (path, class_to_idx[cls_map[imgname]])
      images.append(item)
  return images

class TinyImageNet(VisionDataset):
  """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.
    Args:
      root (string): Root directory of the dataset.
      split (string, optional): The dataset split, supports ``train``, or ``val``.
      transform (callable, optional): A function/transform that  takes in an PIL image
        and returns a transformed version. E.g, ``transforms.RandomCrop``
      target_transform (callable, optional): A function/transform that takes in the
        target and transforms it.
      download (bool, optional): If true, downloads the dataset from the internet and
        puts it in root directory. If dataset is already downloaded, it is not
        downloaded again.
  """
  base_folder = 'tiny-imagenet-200/'
  url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
  filename = 'tiny-imagenet-200.zip'
  md5 = '90528d7ca1a48142e341f4ef8d21d0de'
  
  def __init__(self,
    root, split = "train",
    transform = None,
    target_transform = None,
    download = False
  ):
    super(TinyImageNet, self).__init__(
      root, transform = transform,
      target_transform = target_transform
    )
    self.dataset_path = os.path.join(root, self.base_folder)
    self.loader = default_loader
    self.split = verify_str_arg(split, "split", ("train", "val",))
    if self._check_integrity():
      print('Files already downloaded and verified.')
    elif download:
      self._download()
    else:
      raise RuntimeError('Dataset not found. You can use download=True to download it.')
    if not os.path.isdir(self.dataset_path):
      print('Extracting...')
      extract_archive(os.path.join(root, self.filename))
    _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))
    self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
    img_path, target = self.data[index]
    image = self.loader(img_path)
    if self.transform is not None:
      image = self.transform(image)
    if self.target_transform is not None:
      target = self.target_transform(target)
    return image, target

  def _check_integrity(self):
    return check_integrity(os.path.join(self.root, self.filename), self.md5)

  def _download(self):
    print('Downloading...')
    download_url(self.url, root=self.root, filename=self.filename)
    print('Extracting...')
    extract_archive(os.path.join(self.root, self.filename))

def create_tiny_image_net_datasets(dataset_root):
  normalize = transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
  )
  # Download training data from open datasets.
  train_dataset = TinyImageNet(
    root = dataset_root,
    split = "train",
    download = True,
    transform = transforms.Compose([
      transforms.RandomCrop(64, padding = 4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
    ])
  )
  valid_dataset = TinyImageNet(
    root = dataset_root,
    split = "train",
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  test_dataset = TinyImageNet(
    root = dataset_root,
    split = "val",
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  return train_dataset, valid_dataset, test_dataset

def create_cifar10_datasets(dataset_root):
  normalize = transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
  )
  # Download training data from open datasets.
  train_dataset = datasets.CIFAR10(
    root = dataset_root,
    train = True,
    download = True,
    transform = transforms.Compose([
      transforms.RandomCrop(32, padding = 4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
    ])
  )
  valid_dataset = datasets.CIFAR10(
    root = dataset_root,
    train = True,
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  test_dataset = datasets.CIFAR10(
    root = dataset_root,
    train = False,
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  return train_dataset, valid_dataset, test_dataset

def create_cifar100_datasets(dataset_root):
  normalize = transforms.Normalize(
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225)
  )
  # Download training data from open datasets.
  train_dataset = datasets.CIFAR100(
    root = dataset_root,
    train = True,
    download = True,
    transform = transforms.Compose([
      transforms.RandomCrop(32, padding = 4),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      normalize
    ])
  )
  valid_dataset = datasets.CIFAR100(
    root = dataset_root,
    train = True,
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  test_dataset = datasets.CIFAR100(
    root = dataset_root,
    train = False,
    download = True,
    transform = transforms.Compose([
      transforms.ToTensor(),
      normalize
    ])
  )
  return train_dataset, valid_dataset, test_dataset

def create_data_loaders(
  train_dataset,
  valid_dataset,
  test_dataset,
  train_batch_size = 512,
  valid_batch_size = 100,
  test_batch_size = 100,
  train_num_workers = 0,
  valid_num_workers = 0,
  test_num_workers = 0,
  create_mode = 2
):
  if create_mode == 0:
    total_size = len(train_dataset)
    valid_size = total_size * 2 // 100
    train_size = total_size - valid_size
    train_loader = DataLoader(
      train_dataset,
      batch_size = train_batch_size,
      sampler = VanillaChunkSampler(train_size, 0, shuffle = True),
      num_workers = train_num_workers,
      pin_memory = True
    )
    valid_loader = DataLoader(
      valid_dataset,
      batch_size = valid_batch_size,
      sampler = VanillaChunkSampler(valid_size, train_size, shuffle = True),
      num_workers = valid_num_workers,
      pin_memory = True
    )
    test_loader = DataLoader(
      test_dataset,
      batch_size = test_batch_size,
      num_workers = test_num_workers,
      pin_memory = True
    )
  elif create_mode == 1:
    total_size = len(train_dataset)
    valid_size = total_size * 2 // 100
    train_size = total_size - valid_size
    train_sampler = MasterChunkSampler([train_size, total_size])
    valid_sampler = train_sampler.get_slave_sampler([1])
    train_loader = DataLoader(
      train_dataset,
      batch_size = train_batch_size,
      sampler = train_sampler,
      num_workers = train_num_workers,
      pin_memory = True
    )
    valid_loader = DataLoader(
      valid_dataset,
      batch_size = valid_batch_size,
      sampler = valid_sampler,
      num_workers = valid_num_workers,
      pin_memory = True
    )
    test_loader = DataLoader(
      test_dataset,
      batch_size = test_batch_size,
      num_workers = test_num_workers,
      pin_memory = True
    )
  elif create_mode == 2:
    total_size = len(train_dataset)
    valid_size = total_size * 2 // 100
    train_loader = DataLoader(
      train_dataset,
      batch_size = train_batch_size,
      shuffle = True,
      num_workers = train_num_workers,
      pin_memory = True
    )
    valid_loader = DataLoader(
      valid_dataset,
      batch_size = valid_batch_size,
      sampler = MasterChunkSampler([valid_size, total_size]),
      num_workers = valid_num_workers,
      pin_memory = True
    )
    test_loader = DataLoader(
      test_dataset,
      batch_size = test_batch_size,
      num_workers = test_num_workers,
      pin_memory = True
    )
  else:
    raise ValueError("Unavailable create mode!")
  return train_loader, valid_loader, test_loader