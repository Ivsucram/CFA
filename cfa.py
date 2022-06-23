import argparse
import ssl
from abc import ABC

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import random
import warnings
from tqdm import tqdm
from typing import List, Tuple 
from os import mkdir, remove
from os.path import exists
import matplotlib.pyplot as plt

from avalanche.benchmarks.classic import SplitMNIST, SplitCIFAR10, SplitCIFAR100, SplitTinyImageNet, SplitOmniglot
from avalanche.benchmarks.generators import nc_benchmark
from avalanche.benchmarks.scenarios.new_classes.nc_scenario import NCExperience, NCScenario

paper_name = 'cfa'

parser = argparse.ArgumentParser(f'./{paper_name}.py',
                                 description='Class-Incremental Learning via Knowledge Amalgamation')
parser.add_argument('--dataset', type=str, default='mnist', help='Dataset to use', required=False,
                    choices=['usps', 'mnist', 'cifar10', 'cifar100', 'tiny10', 'tiny20', 'omniglot'])
parser.add_argument('--seed', type=int, default=None, metavar='N', help='Set a seed to compare runs')
parser.add_argument('--cuda', action='store_true', help='enable  CUDA')
parser.add_argument('--cuda_device', type=int, default=0, help='Cuda device identifier')

# Teacher models configuration
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=2 ** 6, help='Batch size for base model training',
                    choices=[2 ** 3, 2 ** 4, 2 ** 5, 2 ** 6, 2 ** 7, 2 ** 8])
parser.add_argument('--epochs', type=int, default=1000, help='Number of epochs per task (base learning)',
                    choices=[5, 10, 50, 100, 1000])
parser.add_argument('--force_base_retraining', type=bool, default=True, help='Force base model retraining')

# Amalgamation configuration
parser.add_argument('--cfl_lr', type=float, default=None, help='Common feature amalgamation learning rate')
parser.add_argument('--amalgamation_strategy', type=str, default='all_together', help='Amalgamation Strategy',
                    choices=['all_together', 'one_at_a_time'])
parser.add_argument('--amalgamation_epochs', type=int, default=100, help='Amalgamation epochs',
                    choices=[10, 100, 500, 1000])

# Memory strategy configuration
parser.add_argument('--memory_strategy', type=str, default='grow', help='Memory Strategy',
                    choices=['fixed', 'grow'])
parser.add_argument('--memory_budget', type=int, default=1000, help='Memory Budget',
                    choices=[100, 200, 500, 1000, 2000])
parser.add_argument('--alpha', type=float, default=0.5, help='Alpha')


# Helpers
def enum(**enums):
    return type('Enum', (), enums)


FIELD = enum(EXT_IDX='last_searched_external_idx',
             INT_IDX='last_searched_internal_idx',
             N_ELEM='n_elem',
             CLASSES_LIST='classes_list')


class AverageTracker():
    FIELD = enum(VALUE='value',
                 COUNT='count')

    def __init__(self):
        self.book = dict()

    def reset(self, key: str = None) -> None:
        item = self.book.get(key, {})
        if key is None:
            self.book.clear()
        else:
            item[self.FIELD.VALUE] = 0.
            item[self.FIELD.COUNT] = 0
            self.book[key] = item

    def update(self, key: str, val: torch.Tensor) -> None:
        item = self.book.get(key, None)
        if item is None:
            self.reset(key)
            self.update(key, val)
        else:
            item[self.FIELD.VALUE] += val
            item[self.FIELD.COUNT] += 1

    def get(self, key: str) -> float:
        item = self.book.get(key, None)
        assert item is not None
        return item[self.FIELD.VALUE] / float(item[self.FIELD.COUNT]) if float(item[self.FIELD.COUNT]) > 0. else 0.

    def count(self, key: str) -> float:
        item = self.book.get(key, None)
        assert item is not None
        return item[self.FIELD.COUNT]

# Code
class CommonFeatureLearningLoss(torch.nn.Module):
    def __init__(self, beta: float = 1.0):
        super(CommonFeatureLearningLoss, self).__init__()
        self.beta = beta

    def forward(self, hs: torch.Tensor, ht: torch.Tensor, ft_: torch.Tensor, ft: torch.Tensor) -> torch.Tensor:
        kl_loss = 0.0
        mse_loss = 0.0
        for ht_i in ht:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                kl_loss += torch.nn.functional.kl_div(torch.log_softmax(hs, dim=1), torch.softmax(ht_i, dim=1))
        for i in range(len(ft_)):
            mse_loss += torch.nn.functional.mse_loss(ft_[i], ft[i])

        return kl_loss + self.beta * mse_loss


class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=inplanes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(in_channels=planes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)
        self.downsample = None
        if stride > 1 or inplanes != planes:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels=inplanes,
                                                                  out_channels=planes,
                                                                  kernel_size=(1, 1),
                                                                  stride=stride,
                                                                  bias=False)
                                                  )

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x


class CommonFeatureBlocks(torch.nn.Module):
    def __init__(self, n_student_channels: int, n_teacher_channels: List[int], n_hidden_channel: int):
        super(CommonFeatureBlocks, self).__init__()

        ch_s = n_student_channels  # Readability
        ch_ts = n_teacher_channels  # Readability
        ch_h = n_hidden_channel  # Readability

        self.align_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.align_t.append(
                torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_t,
                                                    out_channels=2 * ch_h,
                                                    kernel_size=(1, 1),
                                                    bias=False),

                                    torch.nn.ReLU(inplace=True)
                                    )
            )

        self.align_s = torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_s,
                                                           out_channels=2 * ch_h,
                                                           kernel_size=(1, 1),
                                                           bias=False),

                                           torch.nn.ReLU(inplace=True)
                                           )

        self.extractor = torch.nn.Sequential(ResidualBlock(inplanes=2 * ch_h,
                                                           planes=ch_h,
                                                           stride=1),

                                             ResidualBlock(inplanes=ch_h,
                                                           planes=ch_h,
                                                           stride=1),

                                             ResidualBlock(inplanes=ch_h,
                                                           planes=ch_h,
                                                           stride=1)
                                             )

        self.dec_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.dec_t.append(
                torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_h,
                                                    out_channels=ch_t,
                                                    kernel_size=(3, 3),
                                                    stride=1,
                                                    padding=1,
                                                    bias=False),

                                    torch.nn.ReLU(inplace=True),

                                    torch.nn.Conv2d(in_channels=ch_t,
                                                    out_channels=ch_t,
                                                    kernel_size=(1, 1),
                                                    stride=1,
                                                    padding=0,
                                                    bias=False)
                                    )
            )

    def forward(self, fs: torch.Tensor, ft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aligned_t = [align(f) for align, f in zip(self.align_t, ft)]
        aligned_s = self.align_s(fs)
        ht = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)
        ft_ = [dec(h) for dec, h in zip(self.dec_t, ht)]
        return hs, ht, ft_

class MySimpleModel(torch.nn.Module):
    @property
    def features(self) -> torch.Tensor:
        assert self.handles is not None
        return self.resnet.layer4.output

    @property
    def feature_dimension(self) -> int:
        return self.resnet.layer4[-1].conv2.out_channels

    @property
    def soft_output(self) -> torch.Tensor:
        return self.fc.output

    @property
    def n_output(self) -> int:
        return self.fc[-1].out_features

    def __init__(self, n_output: int):
        super(MySimpleModel, self).__init__()
        self.handles = {}

        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet.fc_backup = self.resnet.fc
        self.resnet.fc = torch.nn.Sequential()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.resnet.fc_backup.in_features, 
                            self.resnet.fc_backup.in_features // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.resnet.fc_backup.in_features // 2, 
                            n_output)
        )

    def register_hooks(self) -> None:
        def forward_hook(module: torch.nn.modules.container.Sequential, _: tuple, output: torch.Tensor):
            module.output = output

        self.handles['conv_layer'] = self.resnet.layer4.register_forward_hook(forward_hook)
        self.handles['fc_layer'] = self.fc.register_forward_hook(forward_hook)

    def remove_hooks(self) -> None:
        assert self.handles is not None
        for k, v in self.handles.items():
            self.handles[k].remove()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        return torch.softmax(x, dim=1).argmax(1)


def save_load_best_model(model: MySimpleModel, experience: NCExperience, is_train=True, pbar=None) -> Tuple[MySimpleModel, float]:
    n_task = experience.current_experience
    path = f'./state'
    state_path = f'{path}/cfa_{args.dataset}_{n_task + 1}'

    if not exists(path):
        mkdir(path)
    if not exists(state_path):
        torch.save(model.state_dict(), state_path)
    assert exists(path)
    assert exists(state_path)

    with torch.no_grad():
        model.eval()
        corrects = 0
        total_task = 0

        if not is_train:
            model.load_state_dict(torch.load(state_path))
        
        data_loader = torch.utils.data.DataLoader(experience.dataset, batch_size=args.batch_size, shuffle=True)

        for _, data in enumerate(data_loader):
            x = data[0].to(device)
            y = (data[1] - min(experience.classes_in_this_experience)).to(device)
            total_task += len(x)
            corrects += int(sum(model.predict(x) == y))
        accuracy = (corrects / total_task) if total_task > 0 else 0

        if is_train:
            description = f'Train accuracy for base task {n_task + 1}: {accuracy * 100:.2f}% ({corrects}/{total_task})'

            if pbar is None:
                print(description)
            else:
                pbar.set_description_str(description)
            torch.save(model.state_dict(), state_path)
            model.train()
        else:
            model.load_state_dict(torch.load(state_path))

    return model, accuracy


def amalgamate(teachers: List[MySimpleModel], data_array: List = [], labels: List = [],
               train: NCScenario = None, test: NCScenario = None, epochs: int = 100) -> Tuple[MySimpleModel, List[int], List[int], np.ndarray, np.ndarray]:

    def memory_keys(all_data: NCScenario, teachers: List[MySimpleModel], labels: List[int], idx: int, previous_data_idxs: List[int] = None) -> List[int]:
        if args.memory_strategy == 'grow':
            n_elem = args.memory_budget // len(labels[idx])
            return get_mean_exemplar_keys(all_data, teachers[idx], labels[idx], n_elem, previous_data_idxs)
        elif args.memory_strategy == 'fixed':
            n_elem = args.memory_budget // sum(len(v) for v in labels)
            return get_mean_exemplar_keys(all_data, teachers[idx], labels[idx], n_elem, previous_data_idxs)

    def get_mean_exemplar_keys(all_data: NCScenario, teacher: MySimpleModel, labels: List[int], n_elem_per_class: int, previous_data_idxs: List[int] = None) -> List[int]:
        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        current_batch_size = 0
        n_samples = 0
        label_mean = {}
        
        with torch.no_grad():
            for label in labels:
                teacher(torch.rand((1,3,224,224)).to(device))
                label_mean[label] = torch.zeros_like(teacher.features)
                if previous_data_idxs is None:
                    for _, [x, y, _] in enumerate(all_data[0].dataset):
                        if x is None:
                            break
                    
                        if current_batch_size < args.batch_size and x is not None:
                            if y == label:
                                batch_sample[current_batch_size] = x
                                current_batch_size += 1
                        elif x is None and current_batch_size == 0:
                            break
                        elif current_batch_size > 0 or x is None:
                            batch_sample = batch_sample[:current_batch_size].to(device)
                            teacher(batch_sample)
                            label_mean[label] += sum(teacher.features, 1).unsqueeze(0)
                            n_samples += current_batch_size
                            current_batch_size = 0
                else:
                    for _, idx in enumerate(previous_data_idxs):
                        x, y, _ = all_data[0].dataset[idx]
                        if current_batch_size < args.batch_size:
                            if y == label:
                                batch_sample[current_batch_size] = x
                                current_batch_size += 1
                        elif current_batch_size == 0:
                            break
                        elif current_batch_size > 0:
                            batch_sample = batch_sample[:current_batch_size].to(device)
                            teacher(batch_sample)
                            label_mean[label] += sum(teacher.features, 1).unsqueeze(0)
                            n_samples += current_batch_size
                            current_batch_size = 0

                label_mean[label] /= n_samples
                n_samples = 0

        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        batch_idx = np.empty(args.batch_size, dtype=int)
        current_batch_index = 0
        label_idx_distance = {}
        label_idx = {}

        with torch.no_grad():
            for label in labels:
                
                label_idx_distance[label] = {}
                label_idx[label] = []
                for idx, [x, y, _] in enumerate(all_data[0].dataset):
                    if x is None:
                        break

                    if current_batch_index < args.batch_size and x is not None:
                        if y == label:
                            batch_sample[current_batch_index] = x
                            batch_idx[current_batch_index] = idx
                            label_idx_distance[label][idx] = np.inf
                            current_batch_index += 1
                    elif x is None and current_batch_index == 0:
                        break
                    elif current_batch_index > 0 or x is None:
                        batch_sample = batch_sample[:current_batch_index].to(device)
                        teacher(batch_sample)
                        for idx, elem in enumerate(batch_idx):
                            label_idx_distance[label][elem] = float(torch.dist(teacher.features[idx], label_mean[label], 2))
                        current_batch_index = 0 
            
                label_idx[label] = [idx for idx, _ in sorted(label_idx_distance[label].items(), key=lambda x: x[1])][:n_elem_per_class]

        return np.concatenate([v for k, v in label_idx.items()], 0)

    def get_conf_keys(all_data: NCScenario, teacher: MySimpleModel, labels: List[int], n_elem: int = args.memory_budget) -> List[int]:
        batch_sample = torch.empty((args.batch_size, 3, 224, 224))
        batch_idx = np.empty(args.batch_size, dtype=int)
        current_batch_size = 0
        conf = {}
        for label in labels:
            conf[label] = {}
            for idx, [x, y, _] in enumerate(all_data[0].dataset):
                if x is None:
                    break

                if current_batch_size < args.batch_size and x is not None:
                    if y == label:
                        batch_sample[current_batch_size] = x
                        batch_idx[current_batch_size] = idx
                        conf[label][idx] = 0
                        current_batch_size += 1
                elif x is None and current_batch_size == 0:
                    break
                elif current_batch_size > 0 or x is None:
                    batch_sample = batch_sample[:current_batch_size].to(device)
                    batch_idx = batch_idx[:current_batch_size]

                    soft_top_2 = torch.softmax(teacher(batch_sample), 1).topk(2)[0].tolist()
                    for i, j in enumerate(batch_idx):
                        conf[label][j] = soft_top_2[i][0] - soft_top_2[i][1]
                    
                    batch_sample = torch.empty((args.batch_size, 3, 224, 224))
                    batch_idx = np.empty(args.batch_size, dtype=int)
                    current_batch_size = 0

        idxs = []
        for label in conf.keys():
            idxs = idxs + list(dict(sorted(conf[label].items(),
                                           key=lambda x: x[1],
                                           reverse=True)).keys())[:(n_elem // len(labels))]

        return idxs[:n_elem]

    student = MySimpleModel(sum([teacher.n_output for teacher in teachers])).to(device)
    cfl_blk = CommonFeatureBlocks(student.feature_dimension,
                                  [teachers[0].feature_dimension, teachers[1].feature_dimension],
                                  int(sum([teacher.feature_dimension for teacher in teachers])/len(teachers))).to(device)

    cfl_lr = args.lr * 10 if args.cfl_lr is None else args.cfl_lr

    params_10x = [param for name, param in student.named_parameters() if 'fc' in name]
    params_1x = [param for name, param in student.named_parameters() if 'fc' not in name]
    optimizer = torch.optim.Adam([{'params': params_1x,            'lr': args.lr},
                                  {'params': params_10x,           'lr': args.lr * 10},
                                  {'params': cfl_blk.parameters(), 'lr': cfl_lr}])

    student.train()
    [teacher.register_hooks() for teacher in teachers]
    [teacher.eval() for teacher in teachers]
    student.register_hooks()
    average_tracker = AverageTracker()

    common_feature_learning_criterion = CommonFeatureLearningLoss().to(device)

    print('Adjusting replay memory - sorry the delay, this part of the code is not optimized')
    data_idx = []
    for idx, data in enumerate(data_array):
        data_idx.append(memory_keys(train, teachers, labels, idx, data))
        data_array[idx] = torch.stack([train[0].dataset[idx_][0] for idx_ in data_idx[idx]])
    print('Replay memory adjusted')

    all_data = torch.cat([data for data in data_array])
    p = torch.randperm(len(all_data))
    all_data = all_data[p]

    student.eval()
    with torch.no_grad():
        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        b_accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for _, [x, y, _] in enumerate(test[0].dataset):
            if int(y) not in labels_:
                continue

            label = torch.tensor(labels_.index(y)).to(device)
            sample = x.view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                if label in task_labels:
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            b_accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0

    student.train()
    with tqdm(unit='Epoch', total=epochs) as pbar:
        while pbar.n < epochs:
            average_tracker.reset()
            batch_sample = torch.empty((args.batch_size, 3, 224, 224))
            current_batch_index = 0
            for _, data in enumerate(all_data):
                if current_batch_index < args.batch_size and data is not None:
                    batch_sample[current_batch_index] = data
                    current_batch_index += 1
                elif data is None and current_batch_index == 0:
                    break
                elif current_batch_index > 0 or data is None:
                    batch_sample = batch_sample[:current_batch_index].to(device)
                    current_batch_index = 0

                    optimizer.zero_grad()
                    with torch.no_grad():
                        [teacher(batch_sample) for teacher in teachers]
                        teacher_soft = torch.cat(tuple([teacher.soft_output for teacher in teachers]), dim=1)

                    student(batch_sample)
                    batch_sample = torch.empty((args.batch_size, 3, 224, 224))
                    student_soft = student.soft_output

                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore')
                        cross_entropy_loss = torch.nn.functional.kl_div(torch.log_softmax(student_soft, dim=1),
                                                                        torch.softmax(teacher_soft, dim=1))

                    hs, ht, ft_ = cfl_blk(student.features, [teacher.features for teacher in teachers])
                    common_features_loss = 10 * common_feature_learning_criterion(hs, ht, ft_, [teacher.features for teacher in teachers])

                    loss = args.alpha * cross_entropy_loss + (1 - args.alpha) * common_features_loss
                    loss.backward()
                    optimizer.step()

                    average_tracker.update('loss', loss.item())
                    average_tracker.update('ce', cross_entropy_loss.item())
                    average_tracker.update('cf', common_features_loss.item())

                    description = f'Amalgamating ' \
                                  f'Loss={average_tracker.get("loss"):.2f} '\
                                  f'(cross entropy={average_tracker.get("ce"):.2f}, '\
                                  f'common features={average_tracker.get("cf"):.2f})'
                    pbar.set_description_str(description)
                    pbar.refresh()
            pbar.update()
            all_data = torch.cat([data for data in data_array])
            p = torch.randperm(len(all_data))
            all_data = all_data[p]
    [teacher.remove_hooks() for teacher in teachers]
    student.remove_hooks()

    student.eval()
    with torch.no_grad():
        corrects = np.zeros((len(teachers)), int)
        total_samples = np.zeros((len(teachers)), int)
        accuracy = np.zeros((len(teachers)))
        labels_ = [label for l in labels for label in l]
        for _, [x, y, _] in enumerate(test[0].dataset):
            if int(y) not in labels_:
                continue

            label = torch.tensor(labels_.index(y)).to(device)
            sample = x.view(1, 3, 224, 224).to(device)
            pred = student.predict(sample)
            for idx, task_labels in enumerate(labels):
                if label in task_labels:
                    corrects[idx] = corrects[idx] + int(pred == label)
                    total_samples[idx] = total_samples[idx] + 1

        for idx, _ in enumerate(teachers):
            accuracy[idx] = (corrects[idx] / total_samples[idx]) if total_samples[idx] > 0 else 0

    return student, [data for d in data_idx for data in d], accuracy, b_accuracy


def load_dataset(dataset: str, force_unique_task: bool = False) -> Tuple[NCScenario, NCScenario]:
    if dataset in ['mnist', 'usps', 'omniglot']:
        transforms = T.Compose([T.Grayscale(3), T.Resize((224, 224)), T.ToTensor()])
        
        if dataset == 'mnist':
            args.n_tasks = 1 if force_unique_task else 5
            data = SplitMNIST(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 10),
                              train_transform=transforms, eval_transform=transforms)
        elif dataset == 'usps':
            args.n_tasks = 1 if force_unique_task else 5
            usps_train = torchvision.datasets.USPS(root='./data', train=True, download=True)
            usps_test = torchvision.datasets.USPS(root='./data', train=False, download=True)

            data = nc_benchmark(usps_train, usps_test, n_experiences=args.n_tasks, seed=args.seed, task_labels=True,
                                fixed_class_order=range(0, 10), train_transform=transforms, eval_transform=transforms)
        elif dataset == 'omniglot':
            args.n_tasks = 1 if force_unique_task else 241
            data = SplitOmniglot(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 964),
                                 train_transform=transforms, eval_transform=transforms)

    elif dataset in ['cifar10', 'cifar100', 'tinyImageNet10', 'tiny10', 'tinyImageNet20', 'tiny20']:
        transforms = T.Compose([T.Resize((224, 224)), T.ToTensor()])

        if dataset == 'cifar10':
            args.n_tasks = 1 if force_unique_task else 5
            data = SplitCIFAR10(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 10),
                                train_transform=transforms, eval_transform=transforms)
        elif dataset == 'cifar100':
            args.n_tasks = 1 if force_unique_task else 10
            data = SplitCIFAR100(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 100),
                                 train_transform=transforms, eval_transform=transforms)
        elif dataset in ['tinyImageNet10', 'tiny10']:
            args.n_tasks = 1 if force_unique_task else 10
            data = SplitTinyImageNet(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 200),
                                     train_transform=transforms, eval_transform=transforms)
        elif dataset in ['tinyImageNet20', 'tiny20']:
            args.n_tasks = 1 if force_unique_task else 20
            data = SplitTinyImageNet(n_experiences=args.n_tasks, seed=args.seed, fixed_class_order=range(0, 200),
                                     train_transform=transforms, eval_transform=transforms)

    return data.train_stream, data.test_stream


def main(args):
    path = f'./state'
    state_path = f'{path}/{paper_name}_{args.dataset}'
    is_training_base_model = False

    # Prepare data
    train_stream, test_stream = load_dataset(args.dataset)
    

    # Training base model
    if args.force_base_retraining is not None and args.force_base_retraining:
        for i in range(args.n_tasks):
            if exists(f'{state_path}_{i + 1}'):
                remove(f'{state_path}_{i + 1}')
    if exists(path):
        for i in range(args.n_tasks):
            if not exists(f'{state_path}_{i + 1}'):
                is_training_base_model = True
                break
    else:
        is_training_base_model = True

    if is_training_base_model:
        print('Training base model')
        for experience in train_stream:
            model = MySimpleModel(len(experience.classes_in_this_experience)).to(device)
            criterion = torch.nn.CrossEntropyLoss().to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

            with tqdm(unit='Epoch', total=args.epochs) as pbar:
                train_loader = torch.utils.data.DataLoader(experience.dataset, batch_size=args.batch_size, shuffle=True)
                while pbar.n < args.epochs:
                    model.train()
                    for _, data in enumerate(train_loader):
                        x = data[0].to(device)
                        y = (data[1] - min(experience.classes_in_this_experience)).to(device)
                        optimizer.zero_grad()
                        output = model(x)
                        criterion(output, y).backward()
                        optimizer.step()
                    save_load_best_model(model, experience, pbar=pbar)
                    pbar.update() 

    base_models = []
    experiences = []
    print('Base model performance')
    for experience in test_stream:
        model = MySimpleModel(len(experience.classes_in_this_experience)).to(device)
        model, accuracy = save_load_best_model(model, experience, False)
        print(f'Test accuracy for base task {experience.current_experience + 1} {experience.classes_in_this_experience}: {accuracy * 100:.2f}')
        if experience.current_experience == 0:
            accuracy_0 = accuracy
        base_models.append(model.cpu())
        experiences.append(experience)

    args.n_tasks_original = args.n_tasks
    n_tasks = args.n_tasks
    accuracies = np.zeros((n_tasks, n_tasks))
    b_accuracies = np.zeros((n_tasks, n_tasks))
    accuracies[0, 0] = accuracy_0
    train_stream, test_stream = load_dataset(args.dataset, True)
    if args.amalgamation_strategy == 'one_at_a_time':
        amalgamated_model, data, accuracy, b_accuracy = amalgamate(teachers=[base_models[0].to(device), base_models[1].to(device)],
                                                                   data_array=[None, None],
                                                                   labels=[experiences[0].classes_seen_so_far, experiences[1].classes_in_this_experience],
                                                                   train=train_stream,
                                                                   test=test_stream,
                                                                   epochs=args.amalgamation_epochs)
        accuracies[0, 1] = accuracy[0]
        accuracies[1, 1] = accuracy[1]
        b_accuracies[0, 1] = b_accuracy[0]
        b_accuracies[1, 1] = b_accuracy[1]
        if n_tasks > 2:
            for i in range(1, n_tasks):
                amalgamated_model, data, accuracy, b_accuracy = amalgamate(teachers=[amalgamated_model.to(device), base_models[i].to(device)],
                                                                           data_array=[data, None],
                                                                           labels=[experiences[i-1].classes_seen_so_far, experiences[i].classes_in_this_experience],
                                                                           train=train_stream,
                                                                           test=test_stream,
                                                                           epochs=args.amalgamation_epochs)
                accuracies[i - 1, i] = accuracy[0]
                accuracies[i, i] = accuracy[1]
                b_accuracies[i - 1, i] = b_accuracy[0]
                b_accuracies[i, i] = b_accuracy[1]
    elif args.amalgamation_strategy == 'all_together':
        for n_task in range(2, n_tasks + 1, 1):
            _, _, accuracy, b_accuracy = amalgamate(teachers=[base_models[idx].to(device) for idx in range(n_task)],
                                                    data_array=[None] * n_task,
                                                    labels=[experiences[idx].classes_in_this_experience for idx in range(n_task)],
                                                    train=train_stream,
                                                    test=test_stream,
                                                    epochs=args.amalgamation_epochs)
            for i in range(len(accuracy)):
                accuracies[i, n_task - 1] = accuracy[i]
                b_accuracies[i, n_task - 1] = b_accuracy[i]

    print(f'accuracies \n {accuracies}')
    print(f'b_accuracies (random initialization) \n {b_accuracies}')

    acc = np.nanmean(np.where(accuracies != 0, accuracies, np.nan), 0)[-1]
    print(f'ACC: {acc * 100:.2f}%')

    bwt = 0
    for i in range(n_tasks - 1):
        j = {'one_at_a_time': i + 1, 'all_together': -1}
        bwt += accuracies[i, j[args.amalgamation_strategy]] - accuracies[i, i]
    bwt = bwt / (n_tasks - 1)
    print(f'BWT: {bwt * 100:.2f}%')

    fwt = 0
    for i in range(1, n_tasks):
        j = {'one_at_a_time': i, 'all_together': -1}
        fwt += accuracies[i - 1, j[args.amalgamation_strategy]] - b_accuracies[i, i]
    fwt = fwt / (n_tasks - 1)
    print(f'FWT: {fwt * 100:.2f}%')


if __name__ == '__main__':
    args = parser.parse_args()

    # Configure random seed and devices
    if args.seed is None:
        args.seed = random.randint(0, 10000)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_device}' if (torch.cuda.is_available() and args.cuda) else 'cpu')
    print(f'Device: {device}')

    main(args)
