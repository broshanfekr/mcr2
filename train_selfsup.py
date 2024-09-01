import argparse
import os

import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.cluster import SpectralClustering

import train_func as tf
from augmentloader import AugmentLoader
from loss import MaximalCodingRateReduction
import utils
import pickle
import cluster
from tqdm import tqdm

# from torch.cuda.amp import GradScaler, autocast
# scaler = GradScaler()


import warnings   
# Settings the warnings to be ignored 
warnings.filterwarnings('ignore') 


def load_var(load_path):
    file = open(load_path, 'rb')
    variable = pickle.load(file)
    file.close()
    return variable


def save_var(save_path, variable):
    file = open(save_path, 'wb')
    pickle.dump(variable, file)
    print("variable saved.")
    file.close()


def test_step(net, transforms, args):
    net.eval()
    trainset = tf.load_trainset(args.data, path=args.data_dir, 
                                transform=transforms, train=args.is_train_set)
    new_labels = trainset.targets
    trainloader = DataLoader(trainset, batch_size=200)
    
    features = []
    labels = []
    for step, (batch_imgs, batch_lbls) in enumerate(trainloader):

        # with autocast(enabled=True):
        #     batch_features = net(batch_imgs.cuda())

        batch_features = net(batch_imgs.cuda())
            
        features.append(batch_features.cpu().detach())
        labels.append(batch_lbls)
    
    features = torch.cat(features).detach().cpu().numpy()
    labels = torch.cat(labels).detach().cpu().numpy()
    num_classes = len(np.unique(labels))
    args.n = num_classes
    args.save = False

    ensc_res, ensc_plabels = cluster.ensc(args=args, train_features=features, train_labels=labels)
    kmeans_res, kmeans_plabels = cluster.kmeans(args=args, train_features=features, train_labels=labels)

    # sc = SpectralClustering(n_clusters=num_classes, assign_labels='discretize').fit(features)
    # z_based_label_list = sc.labels_
    
    # nmi = normalized_mutual_info_score(labels, z_based_label_list)
        
    return ensc_res, kmeans_res


parser = argparse.ArgumentParser(description='Unsupervised Learning')
parser.add_argument('--arch', type=str, default='resnet10mnist',
                    help='architecture for deep neural network (default: resnet18, clip)')
parser.add_argument('--fd', type=int, default=128,
                    help='dimension of feature dimension (default: 32)')
parser.add_argument('--data', type=str, default='fashionmnist', # 'cifar10', # ,
                    help='dataset for training (default: CIFAR10, sampled_cifar10)')
parser.add_argument('--epo', type=int, default=150,
                    help='number of epochs for training (default: 50)')
parser.add_argument('--bs', type=int, default=1000,
                    help='input batch size for training (default: 1000)')
parser.add_argument('--aug', type=int, default=50,
                    help='number of augmentations per mini-batch (default: 50)')
parser.add_argument('--lr', type=float, default=0.1,
                    help='learning rate (default: 0.001)')
parser.add_argument('--mom', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay (default: 5e-4)')
parser.add_argument('--gam1', type=float, default=20.0,
                    help='gamma1 for tuning empirical loss (default: 1.0)')
parser.add_argument('--gam2', type=float, default=0.05,
                    help='gamma2 for tuning empirical loss (default: 10)')
parser.add_argument('--eps', type=float, default=0.5,
                    help='eps squared (default: 2)')
parser.add_argument('--tail', type=str, default='',
                    help='extra information to add to folder name')

parser.add_argument('--transform', type=str, default='fashionmnist',
                    help='transform applied to trainset (default: default, sampled_cifar')

parser.add_argument('--sampler', type=str, default='random',
                    help='sampler used in augmentloader (default: random')
parser.add_argument('--pretrain_dir', type=str, default=None,
                    help='load pretrained checkpoint for assigning labels')
parser.add_argument('--pretrain_epo', type=int, default=115,
                    help='load pretrained epoch for assigning labels')
parser.add_argument('--save_dir', type=str, default='./saved_models/',
                    help='base directory for saving PyTorch model. (default: ./saved_models/)')
parser.add_argument('--data_dir', type=str, default='../data/cifar10',
                    help='base directory for saving PyTorch model. (default: ./data/)')

parser.add_argument('--gam', type=int, default=300, 
                    help='gamma paramter for subspace clustering (default: 100)')
parser.add_argument('--tau', type=float, default=1.0,
                    help='tau paramter for subspace clustering (default: 1.0)')
parser.add_argument("--verbose", type=bool, default=True)
parser.add_argument("--is_train_set", type=bool, default=False)
args = parser.parse_args()


if __name__ == "__main__":
    ## Pipelines Setup
    model_dir = os.path.join(args.save_dir,
                'selfsup_{}+{}_{}_epo{}_bs{}_aug{}+{}_lr{}_mom{}_wd{}_gam1{}_gam2{}_eps{}{}'.format(
                        args.arch, args.fd, args.data, args.epo, args.bs, args.aug, args.transform,
                        args.lr, args.mom, args.wd, args.gam1, args.gam2, args.eps, args.tail))
    utils.init_pipeline(model_dir)

    ## Prepare for Training
    if args.pretrain_dir is not None:
        net, _ = tf.load_checkpoint(args.pretrain_dir, args.pretrain_epo)
        utils.update_params(model_dir, args.pretrain_dir)  
    else:
        net = tf.load_architectures(args.arch, args.fd)
        
    transforms = tf.load_transforms(args.transform)
    
    if args.arch == "clip":
        transforms_for_orig_data = Compose([
                    Resize(224, interpolation=InterpolationMode.BICUBIC),
                    CenterCrop(224),
                    ToTensor(),
                    Normalize((0.48145466, 0.4578275, 0.40821073), 
                            (0.26862954, 0.26130258, 0.27577711)),
                ])
    else:
        transforms_for_orig_data = Compose([ToTensor()])

    trainset = tf.load_trainset(args.data, path=args.data_dir, train=args.is_train_set)
    trainloader = AugmentLoader(trainset,
                                transforms=transforms,
                                transforms_for_orig=transforms_for_orig_data,
                                sampler=args.sampler,
                                batch_size=args.bs,
                                num_aug=args.aug)

    criterion = MaximalCodingRateReduction(gam1=args.gam1, gam2=args.gam2, eps=args.eps)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.mom, weight_decay=args.wd)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [30, 60], gamma=0.1)
    utils.save_params(model_dir, vars(args))

    best_nmi = 0

    ## Training
    for epoch in range(args.epo):      
        net.train()
        
        if args.verbose:
            train_bar = tqdm(trainloader, desc="epoch {}".format(epoch))
        else:
            train_bar = trainloader
        
        for step, (batch_imgs, _, batch_idx) in enumerate(train_bar):

            # with autocast(enabled=True):
            #     batch_features = net(batch_imgs.cuda())

            batch_features = net(batch_imgs.cuda())
                
            loss, loss_empi, loss_theo = criterion(batch_features, batch_idx)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            utils.save_state(model_dir, epoch, step, loss.item(), *loss_empi, *loss_theo)
        
        if epoch % 5 == 0:
            ensc_res, kmeans_res = test_step(net, transforms_for_orig_data, args=args)    
            row_text = "ensc   -> epoch is: {}, loss is: {:.4f}, NMI is: {:.4f}, ARI is: {:.4f}, ACC is: {:.4f}"
            print(row_text.format(epoch, loss.item(), ensc_res["nmi"], ensc_res["ari"], ensc_res["acc"]))

            row_text = "kmeans -> epoch is: {}, loss is: {:.4f}, NMI is: {:.4f}, ARI is: {:.4f}, ACC is: {:.4f}"
            print(row_text.format(epoch, loss.item(), kmeans_res["nmi"], kmeans_res["ari"], kmeans_res["acc"]))

            print()

            if ensc_res["nmi"] > best_nmi:
                best_nmi = best_nmi
                utils.save_ckpt(model_dir, net, epoch="best")
            utils.save_ckpt(model_dir, net, epoch="last")
            
        scheduler.step()

    ensc_res, kmeans_res = test_step(net, transforms_for_orig_data, args=args)    
    row_text = "ensc   -> epoch is: {}, loss is: {:.4f}, NMI is: {:.4f}, ARI is: {:.4f}, ACC is: {:.4f}"
    print(row_text.format(epoch, loss.item(), ensc_res["nmi"], ensc_res["ari"], ensc_res["acc"]))

    row_text = "kmeans -> epoch is: {}, loss is: {:.4f}, NMI is: {:.4f}, ARI is: {:.4f}, ACC is: {:.4f}"
    print(row_text.format(epoch, loss.item(), kmeans_res["nmi"], kmeans_res["ari"], kmeans_res["acc"]))

    utils.save_ckpt(model_dir, net, epoch)
                
    print("The end")
