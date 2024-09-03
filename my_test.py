import numpy as np
import cluster
import argparse


parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument('--n', type=int, default=10, help='number of clusters for cluster (default: 10)')
parser.add_argument('--gam', type=int, default=300, 
                    help='gamma paramter for subspace clustering (default: 100)')
parser.add_argument('--tau', type=float, default=1.0,
                    help='tau paramter for subspace clustering (default: 1.0)')


def ensc(args, train_features, train_labels):
    """Perform Elastic Net Subspace Clustering.
    
    Options:
        gam (float): gamma parameter in EnSC
        tau (float): tau parameter in EnSC

    """
    res, plabels = cluster.ensc(args, train_features, train_labels)
    print('EnSC acc: {}, nmi: {}, ari: {}'.format(res["acc"], res["nmi"], res["ari"]))
    return res, plabels





args = parser.parse_args()
args.save = False

with open('./saved_models/selfsup_ResNet10MNIST+128_fashionmnist_epo150_bs1000_aug50+fashionmnist_lr0.1_mom0.9_wd0.0005_gam120.0_gam20.05_eps0.5/fashionmnist_features.npy', 'rb') as f:
    train_features = np.load(f)
with open('saved_models/selfsup_ResNet10MNIST+128_fashionmnist_epo150_bs1000_aug50+fashionmnist_lr0.1_mom0.9_wd0.0005_gam120.0_gam20.05_eps0.5/fashionmnist_labels.npy', 'rb') as f:
    train_labels = np.load(f)

ensc(args, train_features, train_labels)