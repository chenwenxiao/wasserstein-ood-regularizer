#!/bin/bash

cd /home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/ood_regularizer/experiment

dataset=('--in_dataset=cifar10 --out_dataset=cifar100'
'--in_dataset=cifar10 --out_dataset=svhn'
'--in_dataset=cifar10 --out_dataset=tinyimagenet'
'--in_dataset=cifar10 --out_dataset=celeba'
'--in_dataset=cifar10 --out_dataset=isun'
'--in_dataset=cifar10 --out_dataset=lsun'
'--in_dataset=cifar100 --out_dataset=cifar10'
'--in_dataset=cifar100 --out_dataset=svhn'
'--in_dataset=cifar100 --out_dataset=tinyimagenet'
'--in_dataset=cifar100 --out_dataset=celeba'
'--in_dataset=cifar100 --out_dataset=isun'
'--in_dataset=cifar100 --out_dataset=lsun'
'--in_dataset=svhn --out_dataset=cifar10'
'--in_dataset=svhn --out_dataset=cifar100'
'--in_dataset=svhn --out_dataset=tinyimagenet'
'--in_dataset=svhn --out_dataset=celeba'
'--in_dataset=svhn --out_dataset=isun'
'--in_dataset=svhn --out_dataset=lsun'
'--in_dataset=tinyimagenet --out_dataset=cifar10'
'--in_dataset=tinyimagenet --out_dataset=cifar100'
'--in_dataset=tinyimagenet --out_dataset=svhn'
'--in_dataset=tinyimagenet --out_dataset=celeba'
'--in_dataset=tinyimagenet --out_dataset=isun'
'--in_dataset=tinyimagenet --out_dataset=lsun'
'--in_dataset=celeba --out_dataset=cifar10'
'--in_dataset=celeba --out_dataset=cifar100'
'--in_dataset=celeba --out_dataset=svhn'
'--in_dataset=celeba --out_dataset=tinyimagenet'
'--in_dataset=celeba --out_dataset=isun'
'--in_dataset=celeba --out_dataset=lsun'
'--in_dataset=mnist --out_dataset=fashion_mnist'
'--in_dataset=mnist --out_dataset=kmnist'
'--in_dataset=mnist --out_dataset=omniglot'
'--in_dataset=mnist --out_dataset=not_mnist'
'--in_dataset=fashion_mnist --out_dataset=mnist'
'--in_dataset=fashion_mnist --out_dataset=kmnist'
'--in_dataset=fashion_mnist --out_dataset=omniglot'
'--in_dataset=fashion_mnist --out_dataset=not_mnist'
'--in_dataset=kmnist --out_dataset=mnist'
'--in_dataset=kmnist --out_dataset=fashion_mnist'
'--in_dataset=kmnist --out_dataset=omniglot'
'--in_dataset=kmnist --out_dataset=not_mnist'
'--in_dataset=omniglot --out_dataset=mnist'
'--in_dataset=omniglot --out_dataset=fashion_mnist'
'--in_dataset=omniglot --out_dataset=kmnist'
'--in_dataset=omniglot --out_dataset=not_mnist'
'--in_dataset=not_mnist --out_dataset=mnist'
'--in_dataset=not_mnist --out_dataset=fashion_mnist'
'--in_dataset=not_mnist --out_dataset=kmnist'
'--in_dataset=not_mnist --out_dataset=omniglot')


algorithm=('models/likelihood/vae.py'
)

mlrun --legacy -- python ${algorithm[$1]} ${dataset[$2]}

exit 0