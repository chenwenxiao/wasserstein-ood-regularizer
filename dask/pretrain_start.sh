#!/bin/bash

cd /home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/ood_regularizer/experiment

pretrain_dataset=('--in_dataset=fashion_mnist --out_dataset=mnist'
'--in_dataset=mnist --out_dataset=fashion_mnist'
'--in_dataset=kmnist --out_dataset=mnist'
'--in_dataset=omniglot --out_dataset=mnist'
'--in_dataset=not_mnist --out_dataset=mnist'
'--in_dataset=cifar10 --out_dataset=svhn'
'--in_dataset=cifar100 --out_dataset=svhn'
'--in_dataset=svhn --out_dataset=cifar10'
'--in_dataset=celeba --out_dataset=svhn'
'--in_dataset=tinyimagenet --out_dataset=svhn')

mlrun --legacy -- python ${$1} ${dataset[$2]}

exit 0