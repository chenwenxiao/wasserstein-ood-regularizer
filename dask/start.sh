#!/bin/bash

cd /home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/ood_regularizer/experiment

dataset=('--in_dataset=celeba --out_dataset=tinyimagenet'
'--in_dataset=celeba --out_dataset=svhn'
'--in_dataset=celeba --out_dataset=cifar10'
'--in_dataset=celeba --out_dataset=cifar100'
'--in_dataset=celeba --out_dataset=isun'
'--in_dataset=celeba --out_dataset=lsun'
'--in_dataset=celeba --out_dataset=constant'
'--in_dataset=celeba --out_dataset=noise'
'--in_dataset=tinyimagenet --out_dataset=celeba'
'--in_dataset=tinyimagenet --out_dataset=svhn'
'--in_dataset=tinyimagenet --out_dataset=isun'
'--in_dataset=tinyimagenet --out_dataset=lsun'
'--in_dataset=tinyimagenet --out_dataset=constant'
'--in_dataset=tinyimagenet --out_dataset=noise'
'--in_dataset=svhn --out_dataset=celeba'
'--in_dataset=svhn --out_dataset=tinyimagenet'
'--in_dataset=svhn --out_dataset=cifar10'
'--in_dataset=svhn --out_dataset=cifar100'
'--in_dataset=svhn --out_dataset=isun'
'--in_dataset=svhn --out_dataset=lsun'
'--in_dataset=svhn --out_dataset=constant'
'--in_dataset=svhn --out_dataset=noise'
'--in_dataset=cifar10 --out_dataset=celeba'
'--in_dataset=cifar10 --out_dataset=svhn'
'--in_dataset=cifar10 --out_dataset=isun'
'--in_dataset=cifar10 --out_dataset=lsun'
'--in_dataset=cifar10 --out_dataset=constant'
'--in_dataset=cifar10 --out_dataset=noise'
'--in_dataset=cifar100 --out_dataset=celeba'
'--in_dataset=cifar100 --out_dataset=svhn'
'--in_dataset=cifar100 --out_dataset=isun'
'--in_dataset=cifar100 --out_dataset=lsun'
'--in_dataset=cifar100 --out_dataset=constant'
'--in_dataset=cifar100 --out_dataset=noise'
'--in_dataset=constant --out_dataset=celeba'
'--in_dataset=constant --out_dataset=tinyimagenet'
'--in_dataset=constant --out_dataset=svhn'
'--in_dataset=constant --out_dataset=cifar10'
'--in_dataset=constant --out_dataset=cifar100'
'--in_dataset=constant --out_dataset=isun'
'--in_dataset=constant --out_dataset=lsun'
'--in_dataset=constant --out_dataset=noise'
'--in_dataset=noise --out_dataset=celeba'
'--in_dataset=noise --out_dataset=tinyimagenet'
'--in_dataset=noise --out_dataset=svhn'
'--in_dataset=noise --out_dataset=cifar10'
'--in_dataset=noise --out_dataset=cifar100'
'--in_dataset=noise --out_dataset=isun'
'--in_dataset=noise --out_dataset=lsun'
'--in_dataset=noise --out_dataset=constant'
'--in_dataset=mnist28 --out_dataset=fashion_mnist28'
'--in_dataset=mnist28 --out_dataset=kmnist28'
'--in_dataset=mnist28 --out_dataset=not_mnist28'
'--in_dataset=mnist28 --out_dataset=omniglot28'
'--in_dataset=mnist28 --out_dataset=constant28'
'--in_dataset=mnist28 --out_dataset=noise28'
'--in_dataset=fashion_mnist28 --out_dataset=mnist28'
'--in_dataset=fashion_mnist28 --out_dataset=kmnist28'
'--in_dataset=fashion_mnist28 --out_dataset=not_mnist28'
'--in_dataset=fashion_mnist28 --out_dataset=omniglot28'
'--in_dataset=fashion_mnist28 --out_dataset=constant28'
'--in_dataset=fashion_mnist28 --out_dataset=noise28'
'--in_dataset=kmnist28 --out_dataset=mnist28'
'--in_dataset=kmnist28 --out_dataset=fashion_mnist28'
'--in_dataset=kmnist28 --out_dataset=not_mnist28'
'--in_dataset=kmnist28 --out_dataset=omniglot28'
'--in_dataset=kmnist28 --out_dataset=constant28'
'--in_dataset=kmnist28 --out_dataset=noise28'
'--in_dataset=not_mnist28 --out_dataset=mnist28'
'--in_dataset=not_mnist28 --out_dataset=fashion_mnist28'
'--in_dataset=not_mnist28 --out_dataset=kmnist28'
'--in_dataset=not_mnist28 --out_dataset=omniglot28'
'--in_dataset=not_mnist28 --out_dataset=constant28'
'--in_dataset=not_mnist28 --out_dataset=noise28'
'--in_dataset=omniglot28 --out_dataset=mnist28'
'--in_dataset=omniglot28 --out_dataset=fashion_mnist28'
'--in_dataset=omniglot28 --out_dataset=kmnist28'
'--in_dataset=omniglot28 --out_dataset=not_mnist28'
'--in_dataset=omniglot28 --out_dataset=constant28'
'--in_dataset=omniglot28 --out_dataset=noise28'
'--in_dataset=constant28 --out_dataset=mnist28'
'--in_dataset=constant28 --out_dataset=fashion_mnist28'
'--in_dataset=constant28 --out_dataset=kmnist28'
'--in_dataset=constant28 --out_dataset=not_mnist28'
'--in_dataset=constant28 --out_dataset=omniglot28'
'--in_dataset=constant28 --out_dataset=noise28'
'--in_dataset=noise28 --out_dataset=mnist28'
'--in_dataset=noise28 --out_dataset=fashion_mnist28'
'--in_dataset=noise28 --out_dataset=kmnist28'
'--in_dataset=noise28 --out_dataset=not_mnist28'
'--in_dataset=noise28 --out_dataset=omniglot28'
'--in_dataset=noise28 --out_dataset=constant28'
'--in_dataset=cifar10 --out_dataset=tinyimagenet'
'--in_dataset=cifar10 --out_dataset=cifar100'
'--in_dataset=cifar100 --out_dataset=tinyimagenet'
'--in_dataset=cifar100 --out_dataset=cifar10'
'--in_dataset=tinyimagenet --out_dataset=cifar10'
'--in_dataset=tinyimagenet --out_dataset=cifar100')

algorithm=('models/likelihood/vae.py --self_ood=True --count_experiment=True'
'models/likelihood/vae.py --use_transductive=False --count_experiment=True'
'models/likelihood/vae.py --use_transductive=False --mixed_ratio=0.2 --count_experiment=True'
'models/likelihood/vae.py --count_experiment=True'
'models/likelihood/vae.py  --pretrain=True --count_experiment=True'
'models/likelihood/vae.py --mixed_ratio=0.2 --count_experiment=True'
'models/increment/vae.py --count_experiment=True'
'models/increment/vae.py --retrain_for_batch=True --count_experiment=True'
'models/ensemble/vae.py --count_experiment=True'
'models/conditional/vae.py --count_experiment=True'
'models/batch_norm/vae.py --count_experiment=True'
'models/likelihood/pixelcnn.py --self_ood=True --count_experiment=True'
'models/likelihood/pixelcnn.py --use_transductive=False --count_experiment=True'
'models/likelihood/pixelcnn.py --use_transductive=False --mixed_ratio=0.2 --count_experiment=True'
'models/likelihood/pixelcnn.py --count_experiment=True'
'models/likelihood/pixelcnn.py --pretrain=True --count_experiment=True'
'models/likelihood/pixelcnn.py --mixed_ratio=0.2 --count_experiment=True'
'models/increment/pixelcnn.py --count_experiment=True'
'models/increment/pixelcnn.py --retrain_for_batch=True --count_experiment=True'
'models/ensemble/pixelcnn.py --count_experiment=True'
'models/batch_norm/pixelcnn.py --count_experiment=True'
'models/conditional/pixelcnn.py --count_experiment=True'
'models/wgan/wasserstein.py --use_transductive=False --count_experiment=True'
'models/wgan/wasserstein.py --use_transductive=False --mixed_ratio=0.2 --count_experiment=True'
'models/wgan/wasserstein.py --count_experiment=True'
'models/wgan/wasserstein.py --pretrain=True --count_experiment=True'
'models/wgan/wasserstein.py --mixed_ratio=0.2 --count_experiment=True'
'models/increment/wasserstein.py --count_experiment=True'
'models/increment/wasserstein.py --retrain_for_batch=True --count_experiment=True'
'models/likelihood/vib.py --count_experiment=True'
'models/conditional/generalized_odin.py --count_experiment=True'
'models/conditional/pure_classifier.py --count_experiment=True'
'models/conditional/pure_classifier.py --use_transductive=False --count_experiment=True'
'models/likelihood/glow.py --self_ood=True --count_experiment=True'
'models/likelihood/glow.py --use_transductive=False --count_experiment=True'
'models/likelihood/glow.py --use_transductive=False --mixed_ratio=0.2 --count_experiment=True'
'models/likelihood/glow.py --count_experiment=True'
'models/likelihood/glow.py --pretrain=True --count_experiment=True '
'models/likelihood/glow.py --mixed_ratio=0.2 --count_experiment=True'
'models/increment/glow.py --count_experiment=True'
'models/increment/glow.py --retrain_for_batch=True --count_experiment=True'
'models/ensemble/glow.py --count_experiment=True'
'models/conditional/glow.py --count_experiment=True'
'models/batch_norm/glow.py --count_experiment=True'
'models/conditional/odin.py --count_experiment=True'
'models/ensemble/classifier.py --count_experiment=True'
'models/likelihood/vae_pretrain_diagram.py --pretrain=True'
'models/likelihood/vae_pretrain_diagram.py'
'models/increment/vae.py --mixed_train_skip=64 --count_experiment=True'
'models/increment/pixelcnn.py --mixed_train_skip=64 --count_experiment=True'
'models/increment/wasserstein.py --mixed_train_skip=64 --count_experiment=True'
'models/increment/glow.py --mixed_train_skip=64 --count_experiment=True'
'models/singleshot/vae.py --count_experiment=True'
)

mlrun --legacy -- python ${algorithm[$1]} ${dataset[$2]}

exit 0