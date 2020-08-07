#!/bin/bash

cd /home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/ood_regularizer/experiment

dataset=('--in_dataset=mnist --out_dataset=fashion_mnist'
'--in_dataset=mnist --out_dataset=kmnist'
'--in_dataset=mnist --out_dataset=not_mnist'
'--in_dataset=mnist --out_dataset=omniglot'
'--in_dataset=mnist --out_dataset=celeba'
'--in_dataset=mnist --out_dataset=tinyimagenet'
'--in_dataset=mnist --out_dataset=svhn'
'--in_dataset=mnist --out_dataset=cifar10'
'--in_dataset=mnist --out_dataset=cifar100'
'--in_dataset=mnist --out_dataset=isun'
'--in_dataset=mnist --out_dataset=lsun'
'--in_dataset=mnist --out_dataset=constant'
'--in_dataset=mnist --out_dataset=noise'
'--in_dataset=fashion_mnist --out_dataset=mnist'
'--in_dataset=fashion_mnist --out_dataset=kmnist'
'--in_dataset=fashion_mnist --out_dataset=not_mnist'
'--in_dataset=fashion_mnist --out_dataset=omniglot'
'--in_dataset=fashion_mnist --out_dataset=celeba'
'--in_dataset=fashion_mnist --out_dataset=tinyimagenet'
'--in_dataset=fashion_mnist --out_dataset=svhn'
'--in_dataset=fashion_mnist --out_dataset=cifar10'
'--in_dataset=fashion_mnist --out_dataset=cifar100'
'--in_dataset=fashion_mnist --out_dataset=isun'
'--in_dataset=fashion_mnist --out_dataset=lsun'
'--in_dataset=fashion_mnist --out_dataset=constant'
'--in_dataset=fashion_mnist --out_dataset=noise'
'--in_dataset=kmnist --out_dataset=mnist'
'--in_dataset=kmnist --out_dataset=fashion_mnist'
'--in_dataset=kmnist --out_dataset=not_mnist'
'--in_dataset=kmnist --out_dataset=omniglot'
'--in_dataset=kmnist --out_dataset=celeba'
'--in_dataset=kmnist --out_dataset=tinyimagenet'
'--in_dataset=kmnist --out_dataset=svhn'
'--in_dataset=kmnist --out_dataset=cifar10'
'--in_dataset=kmnist --out_dataset=cifar100'
'--in_dataset=kmnist --out_dataset=isun'
'--in_dataset=kmnist --out_dataset=lsun'
'--in_dataset=kmnist --out_dataset=constant'
'--in_dataset=kmnist --out_dataset=noise'
'--in_dataset=not_mnist --out_dataset=mnist'
'--in_dataset=not_mnist --out_dataset=fashion_mnist'
'--in_dataset=not_mnist --out_dataset=kmnist'
'--in_dataset=not_mnist --out_dataset=omniglot'
'--in_dataset=not_mnist --out_dataset=celeba'
'--in_dataset=not_mnist --out_dataset=tinyimagenet'
'--in_dataset=not_mnist --out_dataset=svhn'
'--in_dataset=not_mnist --out_dataset=cifar10'
'--in_dataset=not_mnist --out_dataset=cifar100'
'--in_dataset=not_mnist --out_dataset=isun'
'--in_dataset=not_mnist --out_dataset=lsun'
'--in_dataset=not_mnist --out_dataset=constant'
'--in_dataset=not_mnist --out_dataset=noise'
'--in_dataset=omniglot --out_dataset=mnist'
'--in_dataset=omniglot --out_dataset=fashion_mnist'
'--in_dataset=omniglot --out_dataset=kmnist'
'--in_dataset=omniglot --out_dataset=not_mnist'
'--in_dataset=omniglot --out_dataset=celeba'
'--in_dataset=omniglot --out_dataset=tinyimagenet'
'--in_dataset=omniglot --out_dataset=svhn'
'--in_dataset=omniglot --out_dataset=cifar10'
'--in_dataset=omniglot --out_dataset=cifar100'
'--in_dataset=omniglot --out_dataset=isun'
'--in_dataset=omniglot --out_dataset=lsun'
'--in_dataset=omniglot --out_dataset=constant'
'--in_dataset=omniglot --out_dataset=noise'
'--in_dataset=celeba --out_dataset=mnist'
'--in_dataset=celeba --out_dataset=fashion_mnist'
'--in_dataset=celeba --out_dataset=kmnist'
'--in_dataset=celeba --out_dataset=not_mnist'
'--in_dataset=celeba --out_dataset=omniglot'
'--in_dataset=celeba --out_dataset=tinyimagenet'
'--in_dataset=celeba --out_dataset=svhn'
'--in_dataset=celeba --out_dataset=cifar10'
'--in_dataset=celeba --out_dataset=cifar100'
'--in_dataset=celeba --out_dataset=isun'
'--in_dataset=celeba --out_dataset=lsun'
'--in_dataset=celeba --out_dataset=constant'
'--in_dataset=celeba --out_dataset=noise'
'--in_dataset=tinyimagenet --out_dataset=mnist'
'--in_dataset=tinyimagenet --out_dataset=fashion_mnist'
'--in_dataset=tinyimagenet --out_dataset=kmnist'
'--in_dataset=tinyimagenet --out_dataset=not_mnist'
'--in_dataset=tinyimagenet --out_dataset=omniglot'
'--in_dataset=tinyimagenet --out_dataset=celeba'
'--in_dataset=tinyimagenet --out_dataset=svhn'
'--in_dataset=tinyimagenet --out_dataset=isun'
'--in_dataset=tinyimagenet --out_dataset=lsun'
'--in_dataset=tinyimagenet --out_dataset=constant'
'--in_dataset=tinyimagenet --out_dataset=noise'
'--in_dataset=svhn --out_dataset=mnist'
'--in_dataset=svhn --out_dataset=fashion_mnist'
'--in_dataset=svhn --out_dataset=kmnist'
'--in_dataset=svhn --out_dataset=not_mnist'
'--in_dataset=svhn --out_dataset=omniglot'
'--in_dataset=svhn --out_dataset=celeba'
'--in_dataset=svhn --out_dataset=tinyimagenet'
'--in_dataset=svhn --out_dataset=cifar10'
'--in_dataset=svhn --out_dataset=cifar100'
'--in_dataset=svhn --out_dataset=isun'
'--in_dataset=svhn --out_dataset=lsun'
'--in_dataset=svhn --out_dataset=constant'
'--in_dataset=svhn --out_dataset=noise'
'--in_dataset=cifar10 --out_dataset=mnist'
'--in_dataset=cifar10 --out_dataset=fashion_mnist'
'--in_dataset=cifar10 --out_dataset=kmnist'
'--in_dataset=cifar10 --out_dataset=not_mnist'
'--in_dataset=cifar10 --out_dataset=omniglot'
'--in_dataset=cifar10 --out_dataset=celeba'
'--in_dataset=cifar10 --out_dataset=svhn'
'--in_dataset=cifar10 --out_dataset=isun'
'--in_dataset=cifar10 --out_dataset=lsun'
'--in_dataset=cifar10 --out_dataset=constant'
'--in_dataset=cifar10 --out_dataset=noise'
'--in_dataset=cifar100 --out_dataset=mnist'
'--in_dataset=cifar100 --out_dataset=fashion_mnist'
'--in_dataset=cifar100 --out_dataset=kmnist'
'--in_dataset=cifar100 --out_dataset=not_mnist'
'--in_dataset=cifar100 --out_dataset=omniglot'
'--in_dataset=cifar100 --out_dataset=celeba'
'--in_dataset=cifar100 --out_dataset=svhn'
'--in_dataset=cifar100 --out_dataset=isun'
'--in_dataset=cifar100 --out_dataset=lsun'
'--in_dataset=cifar100 --out_dataset=constant'
'--in_dataset=cifar100 --out_dataset=noise'
'--in_dataset=constant --out_dataset=mnist'
'--in_dataset=constant --out_dataset=fashion_mnist'
'--in_dataset=constant --out_dataset=kmnist'
'--in_dataset=constant --out_dataset=not_mnist'
'--in_dataset=constant --out_dataset=omniglot'
'--in_dataset=constant --out_dataset=celeba'
'--in_dataset=constant --out_dataset=tinyimagenet'
'--in_dataset=constant --out_dataset=svhn'
'--in_dataset=constant --out_dataset=cifar10'
'--in_dataset=constant --out_dataset=cifar100'
'--in_dataset=constant --out_dataset=isun'
'--in_dataset=constant --out_dataset=lsun'
'--in_dataset=constant --out_dataset=noise'
'--in_dataset=noise --out_dataset=mnist'
'--in_dataset=noise --out_dataset=fashion_mnist'
'--in_dataset=noise --out_dataset=kmnist'
'--in_dataset=noise --out_dataset=not_mnist'
'--in_dataset=noise --out_dataset=omniglot'
'--in_dataset=noise --out_dataset=celeba'
'--in_dataset=noise --out_dataset=tinyimagenet'
'--in_dataset=noise --out_dataset=svhn'
'--in_dataset=noise --out_dataset=cifar10'
'--in_dataset=noise --out_dataset=cifar100'
'--in_dataset=noise --out_dataset=isun'
'--in_dataset=noise --out_dataset=lsun'
'--in_dataset=noise --out_dataset=constant')

algorithm=('models/likelihood/vae.py --self_ood=True --count_experiment=True'
'models/likelihood/vae.py --use_transductive=False --count_experiment=True'
'models/likelihood/vae.py --count_experiment=True'
'models/likelihood/vae.py --mixed_radio=mixed_ratio=0.1 --count_experiment=True'
'models/ensemble/vae.py --count_experiment=True'
'models/conditional/vae.p --count_experiment=True'
'models/batch_norm/vae.py --count_experiment=True'
'models/likelihood/pixelcnn.py --self_ood=True --count_experiment=True'
'models/likelihood/pixelcnn.py --use_transductive=False --count_experiment=True'
'models/likelihood/pixelcnn.py --count_experiment=True'
'models/likelihood/pixelcnn.py --mixed_radio=mixed_ratio=0.1 --count_experiment=True'
'models/ensemble/pixelcnn.py --count_experiment=True'
'models/batch_norm/pixelcnn.py --count_experiment=True'
'models/conditional/pixelcnn.py --count_experiment=True'
'models/wgan/wasserstein.py --count_experiment=True'
'models/increment/wasserstein.py --count_experiment=True'
'models/likelihood/vib.py --count_experiment=True'
'models/conditional/generalized_odin.py --count_experiment=True'
'models/conditional/pure_classifier.py --count_experiment=True'
'models/likelihood/glow.py --self_ood=True --count_experiment=True'
'models/likelihood/glow.py --use_transductive=False --count_experiment=True'
'models/likelihood/glow.py --count_experiment=True'
'models/likelihood/glow.py --mixed_radio=mixed_ratio=0.1 --count_experiment=True'
'models/increment/glow.py --count_experiment=True'
'models/ensemble/glow.py --count_experiment=True'
'models/conditional/glow.py --count_experiment=True'
'models/batch_norm/glow.py --count_experiment=True')

mlrun --legacy -- python ${algorithm[$1]} ${dataset[$2]}

exit 0