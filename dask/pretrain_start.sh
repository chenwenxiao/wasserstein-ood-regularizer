#!/bin/bash

cd /home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/ood_regularizer/experiment

dataset=('--in_dataset=fashion_mnist --out_dataset=mnist'
'--in_dataset=mnist --out_dataset=fashion_mnist'
'--in_dataset=kmnist --out_dataset=mnist'
'--in_dataset=omniglot --out_dataset=mnist'
'--in_dataset=not_mnist --out_dataset=mnist'
'--in_dataset=constant --out_dataset=svhn'
'--in_dataset=noise --out_dataset=svhn'
'--in_dataset=cifar10 --out_dataset=svhn'
'--in_dataset=cifar100 --out_dataset=svhn'
'--in_dataset=svhn --out_dataset=cifar10'
'--in_dataset=celeba --out_dataset=svhn'
'--in_dataset=tinyimagenet --out_dataset=svhn')

algorithm=('models/likelihood/vae.py --self_ood=True'
'models/likelihood/pixelcnn.py --self_ood=True'
'models/wgan/wasserstein.py'
'models/likelihood/vib.py'
'models/ensemble/pixelcnn.py'
'models/ensemble/vae.py'
'models/conditional/pixelcnn.py'
'models/conditional/vae.py'
'models/conditional/generalized_odin.py'
'models/batch_norm/pixelcnn.py'
'models/batch_norm/vae.py'
'models/conditional/pure_classifier.py'
'models/likelihood/glow.py --self_ood=True'
'models/ensemble/glow.py'
'models/conditional/glow.py'
'models/batch_norm/glow.py')

mlrun --legacy -- python ${algorithm[$1]} ${dataset[$2]}

exit 0