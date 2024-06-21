# ITKD
code for the paper "instance temporary knowledge distillation"

## installation

```bash
# create env
conda create -n ITKD python=3.9
conda activate ITKD
# pytorch version
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
# other dependencies
pip install -r requirements.txt
```

## weights

weights are putted into pretrained folder. 

## training on cifar100

all training script is in train.sh. modify it to fit in your environment and choose one of the configuration. 