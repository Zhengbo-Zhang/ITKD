# ITKD
code for the paper "instance temporary knowledge distillation"

[Project page](https://www.zayx.me/ITKD.github.io/) | [arXiv]() (will be uploaded later)

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

## Weights

weights are putted into pretrained folder. 

## Training on cifar100

All training scripts reside within the file `train.sh`. Adapt this file to align with your specific environment, then proceed to select a configuration from the available options.  Note that the dataset will be automatically downloaded during the execution phase.

```bash
# modify the content inside and just run it
bash train.sh
```

## Inference on cifar100
The testing scripts mirror the training scripts in their structure and usage. Simply modify the content within the scripts to harmonize with your environment.
```bash
# modify the content inside and just run it
bash test.sh
```