# DIP

This repo is for the first task of [CGI-HRDC 2023 - Hypertensive Retinopathy Diagnosis Challenge](https://codalab.lisn.upsaclay.fr/competitions/11877#learn_the_details-terms_and_conditions)

## Introduction

Hypertensive retinopathy (HR) refers to retinal damage caused by high blood pressure. Elevated blood pressure initially causes changes in the retina, causing spasmodic constriction of the retinal arteries. If the blood pressure is controlled in time during this period, the damage to the blood vessels is reversible. However, the analysis of hypertensive retinopathy is limited by the manual inspection process of image by image, which is time-consuming and relies heavily on the experience of ophthalmologists. Therefore, an effective computer-aided system is essential to help ophthalmologists analyze the progression of disease.

In order to promote the application of machine learning and deep learning algorithms in computer-aided automatic clinical hypertensive retinopathy diagnosis, we organize the hypertensive retinopathy diagnosis challenge. With this dataset, various algorithms can test their performance and make a fair comparison with other algorithms.

Task 1 is hypertension classification. Given a fundus image of a patient's eye, the task is to confirm whether this patient suffers from hypertension. Category 0 represents no hypertension and category 1 represents hypertension. This is a two-class classification task.

The backbone of our model is ResNet34. Since the given dataset is quite small, we utilized several augmentation methods and introduced SimCLR, a contrastive learning method to pretrain our model. The testing result of our implementation is shown below:

| Kappa | F1 | Specificity | Average | CPU Time |
| --- | --- | --- | --- | --- |
| 0.3472 (6) | 0.6270 (8) | 0.7986 (2) | 0.5909 (5) | 0.1071 (8) |

## Installation

Start by clone the repo:

```bash
git clone https://github.com/etherwindy/DIP
cd DIP
```
Create conda environment:

```bash
conda env create -f environment.yaml
conda activate dip
```

## Training

First download the dataset of two tasks, and put then in the `dataset` folder. Rename images and labels as `image_task<task id>` and `label_task<task id>` respectively. The structure of the dataset should be like this:

```
dataset/
├── image_task1
│   ├── *.png
├── image_task2
│   ├── *.png
├── label_task1.csv
└── label_task2.csv
```

Then you can run the training script with the following command, and change different models and hyperparameters:

```bash
python main.py [-h] [-m {resnet18,resnet34,resnet50,efficientnet,densenet,convnext,vit}] [-s IMG_SIZE] [-b BATCH_SIZE]
```

## Test

We utilize model ensemble in our implementation, so before proceeding, you need to move the trained models you want to use to `submit/models` folder:

```bash
cp output/<model name>_<batch_size>/best<task id>.pth submit/models/<model name>.pth
```

For example:

```bash
cp output/densenet_512_twohead/best1.pth submit/models/densenet.pth
```

Then set the task you want to evaluate on and the model name you want to ensemble in `submit/model.py`. For example:

```python
TASK = "task1"
MODELS_TO_ENSEMBLE = ["resnet18", "resnet34", "efficientnet", "densenet", "convnext"]
```

Then you can test your trained models:

```bash
python test.py -i <image dir path> -o <output csv file>
```

For example:

```bash
python test.py -i ./dataset/image/ -o ./test.csv
```