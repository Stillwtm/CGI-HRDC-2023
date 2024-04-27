# CGI-HRDC 2023

This repo is for the two tasks of [CGI-HRDC 2023 - Hypertensive Retinopathy Diagnosis Challenge](https://codalab.lisn.upsaclay.fr/competitions/11877#learn_the_details-terms_and_conditions)

## Result

### Task1
| Kappa      | F1         | Specificity | Average    | CPU Time    |
|------------|------------|-------------|------------|-------------|
| 0.3889 (3) | 0.6423 (7) | 0.8403 (4)  | 0.6238 (2) | 0.8611 (22) |

### Task2
| Kappa      | F1         | Specificity | Average    | CPU Time    |
|------------|------------|-------------|------------|-------------|
| 0.4029 (2) | 0.5576 (3) | 0.9389 (2)  | 0.6331 (1) | 0.5978 (19) |

## Installation

Start by clone the repo:

```bash
git clone https://github.com/Stillwtm/CGI-HRDC-2023.git
cd CGI-HRDC-2023
```
Create conda environment:

```bash
conda env create -f environment.yaml
conda activate CGI-HRDC-2023
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