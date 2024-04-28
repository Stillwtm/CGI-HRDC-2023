# CGI-HRDC 2023

This repo is for the two tasks of [CGI-HRDC 2023 - Hypertensive Retinopathy Diagnosis Challenge](https://codalab.lisn.upsaclay.fr/competitions/11877#learn_the_details-terms_and_conditions)

## Result

### Task1
| Kappa      | F1         | Specificity | Average    | CPU Time    |
|------------|------------|-------------|------------|-------------|
| 0.4306 (1) | 0.6772 (4) | 0.8333 (5)  | 0.6470 (1) | 0.6889 (25) |

### Task2
| Kappa      | F1         | Specificity | Average    | CPU Time    |
|------------|------------|-------------|------------|-------------|
| 0.4029 (6) | 0.5576 (14) | 0.9389 (3)  | 0.6331 (3) | 0.5978 (28) |

## Installation

Start by clone the repo:

```bash
git clone https://github.com/Stillwtm/CGI-HRDC-2023.git
cd CGI-HRDC-2023
```
Create conda environment:

```bash
conda env create -f environment.yaml
conda activate hrdc2023
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

Then you can run the first-stage training script with the following command, and change different models and hyperparameters:

```bash
python main.py [-m {resnet18,resnet34,resnet50,efficientnet,densenet,convnext,vit}] [-s IMG_SIZE] [-b BATCH_SIZE] [-f]
```

After training, you have the option to select a specific task, load the optimal checkpoint, and proceed with second-stage fine-tuning:

```bash
python finetune.py [-m {resnet18,resnet34,resnet50,efficientnet,densenet,convnext,vit}] [-s IMG_SIZE] [-b BATCH_SIZE] [-f] [-p MODEL_PATH] [-t {task1,task2}]
```

## Test

We utilize model ensemble in our implementation, so before proceeding, you need to move the trained models you want to use to `submit/models` folder, for example:

```bash
cp output/convnext_512_False_finetune_task2/epoch_4.pth submit/models/convnext.pth
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
python test.py -i ./dataset/image_task1/ -o ./test.csv
```