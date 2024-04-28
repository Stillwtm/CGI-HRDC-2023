import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image

from preprocess import preprocess_image

TASK = "task1"
MODELS_TO_ENSEMBLE = ["resnet18", "resnet34", "efficientnet", "densenet", "convnext"]


class model:
    def __init__(self):
        self.device = torch.device("cpu")

    def load(self, dir_path):
        """
        load the model and weights.
        dir_path is a string for internal use only - do not remove it.
        all other paths should only contain the file name, these paths must be
        concatenated with dir_path, for example: os.path.join(dir_path, filename).
        make sure these files are in the same directory as the model.py file.
        :param dir_path: path to the submission directory (for internal use only).
        :return:
        """
        self.models = []
        for model_name in MODELS_TO_ENSEMBLE:
            model_path = os.path.join(dir_path, "models", f"{model_name}.pth")
            model = TwoHeadModel(model_name=model_name, num_classes=2)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.models.append(model)
        
        self.ensemble = VoteEnsemble(self.models)

    def predict(self, input_image):
        """
        perform the prediction given an image.
        input_image is a ndarray read using cv2.imread(path_to_image, 1).
        note that the order of the three channels of the input_image read by cv2 is BGR.
        :param input_image: the input image to the model.
        :return: an int value indicating the class for the input image
        """
        image = preprocess_image(input_image)
        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5024, 0.5013, 0.5009], std=[0.0007, 0.0008, 0.0009])
        ])
        image = transform(Image.fromarray(image))
        image = image.to(self.device, torch.float32)

        pred_class = self.ensemble.predict(image[None], task=TASK)

        return pred_class
    
def create_model(model, pretrained=True, num_classes = 2):
    if model == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model == "resnet34":
        model = models.resnet34(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model == "resnet50":
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model == "efficientnet":
        model = models.efficientnet_v2_s(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    elif model == "densenet":
        model = models.densenet121(pretrained=pretrained)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
    elif model == "convnext":
        model = models.convnext_tiny(pretrained=pretrained)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif model == 'vit':
        model = models.vit_b_16(pretrained=True)
        model.heads.head = nn.Linear(model.heads.head.in_features, 2)
    elif model == 'swin':
        model = models.swin_t(pretrained=True)
    else:
        raise ValueError("Invalid model type")
    
    return model


class TwoHeadModel(nn.Module):
    def __init__(self, model_name, num_classes=2):
        super().__init__()

        backbone = create_model(model_name)

        if "resnet" in model_name:
            in_features = backbone.fc.in_features
            del backbone.fc
        elif model_name == "efficientnet":
            in_features = backbone.classifier[1].in_features
            del backbone.classifier[1]
        elif model_name == "densenet":
            in_features = backbone.classifier.in_features
            backbone.add_module("relu", nn.ReLU())
            backbone.add_module("avg_pool", nn.AdaptiveAvgPool2d((1, 1)))
            del backbone.classifier
        elif model_name == "convnext":
            in_features = backbone.classifier[2].in_features
            del backbone.classifier[2]
        else:
            raise ValueError("Invalid model type")
        hidden_size = 64
        
        self.head1 = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )
        self.head2 = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes),
        )

        self.backbone = nn.Sequential(
            *list(backbone.children()),
            nn.Flatten(1),
        )
    
    def forward(self, x1, x2):
        x1 = self.backbone(x1)
        x2 = self.backbone(x2)
        y1 = self.head1(x1)
        y2 = self.head2(x2)
        return y1, y2
    
    def predict_task1(self, x):
        x = self.backbone(x)
        y = self.head1(x)
        return y

    def predict_task2(self, x):
        x = self.backbone(x)
        y = self.head2(x)
        return y

class VoteEnsemble():
    def __init__(self, models):
        self.models = models

    @torch.no_grad()
    def predict(self, x, task):
        if task == "task1":
            outs = [model.predict_task1(x) for model in self.models]
        elif task == "task2":
            outs = [model.predict_task2(x) for model in self.models]
        preds = torch.cat([torch.argmax(out, dim=1) for out in outs])
        final_pred = torch.mode(preds)[0].item()
        return final_pred
