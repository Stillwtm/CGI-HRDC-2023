import os

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import argparse
from torchvision.models import resnet

from submit.model import create_model, TwoHeadModel
from dataloader import create_dataloader
from metric import classification_metrics

# TODO: hyperparameter
EPOCH_NUM = 50
BATCH_SIZE = 64
# VAL_BATCH_SIZE = 64
lr = 5e-5
weight_decay = 1e-5


def load_weights(model, save_dir: str, epoch: int, device):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    state_dict = torch.load(weight_file, map_location=device)
    state_dict = {k[len("module."):]: v for k, v in state_dict.items() if k.startswith("module.")}
    model.load_state_dict(state_dict)


def save_weights(model, save_dir: str, epoch: int):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), weight_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["resnet18", "resnet34", "resnet50", "efficientnet", "densenet", "convnext", "vit"], default="resnet34")
    parser.add_argument("-s", "--img-size", type=int, default=512)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    
    log_dir = f"output/{args.model}_{args.img_size}_twohead"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    training_log = open(os.path.join(log_dir, "training.log"), "w")

    writer = SummaryWriter(log_dir)

    # model = ResNet(model=args.model, pretrained=True)
    # backbone = torch.load(f"pretrainedModel/resnet{args.resnet}.pth")
    # backbone = resnet.resnet34(pretrained=True)
    # features = model.backbone.fc.in_features
    # backbone.fc = nn.Linear(features, 1)
    # model.backbone = backbone

    # backbone = create_model(args.model, pretrained=True, num_classes=2)
    model = TwoHeadModel(model_name=args.model)
    model.to(device)
    
    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.CrossEntropyLoss()
    train_loader1, val_loader1 = create_dataloader(args.batch_size, args.batch_size, img_size=args.img_size, use_full_data=False, task="task1")
    train_loader2, val_loader2 = create_dataloader(args.batch_size, args.batch_size, img_size=args.img_size, use_full_data=False, task="task2")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_score1 = best_score2 = 0
    for epoch in range(EPOCH_NUM):
        print(f"=== Epoch{epoch} ===")
        model.train()
        for i, ((img1, label1), (img2, label2)) in enumerate(zip(train_loader1, train_loader2)):
            img1 = img1.to(device, torch.float32)
            img2 = img2.to(device, torch.float32)
            label1 = label1.to(device, torch.long)
            label2 = label2.to(device, torch.long)

            out1, out2 = model(img1, img2)
            loss = criterion(out1, label1) + criterion(out2, label2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{EPOCH_NUM - 1}], Step [{i + 1}/{len(train_loader1)}], Loss: {loss.item()}')
            training_log.write(f"Epoch [{epoch}/{EPOCH_NUM - 1}], Step [{i + 1}/{len(train_loader1)}], Loss: {loss.item()}\n")
            SummaryWriter.add_scalar(writer, 'train_loss', loss.item(), epoch * len(train_loader1) + i)

        model.eval()
        with torch.no_grad():
            y_true1, y_pred1, y_true2, y_pred2 = [], [], [], []
            for i, ((img1, label1), (img2, label2)) in enumerate(zip(val_loader1, val_loader2)):
                img1 = img1.to(device, torch.float32)
                img2 = img2.to(device, torch.float32)

                out1, out2 = model(img1, img2)
                # predict = torch.where(output > 0.5, torch.ones_like(output), torch.zeros_like(output)).cpu()
                predict1 = torch.argmax(out1, dim=1).cpu()
                predict2 = torch.argmax(out2, dim=1).cpu()

                y_true1.append(label1)
                y_true2.append(label2)
                y_pred1.append(predict1)
                y_pred2.append(predict2)

            y_true1 = torch.cat(y_true1).numpy()
            y_pred1 = torch.cat(y_pred1).numpy()
            y_true2 = torch.cat(y_true2).numpy()
            y_pred2 = torch.cat(y_pred2).numpy()
            metric1 = classification_metrics(y_true1, y_pred1)
            metric2 = classification_metrics(y_true2, y_pred2)
            score1 = (metric1["qwk"] + metric1["f1"] + metric1["spe"]) / 3
            score2 = (metric2["qwk"] + metric2["f1"] + metric2["spe"]) / 3
        
        print(metric1, score1)
        print(metric2, score2)
        training_log.write(f"{metric1} {score1}\n")
        training_log.write(f"{metric2} {score2}\n")
        SummaryWriter.add_scalar(writer, 'score1', score1, epoch * len(train_loader1) + i)
        SummaryWriter.add_scalar(writer, 'score2', score2, epoch * len(train_loader2) + i)
        
        save_weights(model, log_dir, epoch)
        if score1 > best_score1:
            best_score1 = score1
            print(f"New best task1! Saving weights of epoch {epoch}...")
            training_log.write(f"New best task1! Saving weights of epoch {epoch}...\n")
            torch.save(model.state_dict(), f"{log_dir}/best1.pth")
        if score2 > best_score2:
            best_score2 = score2
            print(f"New best task2! Saving weights of epoch {epoch}...")
            training_log.write(f"New best task2! Saving weights of epoch {epoch}...\n")
            torch.save(model.state_dict(), f"{log_dir}/best2.pth")

    print(f"Finish training. Best task1 score: {best_score1}. Best task2 score: {best_score2}.")
    training_log.write(f"Finish training. Best task1 score: {best_score1}. Best task2 score: {best_score2}\n")
    
    training_log.close()


if __name__ == "__main__":
    main()
