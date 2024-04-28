import os
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
import argparse

from submit.model import TwoHeadModel
from dataloader import create_dataloader
from metric import classification_metrics

# Hyperparameter
EPOCH_NUM = 5
BATCH_SIZE = 64
lr = 1e-5
weight_decay = 1e-4


def save_weights(model, save_dir: str, epoch: int):
    weight_file = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save(model.state_dict(), weight_file)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, choices=["resnet18", "resnet34", "resnet50", "efficientnet", "densenet", "convnext", "vit"], default="resnet34")
    parser.add_argument("-s", "--img-size", type=int, default=512)
    parser.add_argument("-b", "--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("-f", "--full-data", action="store_true", default=False)
    parser.add_argument("-p", "--model-path", type=str)
    parser.add_argument("-t", "--task", type=str, choices=["task1", "task2"])
    args = parser.parse_args()

    torch.cuda.set_device(0)
    device = torch.device("cuda", 0)
    
    log_dir = f"output/{args.model}_{args.img_size}_{args.full_data}_finetune_{args.task}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    training_log = open(os.path.join(log_dir, "training.log"), "w")

    writer = SummaryWriter(log_dir)

    model = TwoHeadModel(model_name=args.model, num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    train_loader, val_loader = create_dataloader(args.batch_size, args.batch_size, img_size=args.img_size, use_full_data=args.full_data, task=args.task)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    best_score = 0
    for epoch in range(EPOCH_NUM):
        print(f"=== Epoch{epoch} ===")
        model.train()
        for i, (img, label) in enumerate(train_loader):
            img = img.to(device, torch.float32)
            label = label.to(device, torch.long)

            if args.task == "task1":
                out = model.predict_task1(img)
            elif args.task == "task2":
                out = model.predict_task2(img)
            loss = criterion(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f'Epoch [{epoch}/{EPOCH_NUM - 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}')
            training_log.write(f"Epoch [{epoch}/{EPOCH_NUM - 1}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item()}\n")
            SummaryWriter.add_scalar(writer, 'train_loss', loss.item(), epoch * len(train_loader) + i)

        model.eval()
        with torch.no_grad():
            y_true, y_pred = [], []
            for i, (img, label) in enumerate(val_loader):
                img = img.to(device, torch.float32)

                if args.task == "task1":
                    out = model.predict_task1(img)
                elif args.task == "task2":
                    out = model.predict_task2(img)
                predict = torch.argmax(out, dim=1).cpu()

                y_true.append(label)
                y_pred.append(predict)

            y_true = torch.cat(y_true).numpy()
            y_pred = torch.cat(y_pred).numpy()
      
            metric = classification_metrics(y_true, y_pred)
            score = (metric["qwk"] + metric["f1"] + metric["spe"]) / 3
        
        print(metric, score)
        training_log.write(f"{metric} {score}\n")
        SummaryWriter.add_scalar(writer, 'score', score, epoch * len(train_loader) + i)
        
        save_weights(model, log_dir, epoch)
        if score > best_score:
            best_score = score
            print(f"New best! Saving weights of epoch {epoch}...")
            training_log.write(f"New best! Saving weights of epoch {epoch}...\n")
            torch.save(model.state_dict(), f"{log_dir}/best.pth")

    print(f"Finish training. Best task1 score: {best_score}.")
    training_log.write(f"Finish training. Best task1 score: {best_score}.\n")
    
    training_log.close()


if __name__ == "__main__":
    main()
