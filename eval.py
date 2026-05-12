

import torch
import torch.nn as nn
from data.dataloader import get_dataloaders
from models.custom_model import CustomNet
from utils.visualization import visualize_samples
from train import train
import wandb


def validate(model, val_loader, criterion, device):
    
    model.eval()
    val_loss, correct, total = 0, 0, 0

    with torch.no_grad():

        for inputs, targets in val_loader:
          
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    val_accuracy = 100 * correct / total
    
    print(f"Validation Loss: {val_loss / len(val_loader)}, Validation Accuracy: {val_accuracy}")
   
    return val_accuracy, val_loss



if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_dataloaders("data/tiny-imagenet-200", batch_size=64)

    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    

    # In your __main__ block, before the training loop:
    wandb.init(project="FAIMDL-Lab3", config={
        "learning_rate": 0.001,
        "epochs": 10,
        "batch_size": 64,
        "model": "CustomNet"
    })

    
    best_acc = 0
    for epoch in range(1, 11):
        train_accuracy, train_loss = train(epoch=epoch, model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        val_accuracy, val_loss = validate(model=model, val_loader=val_loader, criterion=criterion, device=device)
        
        # Inside the training loop, after each epoch (after validate):
        wandb.log({
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch": epoch
        })

        
        if val_accuracy > best_acc:
            best_acc =  val_accuracy
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
    
    print(f"Best validation accuracy: {best_acc}")
    
    # At the very end:
    wandb.finish()
    
    visualize_samples(dataloader_train=train_loader)