

import torch
import torch.nn as nn
from data.dataloader import get_dataloaders
from models.custom_model import CustomNet
from utils.visualization import visualize_samples


def train(epoch, model, train_loader, criterion, optimizer, device):
    
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, targets in train_loader:
        
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        
        loss = criterion(outputs, targets)
        loss.backward()

        optimizer.step()
        running_loss += loss.item()
        
        _, predicted = outputs.max(1)    
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    print(f"Train Epoch: {epoch}\n Loss: {running_loss/len(train_loader)}\n Accuracy: {100*correct/total}")


# Validation loop function
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
    
    Accuracy = 100 * correct / total
    
    print(f"Validation Loss: {val_loss / len(val_loader)}\n Validation Accuracy: {Accuracy}")
   
    return Accuracy



if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_dataloaders("data/tiny-imagenet-200", batch_size=64)

    model = CustomNet(num_classes=200).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    
    best_acc = 0
    for epoch in range(1, 11):
        train(epoch=epoch, model=model, train_loader=train_loader, criterion=criterion, optimizer=optimizer, device=device)
        val_acc = validate(model=model, val_loader=val_loader, criterion=criterion, device=device)
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "checkpoints/best_model.pth")
    
    print(f"Best validation accuracy: {best_acc}")
    
    visualize_samples(dataloader_train=train_loader)
    

