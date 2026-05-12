

import torch
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



    

