#imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader, SubsetRandomSampler, Subset
from timm import create_model, list_models
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


#globals
data_directory = "C://Users//andre//Downloads//archive"
random_state = 99
class_numbers = 13
epochs = 1000
class_labels = ['Academic_Art', 'Art_Nouveau', 'Baroque', 'Expressionism', 'Japanese_Art', 'Neoclassicism', 'Primitivism', 'Realism', 'Renaissance', 'Rococo', 'Romanticism', 'Symbolism', 'Western_Medieval']

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
    ])

def prepare_data():
    train_data = datasets.ImageFolder(data_directory, transform=data_transforms)
    class_labels = train_data.targets

    train_index, test_index = train_test_split(range(len(train_data)), test_size=0.2, random_state=random_state, stratify=class_labels)

    test_class_labels = [class_labels[i] for i in test_index]

    test_index, validation_index = train_test_split(test_index, test_size=0.5, random_state=random_state, stratify=test_class_labels)

    train_loader = DataLoader(Subset(train_data, train_index), batch_size=32, shuffle=True)
    val_loader = DataLoader(Subset(train_data, validation_index), batch_size=32, shuffle=False)
    test_loader = DataLoader(Subset(train_data, test_index), batch_size=32, shuffle=False)
    return (train_loader, val_loader, test_loader)

def load_transformer(train_loader, val_loader, test_loader):
    model = create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=class_numbers, in_chans = 3)
    
    """
    #make model head a three layer neural net
    model.head.fc = nn.Sequential(
    nn.Linear(768, 512),
    nn.ReLU(),
    nn.Linear(512, 256),
    nn.ReLU(),
    nn.Linear(256, class_numbers)
)"""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    
    #freezing all layers but head and last swin layer
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.layers[-1].parameters():
        param.requires_grad = True
    
    for param in model.head.parameters():
        param.requires_grad = True
    
    """
    for name, param in model.named_parameters():
        if "head" not in name:
            param.requires_grad = False
    
    #continue finetuning
    state_dict = torch.load("swin_transformer_finetuned.pth", map_location=device)
    model.load_state_dict(state_dict)
    """

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_loss = float('inf')
    best_acc = 0
    
    train_acc = []
    train_loss = []
    val_acc = []
    val_loss = []

    # Confusion Matrix metrics
    test_true = []
    test_pred = []
    val_true = []
    val_pred = []
    

    running_train_correct = 0
    running_train_loss = 0
    running_val_correct = 0
    running_val_loss = 0

    model.eval()
    # run test once to get epoch zero performance
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = loss_function(outputs,labels)
            running_train_correct += (predicted == labels).sum().item()
            running_train_loss += loss.item()
        # Validation
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # total += labels.size(0)
            running_val_loss += loss_function(outputs,labels).item()
            running_val_correct += (predicted == labels).sum().item()
            
        
    running_train_correct = running_train_correct / len(train_loader.dataset)
    running_train_loss = running_train_loss / len(train_loader) 
    running_val_correct = running_val_correct / len(val_loader.dataset) 
    running_val_loss = running_val_loss / len(val_loader) 

    print(f"""Epoch: 0/{epochs}.\nTrain Accuracy: {running_train_correct:.5f} Train Loss: {running_train_loss:.5f}\nVal Accuracy: {running_val_correct:.5f} Val Loss: {running_val_loss:.5f}\n""")
    
    train_acc.append(running_train_correct)
    train_loss.append(running_train_loss)
    val_acc.append(running_val_correct)
    val_loss.append(running_val_loss)
    
    for e in range(epochs):
        running_train_correct = 0
        running_train_loss = 0
        running_val_correct = 0
        running_val_loss = 0
        
        # Training 
        model.train()
        for i, data in enumerate(train_loader):
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            loss = loss_function(outputs,labels)
            loss.backward()
            optimizer.step()
            
            running_train_correct += (predicted == labels).sum().item()
            running_train_loss += loss.item()

        model.eval()
        # Validation
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                # total += labels.size(0)
                running_val_loss += loss_function(outputs,labels).item()
                running_val_correct += (predicted == labels).sum().item()
                #print(predicted, labels)
                val_pred += predicted.tolist()
                val_true += labels.tolist()
            
        running_train_correct = running_train_correct / len(train_loader.dataset)
        running_train_loss = running_train_loss / len(train_loader) 
        running_val_correct = running_val_correct / len(val_loader.dataset) 
        running_val_loss = running_val_loss / len(val_loader) 
        

        train_acc.append(running_train_correct)
        train_loss.append(running_train_loss)
        val_acc.append(running_val_correct)
        val_loss.append(running_val_loss)
        
        print(f"""Epoch: {e+1}/{epochs}.\nTrain Accuracy: {running_train_correct:.5f} Train Loss: {running_train_loss:.5f}\nVal Accuracy: {running_val_correct:.5f} Val Loss: {running_val_loss:.5f}\n""")
        if running_val_loss < best_loss or running_val_correct >  best_acc:
            best_loss = running_val_loss
            best_acc = running_val_correct
            torch.save(model.state_dict(), "swin_transformer_finetuned.pth")
            print("pth saved!")

    test_acc = 0
    test_loss = 0
    running_test_correct = 0
    running_test_loss = 0
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # total += labels.size(0)
            running_test_loss += loss_function(outputs,labels).item()
            running_test_correct += (predicted == labels).sum().item()
            test_true += labels.tolist()
            test_pred += predicted.tolist()
            
    test_acc = running_test_correct / len(test_loader.dataset) 
    test_loss = running_test_loss / len(test_loader) 
    print(f"Test Accuracy: {test_acc:.5f}\nTest Loss: {test_loss:.5f}")
    return train_acc, train_loss, val_acc, val_loss, val_true, val_pred, test_true, test_pred

def plot(train_acc, train_loss, val_acc, val_loss):
    fig, ax = plt.subplots(1, 2, figsize=(11,5))
    plt.subplots_adjust(wspace=0.25)
    epochs = list(range(0, len(train_acc)))
    ax[0].plot(epochs, train_acc, marker='o', linestyle='-', label="Train Accuracy", color='blue')
    ax[0].plot(epochs, val_acc, marker='^', linestyle='-', label="Validation Accuracy", color='red')
    
    ax[1].plot(epochs, train_loss, marker='s', linestyle='-', label="Train Loss", color='blue')
    ax[1].plot(epochs, val_loss, marker='d', linestyle='-', label="Validation Loss", color='red')
    
    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')
    ax[0].set_ylabel('Accuracy')
    ax[1].set_ylabel('Loss')
    ax[1].set_yscale('log')
    ax[0].set_title(f'Training vs Test Accuracy over {len(epochs)-1} Epochs')
    ax[1].set_title(f'Training vs Test Loss over {len(epochs)-1} Epochs')
    ax[0].legend(loc="lower right")
    ax[1].legend(loc="upper right")
    ax[0].grid(True, which='both', linestyle='--', linewidth=0.5)
    ax[1].grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(f"swin_results_test.png")
    # plt.show()

def confusion(val_true, val_pred, test_true, test_pred):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))  # Create a figure with two subplots

    # Validation confusion matrix
    cm_val = confusion_matrix(np.array(val_true), np.array(val_pred), normalize='true')
    cm_val = np.around(cm_val, decimals=2)
    disp_val = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=np.array(class_labels))
    disp_val.plot(ax=axes[0], colorbar=False)
    axes[0].set_title("Validation Set")
    axes[0].set_xticklabels(class_labels, rotation=90)

    # Test confusion matrix
    cm_test = confusion_matrix(np.array(test_true), np.array(test_pred), normalize='true')
    cm_test = np.around(cm_test, decimals=2)
    disp_test = ConfusionMatrixDisplay(confusion_matrix=cm_test, display_labels=np.array(class_labels))
    disp_test.plot(ax=axes[1], colorbar=False)
    axes[1].set_title("Test Set")
    axes[1].set_xticklabels(class_labels, rotation=90)

    plt.tight_layout()
    plt.savefig("confusion_matrices.png", bbox_inches='tight')

def save_image(loader):
    imgs, labs = next(iter(loader))
    to_pil = transforms.ToPILImage()
    pil_image = to_pil(imgs[0])
    pil_image.save("test.jpg")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = prepare_data()
    train_acc, train_loss, val_acc, val_loss, val_true, val_pred, test_true, test_pred = load_transformer(train_loader, val_loader, test_loader)
    plot(train_acc, train_loss, val_acc, val_loss)
    confusion(val_true, val_pred, test_true, test_pred)