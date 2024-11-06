import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision import transforms
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Define data augmentation transforms
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders with transforms
    trainset = TeamMateDataset(n_images=50, train=True, transform=transform)
    testset = TeamMateDataset(n_images=10, train=False, transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)  # Adjusted batch size
    testloader = DataLoader(testset, batch_size=2, shuffle=False)   # Adjusted batch size

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adjusted learning rate
    criterion1 = torch.nn.CrossEntropyLoss()
    criterion2 = torch.nn.NLLLoss()

    # Saving parameters
    best_train_loss = 1e9

    # Loss lists
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []

    # Epoch Loop
    for epoch in range(1, 100):

        # Start timer
        t = time.time_ns()

        # Train the model
        model.train()
        train_loss_ce = 0
        train_loss_nll = 0

        # Batch Loop
        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss_ce = criterion1(outputs, labels)
            loss_nll = criterion2(outputs, labels)

            # Backward pass
            loss_ce.backward()
            loss_nll.backward()

            # Update the parameters
            optimizer.step()

            # Accumulate the loss
            train_loss_ce += loss_ce.item()
            train_loss_nll += loss_nll.item()

        # Test the model
        model.eval()
        test_loss_ce = 0
        test_loss_nll = 0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []

        # Batch Loop
        for images, labels in tqdm(testloader, total=len(testloader), leave=False):

            # Move the data to the device (CPU or GPU)
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)

            # Compute the loss
            loss_ce = criterion1(outputs, labels)
            loss_nll = criterion2(outputs, labels)

            # Accumulate the loss
            test_loss_ce += loss_ce.item()
            test_loss_nll += loss_nll.item()

            # Get the predicted class from the maximum value in the output-list of class scores
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)

            # Accumulate the number of correct classifications
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

        # Print the epoch statistics
        print(f'Epoch: {epoch}, Train Loss CE: {train_loss_ce / len(trainloader):.4f}, Test Loss CE: {test_loss_ce / len(testloader):.4f}, Train Loss NLL: {train_loss_nll / len(trainloader):.4f}, Test Loss NLL: {test_loss_nll / len(testloader):.4f}, Test Accuracy: {correct / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Update loss lists
        train_losses_ce.append(train_loss_ce / len(trainloader))
        test_losses_ce.append(test_loss_ce / len(testloader))
        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))

        # Update the best model
        if train_loss_ce < best_train_loss:
            best_train_loss = train_loss_ce
            torch.save(model.state_dict(), 'lab8/best_model.pth')

        # Save the model
        torch.save(model.state_dict(), 'lab8/current_model.pth')

        # Create the loss plot
        plt.plot(train_losses_ce, label='Train Loss CE')
        plt.plot(test_losses_ce, label='Test Loss CE')
        plt.plot(train_losses_nll, label='Train Loss NLL')
        plt.plot(test_losses_nll, label='Test Loss NLL')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task8_loss_plot.png')

        # Create the confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10,7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.savefig('lab8/task8_confusion_matrix.png')
        plt.close()