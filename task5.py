import time
import cv2

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.dataset import TeamMateDataset
from torchvision.models import mobilenet_v3_small
from torchvision import transforms


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Create the datasets and dataloaders
    trainset = TeamMateDataset(n_images=50, train=True)
    testset = TeamMateDataset(n_images=10, train=False)
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)
    testloader = DataLoader(testset, batch_size=1, shuffle=False)

    # Define transform with random rotation
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=10)
    ])

    # Create the model and optimizer
    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    # Define loss functions
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_nll = torch.nn.NLLLoss()

    # Saving parameters
    best_train_loss_ce = 1e9
    best_train_loss_nll = 1e9

    # Loss lists
    train_losses_ce = []
    test_losses_ce = []
    train_losses_nll = []
    test_losses_nll = []

    # Epoch Loop
    for epoch in range(1, 100):

        # Start timer
        t = time.time_ns()

        # Train the model with CrossEntropyLoss
        model.train()
        train_loss_ce = 0

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = torch.stack([transform(image) for image in images])
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss_ce = criterion_ce(outputs, labels)
            loss_ce.backward()
            optimizer.step()
            train_loss_ce = train_loss_ce + loss_ce.item()

        model.eval()
        test_loss_ce = 0
        correct_ce = 0
        total = 0

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss_ce = criterion_ce(outputs, labels)
            test_loss_ce = test_loss_ce + loss_ce.item()
            _, predicted_ce = torch.max(outputs, 1)
            total += labels.size(0)
            correct_ce += (predicted_ce == labels).sum().item()

        train_losses_ce.append(train_loss_ce / len(trainloader))
        test_losses_ce.append(test_loss_ce / len(testloader))

        if train_loss_ce < best_train_loss_ce:
            best_train_loss_ce = train_loss_ce
            torch.save(model.state_dict(), 'lab8/best_model_ce.pth')

        torch.save(model.state_dict(), 'lab8/current_model_ce.pth')

        # Train the model with NLLLoss
        model.train()
        train_loss_nll = 0

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):
            images = torch.stack([transform(image) for image in images])
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)
            loss_nll = criterion_nll(log_probs, labels)
            loss_nll.backward()
            optimizer.step()
            train_loss_nll = train_loss_nll + loss_nll.item()

        model.eval()
        test_loss_nll = 0
        correct_nll = 0

        for images, labels in tqdm(testloader, total=len(testloader), leave=False):
            images = images.reshape(-1, 3, 64, 64).to(device)
            labels = labels.to(device)
            outputs = model(images)
            log_probs = F.log_softmax(outputs, dim=1)
            loss_nll = criterion_nll(log_probs, labels)
            test_loss_nll = test_loss_nll + loss_nll.item()
            _, predicted_nll = torch.max(outputs, 1)
            correct_nll += (predicted_nll == labels).sum().item()

        train_losses_nll.append(train_loss_nll / len(trainloader))
        test_losses_nll.append(test_loss_nll / len(testloader))

        if train_loss_nll < best_train_loss_nll:
            best_train_loss_nll = train_loss_nll
            torch.save(model.state_dict(), 'lab8/best_model_nll.pth')

        torch.save(model.state_dict(), 'lab8/current_model_nll.pth')

        # Print the epoch statistics for both loss functions
        print(f'Epoch: {epoch}, Train Loss CE: {train_loss_ce / len(trainloader):.4f}, Test Loss CE: {test_loss_ce / len(testloader):.4f}, Test Accuracy CE: {correct_ce / total:.4f}')
        print(f'Epoch: {epoch}, Train Loss NLL: {train_loss_nll / len(trainloader):.4f}, Test Loss NLL: {test_loss_nll / len(testloader):.4f}, Test Accuracy NLL: {correct_nll / total:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

        # Create the loss plot for CrossEntropyLoss
        plt.plot(train_losses_ce, label='Train Loss CE')
        plt.plot(test_losses_ce, label='Test Loss CE')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task5_loss_plot_ce.png')
        
        # Create the loss plot for NLLLoss
        plt.plot(train_losses_nll, label='Train Loss NLL')
        plt.plot(test_losses_nll, label='Test Loss NLL')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('lab8/task5_loss_plot_nll.png')

    print('Training complete')
    