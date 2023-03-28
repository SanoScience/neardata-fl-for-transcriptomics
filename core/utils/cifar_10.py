import torch
import torch.nn as nn

DEVICE = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def train(net, dataloader, epochs_num):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs_num):
        for images, labels in dataloader:
            images, label = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, dataloader):
    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy
