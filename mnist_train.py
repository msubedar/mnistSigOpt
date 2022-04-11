import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import sigopt as so

DATA_DIR = "./data"

USE_CUDA = True
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


class TrainModel(object):
    def __init__(self, params):
        super(TrainModel, self).__init__()
        self.params = params
        self.create_model()

    def create_model(self):
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        mnist_trainset = MNIST(root=DATA_DIR, train=True, download=True, transform=transform)
        mnist_testset = MNIST(root=DATA_DIR, train=False, download=True, transform=transform)
        batch_size = self.params["batch_size"]

        self.train_loader = torch.utils.data.DataLoader(mnist_trainset, batch_size=batch_size, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(mnist_testset, batch_size=batch_size, shuffle=False)

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, 1)
                self.conv2 = nn.Conv2d(32, 64, 3, 1)
                self.dropout1 = nn.Dropout(0.25)
                self.dropout2 = nn.Dropout(0.5)
                self.fc1 = nn.Linear(9216, 128)
                self.fc2 = nn.Linear(128, 10)

            def forward(self, x):
                x = self.conv1(x)
                x = F.relu(x)
                x = self.conv2(x)
                x = F.relu(x)
                x = F.max_pool2d(x, 2)
                x = self.dropout1(x)
                x = torch.flatten(x, 1)
                x = self.fc1(x)
                x = F.relu(x)
                x = self.dropout2(x)
                output = self.fc2(x)
                return output

        self.net = Net()
        self.net.to(DEVICE)
        return self.net

    def train(self):
        print(self.params)
        num_epochs = self.params["num_epochs"]
        optimizer = torch.optim.Adam(self.net.parameters(), lr=10 ** self.params["log_learning_rate"])
        cost = torch.nn.CrossEntropyLoss()
        for epoch in range(num_epochs):
            self.net.train()
            for bind, (images, labels) in enumerate(self.train_loader):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                outputs = self.net(images)
                loss = cost(outputs, labels)
                loss.backward()
                optimizer.step()
            acc = self.evaluate_model()
            print("training epoch: {:04d},  Test Accuracy = {:03f}".format(epoch, acc))
        return self.net

    # evaluate model
    def evaluate_model(self):
        correct = 0
        total = 0
        self.net.eval()
        for images, labels in self.test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = self.net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
        return float(correct) / total
