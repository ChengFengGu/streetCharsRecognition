# %%
import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

import readData
from readData import train_loader


# %%
# define the model of cnn
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()

        # featuer fetcher
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.MaxPool2d(2),

        )

        #
        self.fc1 = nn.Linear(32 * 3 * 7, 11)
        self.fc2 = nn.Linear(32 * 3 * 7, 11)
        self.fc3 = nn.Linear(32 * 3 * 7, 11)
        self.fc4 = nn.Linear(32 * 3 * 7, 11)
        self.fc5 = nn.Linear(32 * 3 * 7, 11)
        self.fc6 = nn.Linear(32 * 3 * 7, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)

        return c1, c2, c3, c4, c5, c6


# %%
class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model2, self).__init__()
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv

        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

    def forward(self, img):
        feat = self.cnn(img)

        # print(feat.shape)

        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)

        return c1, c2, c3, c4, c5


# %%

# train

model = SVHN_Model2()

# loss function

criterion = nn.CrossEntropyLoss()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), 0.005)

loss_plot, c0_plot = [], []

# 10 iterations for epoch
for epoch in range(10):
    for data in train_loader:
        # c0, c1, c2, c3, c4, c5 = model(data[0])
        c0, c1, c2, c3, c4 = model(data[0])

        loss = criterion(c0, data[1][:, 0]) + \
               criterion(c1, data[1][:, 1]) + \
               criterion(c2, data[1][:, 2]) + \
               criterion(c3, data[1][:, 3]) + \
               criterion(c4, data[1][:, 4])
        # criterion(c5, data[1][:, 5])

        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_plot.append(loss.item())
        c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item() * 1.0 / c0.shape[0])
    print("epoch:{},loss_plot:{},c0_plot:{}".format(epoch, loss_plot, c0_plot))

# %%

x_axis_data = [x for x in range(len(train_loader) * 10)]
import matplotlib.pyplot as plt

plt.subplot(121)
plt.title('loss')
plt.plot(x_axis_data, loss_plot)
plt.subplot(122)
plt.plot(x_axis_data, c0_plot)
plt.show()
