import warnings
import csv
import torch
import time
import cv2
import os
import torch.utils.data as Data
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn
from torch import optim

device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings('ignore')
print(device)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1

(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=4):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(8192 * block.expansion, num_classes)
        self.Sigmoid_fun = nn.Sigmoid()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.Sigmoid_fun(out)
        return out

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def dataReader():
    Input = []
    label = []
    for i in range(1, 11):
        data_path = '../input/touhou/Dataset/Capture_' + str(i) + '/'
        label_path = '../input/touhou/Dataset/KeyCapture_' + str(i) + '.csv'
        for file in os.listdir(data_path):
            img = cv2.imread(data_path + file)
            img = cv2.resize(img, (122, 141))
            transf = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
                ]
            )

            img = transf(img)  # tensor数据格式是torch(C,H,W)
            Input.append(img)

        with open(label_path, 'rt') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = [list(map(int, row)) for row in reader]
        label.extend(rows)

    return Input, label


data_x, data_y = dataReader()
for i in range(len(data_y)-1, -1, -1):
    if sum(data_y[i]) == 0:
        del data_y[i]
        del data_x[i]

data_x = torch.stack(data_x, dim=0)
data_y = torch.FloatTensor(data_y)
print(data_x.shape)
print(data_y.shape)


ResNet = ResNet18()
ResNet.to(device)
epochs = 150
# 定义loss和optimizer
optimizer = optim.Adam(ResNet.parameters(), lr=5e-6, weight_decay=0.0)
criterion = nn.BCELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5, threshold=1e-3)

# 训练
batch_size = 64
torch_dataset= Data.TensorDataset(data_x, data_y)
loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=batch_size,  # 批大小
    shuffle=True,
)

for epoch in range(1, epochs

 + 1):
    for i, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)
        output = ResNet(x)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    TimeStr = time.asctime(time.localtime(time.time()))
    print('Epoch: {} --- {} --- '.format(epoch, TimeStr))
    print('Train Loss of the model: {}'.format(loss))
    print('Learning rate: {}'.format(optimizer.param_groups[0]['lr']))
    # 调整学习率
    scheduler.step(loss)

torch.save(ResNet.state_dict(), '/kaggle/working/ResNet.pt')