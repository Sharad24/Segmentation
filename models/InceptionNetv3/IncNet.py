import os 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision 
from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader
import pandas as pd 
import numpy as np 
import pydicom
from PIL import Image
from torch.autograd import Variable

print(torch.__version__)

data_path = '../../'
train_images_path = os.path.join(data_path,'train/')
test_images_path = os.path.join(data_path, 'test/')
train_labels_path = os.path.join(data_path,'stage_2_train_labels.csv')

train_transform = transforms.Compose([
    transforms.Resize(299),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0,0),(1,1))
]
)

class RSNADataset(Dataset):

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.train_labels = pd.read_csv(csv_file, engine='python').drop(['x','y','width','height'],axis=1).T.reset_index(drop=True).T
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.train_labels)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.train_labels.iloc[idx, 0]) + '.dcm'
        image = pydicom.dcmread(img_name).pixel_array
#        image = image[:, :, np.newaxis]
#        print(image.shape)
        image = Image.fromarray(image)
#        print(image.size)
        label = self.train_labels.iloc[idx,1]

        if self.transform:
            image = self.transform(image)

        return image,label
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv = nn.Conv2d(1, 3, 1)
        self.inception = models.inception_v3(pretrained=False)
        self.inception.fc = nn.Linear(2048, 2)

    def forward(self, x):
        x = self.conv(x)
        x = self.inception(x)
#        x = (F.softmax(self.fc(x[0])), F.softmax(self.fc(x[1])))
        return x

tfmd_dataset = RSNADataset(csv_file=train_labels_path,root_dir=train_images_path,
        transform=train_transform)

dataloader = DataLoader(tfmd_dataset, batch_size=16, shuffle=True, num_workers=1)

import torchvision.models as models
net = Net()

criterion = nn.CrossEntropyLoss()

optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, nesterov=True)



if torch.cuda.is_available():
    net.cuda()
    nn.DataParallel(net)
    print("GPU")
epoch_loss_data = []

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

for epoch in range(10):  # loop over the dataset multiple times

    running_loss, running_loss_total = 0.0, 0.0
    for i, data in enumerate(dataloader, 0):
        # get the inputs
        print("Started")
        inputs, labels = data
        print("Loaded data")

        inputs, labels = Variable(inputs.to(device)), Variable(labels.to(device))

#        print(inputs.size())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        out = net(inputs)
#        print(out.size())
        out1, out2 = out

        print("Got output")

        loss1 = criterion(out1, labels)
        loss2 = criterion(out2, labels)

        loss = loss1 + loss2

        loss.backward()

        optimizer.step()

        print('Loss for the iteration: ', loss.item())

        # print statistics
        running_loss += loss.item()
        running_loss_total += loss.item()
        if i%100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            torch.save(net.state_dict(), 'checkpoint')
            torch.save(optimizer.state_dict(), 'optimizer_checkpoint')
    print('Loss for epoch:', running_loss_total)
    epoch_loss_data.append(running_loss_total)

torch.save(net.state_dict(), 'checkpoint')
torch.save(optimizer.state_dict(), 'optimizer_checkpoint')


