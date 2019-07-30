import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')
if not os.path.exists('./plot'):
    os.mkdir('./plot')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 3,64, 64)
    return x


num_epochs = 50
batch_size = 40
learning_rate = 1e-3
# decay = 0.001

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# dataset = MNIST('./data', transform=img_transform)
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
data_dir = '/dataset/celebA/train'          # this path depends on your computer
dset = datasets.ImageFolder(data_dir, transform=img_transform)
train_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            #64*64s
            nn.Conv2d(3, 8, 3, stride=1, padding=1), nn.BatchNorm2d(8), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            #32*32
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            #16*16
            nn.Conv2d(16, 32, 3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),nn.MaxPool2d(2, stride=2),
            #8*8
            nn.Conv2d(32, 64, 3, stride=1, padding=0), nn.BatchNorm2d(64), nn.ReLU(True),
            #6*6
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, stride=2)
            #3*3
            )
        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(128, 64, 2, stride=2), nn.BatchNorm2d(64), nn.ReLU(True),
            #6*6
            nn.ConvTranspose2d(64, 32, 3, stride=1), nn.BatchNorm2d(32), nn.ReLU(True),
            #8*8
            nn.ConvTranspose2d(32, 16, 2, stride=2, padding=0), nn.BatchNorm2d(16), nn.ReLU(True),
            #16*16
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1), nn.BatchNorm2d(8), nn.ReLU(True),
            #32*32
            nn.ConvTranspose2d(8, 3, 4, stride=2, padding=1), nn.BatchNorm2d(3),nn.Tanh()                               
            )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# def binary_cross_entropy_forClassification(output,target):

#for autoencoder 
model = autoencoder().cuda()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate, weight_decay=1e-5)

#for classification only
epoch_arr=[]
loss_arr=[]

for epoch in range(num_epochs):
    for data in train_loader:
        img, target = data
        target = target.cuda()
        img = Variable(img).cuda()
        noise = Variable(img.data.new(img.size()).normal_(0, 1))
        img_with_noise = torch.add(img, noise)
        img_with_noise = Variable(img_with_noise).cuda()
        # ===================forward=====================
        output = model(img_with_noise)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


  #  ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))
    epoch_arr.append(epoch+1)
    loss_arr.append(loss)

    print(img[0].size(),"img_size")
    print(img_with_noise.size(),"img_with_noise")
    print(output.size(),"output")
    tensor=torch.cat((img,img_with_noise),0)
    tensor=torch.cat((tensor,output),0)
    print(tensor.size())
    pic = to_img(tensor.cpu().data)
    save_image(pic,'./dc_img/combined_image/image_{}.png'.format(epoch))

    # pic = to_img(output.cpu().data)
    # save_image(pic,'./dc_img/image_{}output.png'.format(epoch))
    # pic = to_img(img_with_noise.cpu().data)
    # save_image(pic,'./dc_img/image_{}noise.png'.format(epoch))
    # pic = to_img(img.cpu().data)
    # save_image(pic,'./dc_img/image_{}.png'.format(epoch))

plt.plot(epoch_arr,loss_arr,'bo', epoch_arr, loss_arr,  'k')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./plot/graph.png')
plt.show()

