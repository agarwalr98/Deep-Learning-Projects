import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import os
import matplotlib.pyplot as plt

if not os.path.exists('./dc_img'):
    os.mkdir('./dc_img')

if not os.path.exists('./plot'):
    os.mkdir('./plot')

def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x

num_epochs = 50
batch_size = 40
learning_rate = 1e-3

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder = nn.Sequential(
           
            nn.Conv2d(1, 8, 3, stride=1, padding=1), nn.BatchNorm2d(8), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True)
             ,nn.MaxPool2d(2, stride=2)
            )
        self.classification = nn.Sequential(
            nn.Linear(64*2*2,128),nn.BatchNorm1d(128)  , nn.ReLU(True),
            nn.Linear(128,64),nn.BatchNorm1d(64)  , nn.ReLU(True),
            nn.Linear(64,32),nn.BatchNorm1d(32)  , nn.ReLU(True),
            nn.Linear(32,16),nn.BatchNorm1d(16)  , nn.ReLU(True), 
            nn.Linear(16, 10),nn.BatchNorm1d(10))
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(64, 32, 2, stride=2), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 2, stride=2, padding=0), nn.BatchNorm2d(8), nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1), nn.BatchNorm2d(1),nn.Tanh()                               
            )

    def forward(self, x, layer):
        if layer == 0:
            x = self.encoder(x)
            x = self.decoder(x)
            return x
        if layer == 1:
            x = self.encoder(x)
            x=torch.flatten(x,start_dim=1)
            print(x.size(),"size of x is ::-")
            x = self.classification(x)
            # print(x.size())
            return  F.log_softmax(x,dim=1)
            


model = autoencoder()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


# General AutoEncoder

#for classification only
epoch_arr=[]
loss_arr=[]
for epoch in range(num_epochs):
    for data in dataloader:
        img, target = data
        img = Variable(img)
        # print(img.size())
        target = target
        # ===================forward=====================
        output = model(img, 0)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            # ===================log========================
    pic = to_img(output.cpu().data)
    save_image(pic, './dc_img/image_{}.png'.format(epoch))
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch+1, num_epochs, loss.data.item()))
    

        # epoch_arr.append(epoch+1)
        # loss_arr.append(loss)
for epoch in range(num_epochs):
    for data in dataloader: 
        # print('size mismatch')
        img, target = data
        # print(second.size())
        img = Variable(img)
        target = target
        predicted = model(img, 1)
        optimizer.zero_grad()
        # print(img.size())
        # nn.BatchNorm2d()
        loss = F.nll_loss(predicted, target)
        loss.backward()
        optimizer.step()
        
    print('epoch [{}/{}], loss:{:.4f}'
            .format(epoch + 1, num_epochs, loss.data.item()), 'classifier loss')

    epoch_arr.append(epoch+1)
    loss_arr.append(loss)

    
    
    


plt.plot(epoch_arr,loss_arr,'bo',  epoch_arr,loss_arr,'k')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./plot/graph.png')
plt.show()

torch.save(model.state_dict(), './conv_autoencoder.pth')

