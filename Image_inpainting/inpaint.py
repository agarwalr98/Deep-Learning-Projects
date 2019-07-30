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
        img = Variable(img).cuda()
        
        target = target.cuda()
        # 
         # idx = np.array([28,28,30,31,32,33,34,35])

        
        x = Variable(torch.randn(40, 3, 64, 64))
        x.fill_(1)
        update_values = Variable(torch.randn(1,24))
        update_values.fill_(0)
        
        for i in range(0,batch_size):
            for j in range(0,3):
                x[i,j,np.arange(20,44),20] = update_values
                x[i,j,np.arange(20,44),21] = update_values
                x[i,j,np.arange(20,44),22] = update_values
                x[i,j,np.arange(20,44),23] = update_values
                x[i,j,np.arange(20,44),24] = update_values
                x[i,j,np.arange(20,44),25] = update_values
                x[i,j,np.arange(20,44),26] = update_values
                x[i,j,np.arange(20,44),27] = update_values
                x[i,j,np.arange(20,44),28] = update_values
                x[i,j,np.arange(20,44),29] = update_values
                x[i,j,np.arange(20,44),30] = update_values
                x[i,j,np.arange(20,44),31] = update_values
                x[i,j,np.arange(20,44),32] = update_values
                x[i,j,np.arange(20,44),33] = update_values
                x[i,j,np.arange(20,44),34] = update_values
                x[i,j,np.arange(20,44),35] = update_values
                x[i,j,np.arange(20,44),36] = update_values
                x[i,j,np.arange(20,44),37] = update_values
                x[i,j,np.arange(20,44),38] = update_values
                x[i,j,np.arange(20,44),39] = update_values
                x[i,j,np.arange(20,44),40] = update_values
                x[i,j,np.arange(20,44),41] = update_values
                x[i,j,np.arange(20,44),42] = update_values
                x[i,j,np.arange(20,44),43] = update_values    

        x = x.cuda()

        matrix=torch.mul(img,x)
        x.fill_(0)
        # update_values = Variable(torch.randn(1,8))
        update_values.fill_(0)
        update_values = update_values.cuda()
        x = x.cuda()

        

        matrix = matrix + x
        # print(matrix[0,1,25,26])

        pic = to_img(matrix.cpu().data)
        save_image(pic,'./dc_img/imageOfinpaint_{}.png'.format(epoch))

        matrix = matrix.cuda()
        matrix = Variable(matrix).cuda()
        # ===================forward=====================
        output = model(matrix)
        loss = criterion(output, img)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # tensor=torch.cat(0,(img,matrix))
        # tensor=torch.cat(0,(tensor,output))
        # pic = to_img(tensor.cpu().data)
        # save_image(pic,'./dc_img/combined_image/image_{}.png'.format(epoch))
  #  ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch + 1, num_epochs, loss.data.item()))
    
    epoch_arr.append(epoch+1)
    loss_arr.append(loss)

    pic = to_img(output.cpu().data)
    save_image(pic,'./dc_img/image_{}.png'.format(epoch))
plt.plot(epoch_arr,loss_arr,'bo',  epoch_arr,loss_arr,'k')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('./plot/graph.png')
plt.show()
