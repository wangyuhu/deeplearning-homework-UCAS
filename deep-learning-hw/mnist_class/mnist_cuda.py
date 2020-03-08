import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)
EPOCH = 2
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD = True

train_data = torchvision.datasets.MNIST(root='./mnist', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD)
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(root='./mnist', train=False)
test_x = torch.unsqueeze(test_data.test_data, dim=1).type(torch.FloatTensor).cuda()/255.   # Tensor on GPU
test_y = test_data.test_labels.cuda()
print('mnist data loaded!')
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
                                  nn.ReLU(), nn.MaxPool2d(kernel_size=2)
                                  )
        self.conv2 = nn.Sequential(nn.Conv2d(6,16,5,1,2), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(16,120,5,1,2), nn.ReLU())
        self.out = nn.Linear(120*7*7, 10)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0),-1)
        output = self.out(x)
        return output

model = CNN()
model.cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()
best_test_accuarcy = 0.0
for epoch in range(EPOCH):

    for step, (x,y) in enumerate(train_loader):
        b_x = x.cuda()
        b_y = y.cuda()
        output = model(b_x)
        loss = loss_func(output, b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = model(test_x)
            pred_y = torch.max(test_output,1)[1].cuda().data.squeeze()
            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch:',epoch,'| train loss: %.4f' % loss.data.cpu().numpy(),'| test accuracy: %.4f' % accuracy)
    test_output = model(test_x)
    pred_y = torch.max(test_output, 1)[1].cuda().data.squeeze()
    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    if(best_test_accuarcy < accuracy):
        best_test_accuarcy = accuracy
        checkpoint = {'epoch': epoch,
                      'best_loss': loss.data.cpu().numpy(),
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, './model_trained/best_mnist.pt')
print('best_accyracy: %.4f' % best_test_accuarcy)