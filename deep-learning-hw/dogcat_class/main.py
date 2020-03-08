#coding:utf8
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.autograd import Variable
from data.dataset import DogCat
from tqdm import tqdm
import time


def train_model(model, dataloaders, dataset_sizes, lossfunc, optimizer, scheduler, num_epochs=10):
    start_time = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    train_acc = []
    valid_acc = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #
                model.train(True)  # Set model to training mode
            else:

                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                # if use_gpu:
                #     inputs = Variable(inputs.cuda())
                #     labels = Variable(labels.cuda())
                # else:
                #     inputs, labels = Variable(inputs), Variable(labels)
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                loss = lossfunc(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                running_loss += loss.data
                running_corrects += torch.sum(preds == labels.data).to(torch.float32)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':
                valid_acc.append(epoch_acc)
            else:
                train_acc.append(epoch_acc)

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
        # 这里使用了学习率调整策略
        scheduler.step(valid_acc[-1])
    elapsed_time = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        elapsed_time // 60, elapsed_time % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model, train_acc, valid_acc

TRAIN_DATA_ROOT = './dogs-vs-cats/train/'  # 训练集存放路径
TEST_DATA_ROOT = './dogs-vs-cats/test1'  # 测试集存放路径
BATCH_SIZE = 64
EPOCH = 10
LR = 0.001
MOMENTUM = 0.9
NUM_WORKER = 4

train_data = DogCat(TRAIN_DATA_ROOT,train=True)
val_data = DogCat(TRAIN_DATA_ROOT,train=False)
train_dataloader = Data.DataLoader(train_data,BATCH_SIZE,
                        shuffle=True,num_workers=NUM_WORKER)
val_dataloader = Data.DataLoader(val_data,BATCH_SIZE,
                        shuffle=False,num_workers=NUM_WORKER)
dataloaders = {'train':train_dataloader, 'val':val_dataloader}
#train_inputs, train_labels = train_dataloader
#val_inputs, val_labels = val_dataloader
datasizes = {'train':train_data.__len__(), 'val':val_data.__len__()}
print('data loaded!')
print(datasizes)

model = models.resnet34(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model = model.cuda()
lossfunc = nn.CrossEntropyLoss()
parameters = list(model.parameters())
optimizer = optim.SGD(parameters, lr=LR, momentum=MOMENTUM, nesterov=True)
# 使用ReduceLROnPlateau学习调度器，如果三个epoch准确率没有提升，则减少学习率
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,mode='max',patience=3,verbose=True)
model_trained,train_acc,valid_acc = train_model(model=model,
                           dataloaders=dataloaders,
                           dataset_sizes=datasizes,
                           lossfunc=lossfunc,
                           optimizer=optimizer,
                           scheduler=exp_lr_scheduler,
                           num_epochs=EPOCH)
torch.save(model_trained.state_dict(), './model_trained/resnet34_pretrained.pth')