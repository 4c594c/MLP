'''
Author: 4c594c (aLong)
Date: 2022-12-09 15:59:20
LastEditors: aLong
LastEditTime: 2022-12-09 17:57:28
FilePath: \MLP\train.py
Description: 

Copyright (c) 2022 by 4c594c (aLong), All Rights Reserved. 
'''

import torch
from torchvision import datasets, transforms
from model import *
from tqdm import tqdm


@torch.no_grad()
def init_weights(m):
    if isinstance(m, Linear):
        nn.init.xavier_normal_(m.weight)
        m.bias.fill_(0)


if __name__ == "__main__":

    # 超参数
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    in_features = 28 * 28
    hidden_features = 100
    out_features = 10
    
    epochs = 5
    batchsize = 128
    lr = 0.01

    # 数据集
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.1307], [0.3081])])
    data_train = datasets.MNIST(root=r"../datasets",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="../datasets",
                               transform=transform,
                               train=False,
                               download=True)
    #装载数据
    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=batchsize,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=batchsize,
                                                   shuffle=True)

    # 定义模型
    model = MLP(in_features, hidden_features, out_features,
                num_layers=4).to(device=device, dtype=torch.float32)
    model.apply(init_weights)    
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        accuracy = 0
        sum_loss = 0
        for data in tqdm(data_loader_train, desc=f"epoch:{epoch}/{epochs - 1}"):
            inputs, labels = data  # batchsize*1*28*28
            
            inputs = inputs.view(-1, 28*28).to(device=device, dtype=torch.float32)  # batchsize*28*28
            labels = labels.to(device=device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            preds = outputs.data.argmax(dim=-1)
            
            sum_loss += loss.data
            accuracy += (preds == labels.data).sum()
            
        accuracy = accuracy / len(data_train)
        print(accuracy)
        print(f"sum loss: {sum_loss},  " + f"train accuracy: {accuracy}")
        
    # test
    model.eval()
    test_accuracy = 0
    with torch.no_grad():
        for data in tqdm(data_loader_test, desc="test") :
            inputs, labels = data
            inputs = inputs.view(-1, 28*28).to(device=device, dtype=torch.float32)  # batchsize*28*28
            labels = labels.to(device=device)
            
            outputs = model(inputs)

            preds = outputs.data.argmax(dim=-1)
            test_accuracy += torch.sum(preds == labels.data)
            
        print(f"accuracy:{test_accuracy / len(data_test)}")

