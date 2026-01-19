# -*- coding: utf-8 -*-
"""
Final fixed version - CPU only, no inplace operations
"""
import csv
import numpy as np
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from timeit import default_timer
import random
from Loss_function import LpLoss 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Training on {device}")

def readfile(path):
    with open(path, 'r') as f:
        return [list(map(float, row)) for row in csv.reader(f)]

def openfile(filename, dataset):
    K = readfile(f'./{dataset}/{filename}.csv')
    return np.nan_to_num(np.array(K))

def save_data(filename, data):
    with open(filename, 'w') as f:
        np.savetxt(f, data, delimiter=',')

def save_pig_loss(loss_train_all, loss_test_all, imgpath):
    fig = plt.figure(num=1, figsize=(12, 8), dpi=100)
    plt.plot(range(len(loss_train_all)), loss_train_all, label='train loss')
    plt.plot(range(len(loss_test_all)), loss_test_all, label='test loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss value")
    plt.legend()
    plt.savefig(imgpath)
    plt.close()


def load_data(ntrain, ntest, S): 
    train_C = openfile('train_C', 'data_geo5_r128')
    train_x = openfile('train_x_data', 'data_geo5_r128')
    train_y = openfile('train_y_data', 'data_geo5_r128')
    train_U = openfile('train_U', 'data_geo5_r128')
    
    ntrain = train_C.shape[0]
    
    test_C = openfile('test_C', 'data_geo5_r128')
    test_x = openfile('test_x_data', 'data_geo5_r128')
    test_y = openfile('test_y_data', 'data_geo5_r128')
    test_U = openfile('test_U', 'data_geo5_r128')
    ntest = test_C.shape[0]
    
    # Reshape
    train_C = train_C.reshape(ntrain, S, S)
    test_C = test_C.reshape(ntest, S, S)
    
    train_b = np.zeros((S-2, S-2))
    train_b = np.pad(train_b, pad_width=1, mode='constant', constant_values=1)
    train_b = train_b.reshape(1, S, S)
    test_b = train_b.repeat(ntest, axis=0)
    train_b = train_b.repeat(ntrain, axis=0)
    
    train_x = train_x.reshape(ntrain, S, S)
    train_y = train_y.reshape(ntrain, S, S)
    test_x = test_x.reshape(ntest, S, S)
    test_y = test_y.reshape(ntest, S, S)
    
    train_U = train_U.reshape(ntrain, S, S)
    test_U = test_U.reshape(ntest, S, S)
    
    # Expand dims
    train_C = np.expand_dims(train_C, axis=-1)
    test_C = np.expand_dims(test_C, axis=-1)
    train_x = np.expand_dims(train_x, axis=-1)
    train_y = np.expand_dims(train_y, axis=-1)
    test_x = np.expand_dims(test_x, axis=-1)
    test_y = np.expand_dims(test_y, axis=-1)
    train_U = np.expand_dims(train_U, axis=-1)
    test_U = np.expand_dims(test_U, axis=-1)
    train_b = np.expand_dims(train_b, axis=-1)
    test_b = np.expand_dims(test_b, axis=-1)
    
    # Concatenate
    train_a = np.concatenate((train_C, train_x, train_y, train_b), axis=3)
    test_a = np.concatenate((test_C, test_x, test_y, test_b), axis=3)
    
    train_u = train_U * 10
    test_u = test_U * 10
    
    return train_a, train_u, test_a, test_u

# АРХИТЕКТУРА 

class SpectralConv2d_fast(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_fast, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input, weights):
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FNO2d, self).__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.fc0 = nn.Linear(4, self.width)
        self.conv0 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv4 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)
        self.conv5 = SpectralConv2d_fast(self.width, self.width, self.modes1, self.modes2)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)
        self.w4 = nn.Conv2d(self.width, self.width, 1)
        self.w5 = nn.Conv2d(self.width, self.width, 1)
        
        self.b0 = nn.Conv2d(2, self.width, 1)
        self.b1 = nn.Conv2d(2, self.width, 1)
        self.b2 = nn.Conv2d(2, self.width, 1)
        self.b3 = nn.Conv2d(2, self.width, 1)
        self.b4 = nn.Conv2d(2, self.width, 1)
        self.b5 = nn.Conv2d(2, self.width, 1)
        
        self.c0 = nn.Conv2d(3, self.width, 1)
        self.c1 = nn.Conv2d(3, self.width, 1)
        self.c2 = nn.Conv2d(3, self.width, 1)
        self.c3 = nn.Conv2d(3, self.width, 1)
        self.c4 = nn.Conv2d(3, self.width, 1)
        self.c5 = nn.Conv2d(3, self.width, 1)

        self.fc1 = nn.Linear(self.width, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        grid_mesh = x[:,:,:,1:4]
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        grid_mesh = grid_mesh.permute(0, 3, 1, 2)
        grid = self.get_grid([x.shape[0], x.shape[-2], x.shape[-1]], x.device).permute(0, 3, 1, 2)
       
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x3 = self.b0(grid)
        x4 = self.c0(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x3 = self.b1(grid)
        x4 = self.c1(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x3 = self.b2(grid) 
        x4 = self.c2(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x3 = self.b3(grid) 
        x4 = self.c3(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv4(x)
        x2 = self.w4(x)
        x3 = self.b4(grid) 
        x4 = self.c4(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)
        
        x1 = self.conv5(x)
        x2 = self.w5(x)
        x3 = self.b5(grid)
        x4 = self.c5(grid_mesh)
        x = x1 + x2 + x3 + x4
        x = F.gelu(x)

        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc3(x)
        x = F.gelu(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)


def apply_random_bias(xx):
    """Применяем случайное смещение БЕЗ inplace операций"""
    batch_size = xx.shape[0]
    xx_modified = xx.clone()  # Создаем копию!
    
    for i in range(batch_size):
        x_bias = (random.random() - 0.5) * 20
        y_bias = (random.random() - 0.5) * 20
        
        # Создаем новые тензоры вместо модификации на месте
        new_x = xx_modified[i, :, :, 1] + x_bias
        new_y = xx_modified[i, :, :, 2] + y_bias
        
        xx_modified[i, :, :, 1] = new_x
        xx_modified[i, :, :, 2] = new_y
    
    return xx_modified

if __name__ == "__main__":
    # Конфигурация
    modes = 16
    width = 32
    batch_size = 8
    epochs = 150
    learning_rate = 0.001
    S = 128

    # Папки
    os.makedirs('result_data_geo5_r128_to', exist_ok=True)
    os.makedirs('loss', exist_ok=True)
    os.makedirs('model', exist_ok=True)

    print("Loading data...")
    train_a, train_u, test_a, test_u = load_data(0, 0, S)

    train_a = torch.FloatTensor(train_a)
    train_u = torch.FloatTensor(train_u)
    test_a = torch.FloatTensor(test_a)
    test_u = torch.FloatTensor(test_u)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )

    model = FNO2d(modes, modes, width).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Рекомендую смягчить, если учим 150 эпох
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    myloss = LpLoss(size_average=True)

    print("Starting training...")
    loss_train_all = []
    loss_test_all = []

    for ep in range(epochs):
        model.train()
        sumtrain = 0.0
        ntrain_batches = 0

        for xx, yy in train_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            xx_modified = apply_random_bias(xx)

            optimizer.zero_grad()
            pred = model(xx_modified)

            bsz = pred.shape[0]
            loss = myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1), type=False)

            loss.backward()
            optimizer.step()

            sumtrain += float(loss.item())
            ntrain_batches += 1

        avg_train_loss = sumtrain / max(1, ntrain_batches)

        model.eval()
        sumtest = 0.0
        ntest_batches = 0

        with torch.no_grad():
            for xx, yy in test_loader:
                xx = xx.to(device)
                yy = yy.to(device)

                pred = model(xx)

                bsz = pred.shape[0]
                loss = myloss(pred.reshape(bsz, -1), yy.reshape(bsz, -1), type=False)

                sumtest += float(loss.item())
                ntest_batches += 1

        avg_test_loss = sumtest / max(1, ntest_batches)

        loss_train_all.append(avg_train_loss)
        loss_test_all.append(avg_test_loss)

        if ep % 5 == 0:
            print(f'Epoch {ep}: Train Loss: {avg_train_loss:.6f}, Test Loss: {avg_test_loss:.6f}')

        if ep % 10 == 0:
            torch.save(model.state_dict(), f'model/model_data_geo5_r128_to_ep{ep:04d}.pth')
            print(f'Model saved at epoch {ep}')

        scheduler.step()

    save_data('./result_data_geo5_r128_to/train_loss.csv', np.array(loss_train_all))
    save_data('./result_data_geo5_r128_to/test_loss.csv', np.array(loss_test_all))
    save_pig_loss(loss_train_all, loss_test_all, 'loss/model_data_geo5_r128_to.png')

    print("Training completed!")
    print(f"Final Train Loss: {loss_train_all[-1]:.6f}")
    print(f"Final Test Loss: {loss_test_all[-1]:.6f}")
