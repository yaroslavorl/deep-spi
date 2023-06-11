from deepSPI.model.decoder import Decoder, Coder
from deepSPI.data import get_dataframe
from deepSPI.test import test_model

from torch.utils.data import DataLoader
import torch
import torch.optim as opt
import torch.nn as nn
import numpy as np


def fit_model(model, coder, loss_func, optimizer, train_set, epochs):
    loss_list = []
    for epoch in range(1, epochs + 1):
        for idx, batch in enumerate(train_set):
            batch = batch.to(device)
            encoded_batch = coder(batch)
            outputs = model(encoded_batch)
            loss = loss_func(outputs, batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            print(f'Эпоха: {epoch}/{epochs} Батч: {idx} loss: {loss.item()}')
        print(f'loss на эпохе {np.mean(loss_list)}')
        scheduler.step()
        loss_list = []


def binary_mask(size):
    return torch.randint(0, 2, size, dtype=torch.float32) * 2 - 1


if __name__ == '__main__':
    size_flat_img = 4096
    patterns = 300
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    net = Decoder(patterns, size_img=(64, 64)).to(device)
    coder_layer = Coder(patterns, size_img=(64, 64))
    coder_layer.coder.weight = torch.nn.Parameter(binary_mask((patterns, 1, 64, 64)), requires_grad=False)
    coder_layer.to(device)
    coder_layer.eval()

    optimizer = opt.Adam(net.parameters(), lr=0.003)
    scheduler = opt.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.2)
    loss = nn.MSELoss()
    print(net)

    train_x, test_x = get_dataframe(8000, path='train_X.bin')

    train_x = DataLoader(train_x, batch_size=60, shuffle=True)
    test_x = DataLoader(test_x, batch_size=60)

    fit_model(net, coder_layer, loss, optimizer, train_x, epochs=40)
    test_model(net, coder_layer, device, loss, test_x)
