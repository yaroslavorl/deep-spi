from skimage import metrics
import numpy as np
import torch


def test_model(model, device, loss_func, test_set, mask, patterns, size_img):
    model.eval()
    loss_list = []
    pnsr_list = []

    with torch.no_grad():
        for _, batch in enumerate(test_set):
            batch = batch.to(device)
            outputs = model(torch.matmul(batch.reshape(-1, 1, size_img), mask).reshape(-1, patterns, 1, 1))
            loss_list.append(loss_func(outputs, batch).item())
            pnsr_list.append(metrics.peak_signal_noise_ratio(batch.to("cpu").numpy(), outputs.to("cpu").numpy()))

    print(f'MSE: {np.mean(loss_list)}')
    print(f'PNSR: {np.mean(pnsr_list)}')
