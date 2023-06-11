from skimage import metrics
import numpy as np
import torch


def test_model(model, coder, device, loss_func, test_set):
    model.eval()
    coder.eval()
    loss_list = []
    psnr_list = []

    with torch.no_grad():
        for _, batch in enumerate(test_set):
            batch = batch.to(device)
            encoded_batch = coder(batch)
            outputs = model(encoded_batch)
            loss_list.append(loss_func(outputs, batch).item())
            psnr_list.append(metrics.peak_signal_noise_ratio(batch.to("cpu").numpy(), outputs.to("cpu").numpy()))

    print(f'MSE: {np.mean(loss_list)}')
    print(f'PSNR: {np.mean(psnr_list)}')
