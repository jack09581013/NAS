import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from dataset.data_loader import *
from colorama import Style
import profile
import numpy as np
import os
import utils
import traceback
import datetime
from config import *
from Map2D.build_model import AutoMap2D
import matplotlib.pyplot as plt


def main():
    config = Config_Map2D_Search()
    device = config.device
    exception_count = 0

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    model = AutoMap2D(layers=config.num_layers, filter=config.filter_multiplier, block=config.block_multiplier,
                      step=config.step, device=device)
    model.to(device)

    optimizer = torch.optim.SGD(
        model.auto2d.weight_parameters(),
        config.weight_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    architect_optimizer = torch.optim.SGD(
        model.auto2d.arch_parameters(),
        config.weight_lr,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    # architect_optimizer = torch.optim.Adam(model.auto2d.arch_parameters(),
    #                                        lr=config.arch_lr, betas=(0.9, 0.999),
    #                                        weight_decay=config.arch_weight_decay)

    train_datasetA = Map2D_Dataset('trainA', config.height, config.width)
    train_datasetB = Map2D_Dataset('trainB', config.height, config.width)
    test_dataset = Map2D_Dataset('test', config.height, config.width)

    print(f'Batch size: {config.batch}')
    print('Using dataset:', config.dataset_name)
    print('Image size:', (config.height, config.width))
    print('Number of training data:', len(train_datasetA))
    print('Number of testing data:', len(test_dataset))
    print(f'Total number of model parameters : {sum([p.data.nelement() for p in model.parameters()]):,}')
    print(f'Number of Auto2D Net parameters: {sum([p.data.nelement() for p in model.auto2d.parameters()]):,}')

    train_loaderA = DataLoader(train_datasetA, batch_size=config.batch, shuffle=False,
                               num_workers=config.num_workers, pin_memory=True, drop_last=True)
    train_loaderB = DataLoader(train_datasetB, batch_size=config.batch, shuffle=False,
                               num_workers=config.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)

    best_error = 100
    epoch_loss = []
    for epoch in range(config.epoch):
        try:
            print(f"[{epoch}/{config.epoch}] Start training ...........")
            train_loss = []
            test_error = []

            model.train()
            for batch_index, (X, Y) in enumerate(train_loaderA):
                # utils.plot_image(X[0, :3], bayer=False)
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                optimizer.zero_grad()

                output = model(X)
                loss = F.smooth_l1_loss(output, Y, reduction='mean')
                print(f'loss = {loss:.3f}')
                if loss.data.cpu() is not torch.nan:
                    train_loss.append(loss.data.cpu())
                loss.backward()

                optimizer.step()

            epoch_loss.append(np.array(train_loss).mean())
            print(f'total training loss: {epoch_loss[-1]:.3f}')

            if epoch >= config.alpha_epoch:
                print(f"[{epoch}/{config.epoch}] Start searching architecture ...........")
                for batch_index, (X, Y) in enumerate(train_loaderB):
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    architect_optimizer.zero_grad()

                    output = model(X)
                    loss = F.smooth_l1_loss(output, Y.squeeze(1), reduction='mean')
                    print(f'SA loss = {loss:.3f}')
                    loss.backward()

                    architect_optimizer.step()

            print(f"[{epoch}/{config.epoch}] Start validation ...........")
            model.eval()
            with torch.no_grad():
                for batch_index, (X, Y) in enumerate(test_loader):
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    output = model(X)
                    error = torch.mean(torch.abs(output - Y))
                    print(f'error = {error:.3f}')

                    test_error.append(error.data.cpu())

            total_test_error = np.array(test_error).mean()
            print(f'total testing error: {total_test_error:.3f}, best error: {best_error:.3f}')

            if epoch >= config.alpha_epoch and total_test_error < best_error:
                print(f'Find best model, error = {total_test_error:.3f}')
                torch.save(model.state_dict(), config.save_best_model_path)
                best_error = total_test_error


        except Exception as err:
            # traceback.format_exc()  # Traceback string
            traceback.print_exc()
            exception_count += 1
            # if exception_count >= 50:
            #     exit(-1)
            if config.is_debug:
                exit(-1)

    plt.plot(epoch_loss, label='Train', marker='o')
    plt.savefig(config.save_history_file_path)


if __name__ == '__main__':
    main()
