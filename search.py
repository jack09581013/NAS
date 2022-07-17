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
from LEAStereo.build_model import AutoStereo
import matplotlib.pyplot as plt


def main():
    config = Config_Search()
    device = config.device
    exception_count = 0

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    model = AutoStereo(maxdisp=config.max_disparity,
                       Fea_Layers=config.fea_num_layers, Fea_Filter=config.fea_filter_multiplier,
                       Fea_Block=config.fea_block_multiplier, Fea_Step=config.fea_step,
                       Mat_Layers=config.mat_num_layers, Mat_Filter=config.mat_filter_multiplier,
                       Mat_Block=config.mat_block_multiplier, Mat_Step=config.mat_step, device=device)
    model.to(device)

    optimizer_F = optim.Adam(params=model.feature.weight_parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    optimizer_M = optim.Adam(params=model.matching.weight_parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    architect_optimizer_F = optim.Adam(params=model.feature.arch_parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    architect_optimizer_M = optim.Adam(params=model.matching.arch_parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    if config.dataset_name == 'flyingthings3D':
        train_dataset = FlyingThings3D(config.max_disparity, config.dataset_dir, type='train', use_crop_size=True,
                                       crop_size=(config.height, config.width), crop_seed=0, image='finalpass')
        test_dataset = FlyingThings3D(config.max_disparity, config.dataset_dir, type='test', use_crop_size=True,
                                      crop_size=(config.height, config.width), crop_seed=0, image='finalpass')
    else:
        raise Exception('Cannot find dataset: ' + config.dataset_name)

    print(f'Batch size: {config.batch}')
    print('Using dataset:', config.dataset_name)
    print('Image size:', (config.height, config.width))
    print('Max disparity:', config.max_disparity)
    print('Number of training data:', len(train_dataset))
    print('Number of testing data:', len(test_dataset))
    print(f'Total number of model parameters : {sum([p.data.nelement() for p in model.parameters()]):,}')
    print(f'Number of Feature Net parameters: {sum([p.data.nelement() for p in model.feature.parameters()]):,}')
    print(f'Number of Matching Net parameters: {sum([p.data.nelement() for p in model.matching.parameters()]):,}')

    if config.dataset_name == 'flyingthings3D':
        train_loaderA = DataLoader(random_subset(train_dataset, 10, seed=0), batch_size=config.batch, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=True, drop_last=True)
        train_loaderB = DataLoader(random_subset(train_dataset, 10, seed=1), batch_size=config.batch, shuffle=False,
                                   num_workers=config.num_workers, pin_memory=True, drop_last=True)
        test_loader = DataLoader(random_subset(test_dataset, 10, seed=0), batch_size=config.batch, shuffle=False,
                                 num_workers=config.num_workers, pin_memory=True, drop_last=True)

    else:
        raise Exception('Cannot find dataset: ' + config.dataset_name)

    best_error = 100
    epoch_loss = []
    for epoch in range(config.epoch):
        try:
            print(f"[{epoch}/{config.epoch}] Start training ...........")
            train_loss = []
            test_error = []

            model.train()
            for batch_index, (X, Y, pass_info) in enumerate(train_loaderA):
                # utils.plot_image(X[0, :3], bayer=False)
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                optimizer_F.zero_grad()
                optimizer_M.zero_grad()

                output = model(X[:, 0:3], X[:, 3:6])
                loss = F.smooth_l1_loss(output, Y.squeeze(1), reduction='mean')
                print(f'loss = {loss:.3f}')
                train_loss.append(loss.data.cpu())
                loss.backward()

                optimizer_F.step()
                optimizer_M.step()

            epoch_loss.append(np.array(train_loss).mean())
            print(f'total training loss: {epoch_loss[-1]:.3f}')

            if epoch >= config.alpha_epoch:
                print(f"[{epoch}/{config.epoch}] Start searching architecture ...........")
                for batch_index, (X, Y, pass_info) in enumerate(train_loaderB):
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    architect_optimizer_F.zero_grad()
                    architect_optimizer_M.zero_grad()

                    output = model(X[:, 0:3], X[:, 3:6])
                    loss = F.smooth_l1_loss(output, Y.squeeze(1), reduction='mean')
                    print(f'SA loss = {loss:.3f}')
                    loss.backward()

                    architect_optimizer_F.step()
                    architect_optimizer_M.step()

            print(f"[{epoch}/{config.epoch}] Start validation ...........")
            model.eval()
            with torch.no_grad():
                for batch_index, (X, Y, pass_info) in enumerate(test_loader):
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    output = model(X[:, 0:3], X[:, 3:6])
                    error = torch.mean(torch.abs(output - Y))
                    print(f'error = {error:.3f}')

                    test_error.append(error.data.cpu())

            total_test_error = np.array(test_error).mean()
            print(f'total testing error: {total_test_error:.3f}')

            if total_test_error < best_error:
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
