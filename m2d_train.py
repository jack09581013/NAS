from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from dataset.data_loader import *
from colorama import Style
import profile
import numpy as np
import os
import utils
import traceback
import datetime
from config import *
from Map2D.Map2D import Map2D
from simple_model.SimpleModel import SimpleModel
import matplotlib.pyplot as plt


def main():
    config = Config_Map2D_Train()
    # config = Config_SimpleModel()
    device = config.device
    exception_count = 0

    if 'cuda' in device:
        torch.backends.cudnn.benchmark = True

    if isinstance(config, Config_Map2D_Train):
        model = Map2D(config)
    elif isinstance(config, Config_SimpleModel):
        model = SimpleModel()

    if os.path.isfile(config.model_path):
        print('Load model from ' + config.model_path)
        model.load_state_dict(torch.load(config.model_path))
    else:
        print('Cannot find any model')

    model.to(device)
    optimizer = optim.Adam(params=model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))

    train_dataset = Map2D_Dataset('trainA', config.height, config.width)
    test_dataset = Map2D_Dataset('test', config.height, config.width)

    print(f'Batch size: {config.batch}')
    print('Using dataset:', train_dataset)
    print('Image size:', (config.height, config.width))
    print('Number of training data:', len(train_dataset))
    print('Number of testing data:', len(test_dataset))
    print(f'Total number of model parameters : {sum([p.data.nelement() for p in model.parameters()]):,}')

    if isinstance(config, Config_Map2D_Train):
        print(f'Number of Auto2D Net parameters: {sum([p.data.nelement() for p in model.auto2d.parameters()]):,}')

    train_loader = DataLoader(train_dataset, batch_size=config.batch, shuffle=False,
                              num_workers=config.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch, shuffle=False,
                             num_workers=config.num_workers, pin_memory=True, drop_last=True)

    train_loss = []
    test_loss = []
    for epoch in range(config.epoch):
        try:
            print(f"[{epoch}/{config.epoch}] Start training ...........")
            total_loss = []

            model.train()
            for batch_index, (X, Y) in enumerate(train_loader):
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)

                optimizer.zero_grad()
                output = model(X)
                loss = F.smooth_l1_loss(output, Y)
                print(f'loss = {loss:.3f}')
                total_loss.append(float(loss))
                loss.backward()
                optimizer.step()

            train_loss.append(np.array(total_loss).mean())
            print(f'avg train loss: {train_loss[-1]:.3f}')

            print(f"[{epoch}/{config.epoch}] Start validation ...........")
            total_loss = []

            model.eval()
            with torch.no_grad():
                for batch_index, (X, Y) in enumerate(test_loader):
                    X = X.to(device, non_blocking=True)
                    Y = Y.to(device, non_blocking=True)

                    output = model(X)
                    loss = F.smooth_l1_loss(output, Y)

                    # last epoch
                    if epoch >= config.epoch - 1:
                        print(f'loss = {loss:.3f}')
                        print(output.data.cpu().numpy().reshape(-1)[:6])
                        print(Y.data.cpu().numpy().reshape(-1)[:6])
                        print()

                    total_loss.append(float(loss))

            test_loss.append(np.array(total_loss).mean())
            print(f'avg test loss: {test_loss[-1]:.3f}')

        except Exception as err:
            # traceback.format_exc()  # Traceback string
            traceback.print_exc()
            exception_count += 1
            if config.is_debug or exception_count >= 50:
                exit(-1)

    print(f'Total train loss: {np.array(train_loss).mean():.5f}')
    print(f'Total test loss: {np.array(test_loss).mean():.5f}')
    plt.plot(train_loss, label='Train', marker='o')
    plt.plot(test_loss, label='Test', marker='o')
    plt.savefig(config.save_history_file_path)


if __name__ == '__main__':
    main()
