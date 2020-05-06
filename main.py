# -*- coding: utf-8 -*-

"""
    @date:   2020.04.08 week15 星期三
    @author: samuel ko
    @target: 开整~
"""
import itertools
import os

import torch
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from network.rae_mnist import Encoder, Decoder
from opts.opts import TrainOptions
from utils.libs import LatentSpaceSampler
from utils.losses.loss import TotalLoss
from utils.utils import SaveModel


def main(opts):
    # 0) 创建数据集
    # Create the data loader

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.1307,), std=(0.3081,))])

    data_train = datasets.MNIST(root="./data/",
                                transform=transform,
                                train=True,
                                download=True)

    data_test = datasets.MNIST(root="./data/",
                               transform=transform,
                               train=False)

    data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                    batch_size=opts.batch_size,
                                                    shuffle=True)

    data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                                   batch_size=opts.batch_size,
                                                   shuffle=True)

    # 1) 创建模型以及初始化相机模型和光栅化配置和着色器.
    # Create the model
    encoder = Encoder()
    decoder = Decoder()
    encoder.to(opts.device)
    decoder.to(opts.device)

    optimizer = torch.optim.Adam(itertools.chain(encoder.parameters(),
                                                 decoder.parameters()),
                                 lr=5e-4, betas=(0.5, 0.999), weight_decay=0.0001)

    # 2) 开始训练.
    loss_stat_list = [0.0]
    loss_calc = TotalLoss(loss=opts.loss_choice)
    for ep in range(0, opts.epoch):

        train_bar = tqdm(data_loader_train)
        test_bar = tqdm(data_loader_test)

        encoder.train()
        decoder.train()

        for i, data in enumerate(train_bar):
            optimizer.zero_grad()

            X_train, y_train = data
            X_train = X_train.to(opts.device)

            embeddings = encoder(X_train)
            output = decoder(embeddings)

            loss = loss_calc(output, X_train, embeddings)
            loss_stat_list[0] = loss
            loss.backward()

            optimizer.step()

            # Output training stats
            train_bar.set_description(
                "Epoch {} [{}, {}] [Total Loss] {}".format(ep, i + 1, len(data_loader_train), loss_stat_list[-1]))

        for i, data in enumerate(test_bar):
            encoder.eval()
            decoder.eval()

            X_test, y_test = data
            X_test = X_test.to(opts.device)
            outputs = decoder(encoder(X_test))

            with torch.no_grad():
                save_image(outputs, os.path.join(opts.det, 'images', 'reconstruct' + str(ep) + '.png'), nrow=8,
                           normalize=True)

            latentspace = LatentSpaceSampler(encoder)
            zs = latentspace.get_zs(X_test)
            zs = torch.from_numpy(zs).float().to(opts.device)
            outputs = decoder(zs)

            with torch.no_grad():
                save_image(outputs, os.path.join(opts.det, 'images', 'sampled' + str(ep) + '.png'), nrow=8,
                           normalize=True)

        if ep % 10 == 0:
            SaveModel(encoder, decoder, dir='./', epoch=ep)


if __name__ == "__main__":
    opts = TrainOptions().parse()
    main(opts)
