# -*- coding: utf-8 -*-
import os
from os.path import join
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import torch
from models.DenseBiasNet_state_a_deeper import DenseBiasNet
from utils.dataloader_5c import DatasetFromFolder3D_R
from utils.loss import crossentropy


class AverageMeter(object):
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train_epoch(net_S, opt_S, loss_S, dataloader_R, epoch, n_epochs, Iters):
    loss_S_log = AverageMeter()
    net_S.train()
    for i in range(Iters):
        net_S.zero_grad()
        opt_S.zero_grad()
        # 真实图像训练
        input, target = next(dataloader_R.__iter__())
        if torch.cuda.is_available():
            input = input.cuda(async=True)
            target = target.cuda(async=True)

        seg = net_S(input)
        errS = loss_S(seg, target)
        errS.backward()
        opt_S.step()
        loss_S_log.update(errS.data, target.size(0))

        res = '\t'.join(['Epoch: [%d/%d]' % (epoch + 1, n_epochs),
                         'Iter: [%d/%d]' % (i + 1, Iters),
                         'Loss_S %f' % (loss_S_log.avg)])
        print(res)
    return


def train_net(n_epochs=0, batch_size=1, lr=1e-4, Iters=200, model_name="DenseBiasNet_5c_aug"):
    shape = (128, 128, 128)
    train_image_dir = 'data/train/'
    save_dir = 'results'
    checkpoint_dir = 'weights'
    test_image_dir = 'data/test/image'

    net_S = DenseBiasNet(n_channels=1, n_classes=6)
    net_S.load_state_dict(torch.load('{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, "200")))

    if torch.cuda.is_available():
        net_S = net_S.cuda()

    train_dataset_R = DatasetFromFolder3D_R(train_image_dir, shape=shape, num_classes=6, is_aug=True)
    dataloader_R = DataLoader(train_dataset_R, batch_size=batch_size, shuffle=True)

    opt_S = torch.optim.Adam(net_S.parameters(), lr=lr)

    loss_S = crossentropy()

    for epoch in range(n_epochs):
        train_epoch(net_S, opt_S, loss_S, dataloader_R, epoch, n_epochs, Iters)
        if epoch % 20 == 0:
            torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, epoch))
        predict(net_S, save_dir, "test_once")
    torch.save(net_S.state_dict(), '{0}/{1}_epoch_{2}.pth'.format(checkpoint_dir, model_name, n_epochs))
    predict(net_S, save_dir, test_image_dir)
    dice = Dice()
    mean = np.mean(dice, axis=0)
    print(mean)

def predict(model, save_path, img_path):
    print("Predict test data")
    model.eval()
    image_filenames = [x for x in os.listdir(img_path) if is_image3d_file(x)]
    for imagename in image_filenames:
        print(imagename)

        image = np.fromfile(join(img_path, imagename), dtype=np.uint16)
        n_pieces = int(image.shape[0] / (150 * 150))
        image = image.reshape(1, 1, n_pieces, 150, 150)
        image = np.where(image < 0., 0., image)
        image = np.where(image > 2048., 2048., image)
        image = image.astype(np.float32)
        image = (image) / 2048.

        image = torch.from_numpy(image)
        if torch.cuda.is_available():
            image = image.cuda(0)
        with torch.no_grad():
            predict= model(image).data.cpu().numpy()

        predict = np.argmax(predict[0], axis=0)
        predict = predict.astype(np.uint16)
        predict.tofile((join(save_path, imagename)))

def is_image3d_file(filename):
    return any(filename.endswith(extension) for extension in [".raw"])

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train_net()