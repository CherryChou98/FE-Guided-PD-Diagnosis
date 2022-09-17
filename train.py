import os
import time
import argparse
import numpy as np

from resnet_pd import Res_pd_nofreeze
from utils import seed_model
from PD_Dataset import PdDataSet

import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

class RecorderMeter(object):
    pass


def run_train(args):
    seed_model(gpu_id=0)
    with open(os.path.join(args.log_dir, 'log-'+time.strftime("%Y-%m-%d-%H-%M-%S-", time.localtime())+args.p1), 'a') as f_log:
        print(args.p1)
        print(args.p2)
        time_start=time.time()
        model = Res_pd_nofreeze(pretrained=args.pretrained_model) 

        print("batch_size:", args.batch_size)
        f_log.write("batch_size:" + str(args.batch_size) + "\n")

        if args.checkpoint:
            print("Loading pretrained weights...", args.checkpoint)
            checkpoint = torch.load(args.checkpoint)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)

        data_transforms = transforms.Compose([       #######################
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(scale=(0.02,0.25))])

        train_dataset = PdDataSet(args.data_path, args.label_path, args.p1, mode='train', transform=data_transforms)
        print('Train set size:', train_dataset.__len__())
        f_log.write('Train set size:' + str(train_dataset.__len__()) + '\n')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=True)
        
        data_transforms_val = transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        val_dataset = PdDataSet(args.data_path, args.label_path, args.p2, mode='eval', transform=data_transforms_val)
        val_num = val_dataset.__len__()
        print('Validation set size:', val_num)
        f_log.write('Validation set size:' + str(val_num) + '\n')
        val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=args.workers, shuffle=False, pin_memory=True)

        params = model.parameters()
        if args.optimizer == 'adam':
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), args.lr, momentum=args.momentum, weight_decay=1e-4)
        else:
            raise ValueError("Optimizer not supported.")
        print(optimizer)
        f_log.write(str(optimizer)+'\n')

        model = model.cuda()
        CE_criterion = torch.nn.CrossEntropyLoss()

        best_acc = 0
        for i in range(1, args.epochs + 1):
            train_loss = 0.0
            correct_sum = 0
            iter_cnt = 0
            model.train()
            for batch_i, (imgs, targets) in enumerate(train_loader):
                iter_cnt += 1
                optimizer.zero_grad()
                imgs = imgs.cuda()
                targets = targets.cuda().squeeze()
                outputs = model(imgs.cuda())
                #print(targets.shape, outputs.shape)

                loss = CE_criterion(outputs, targets)
                loss.backward()
                optimizer.step()     
                train_loss += loss
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets).sum()
                correct_sum += correct_num

            train_acc = correct_sum.float() / float(train_dataset.__len__())
            train_loss = train_loss/iter_cnt
            print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))
            f_log.write('[Epoch %d] Training accuracy: %.4f. Loss: %.3f LR: %.6f\n' %
                (i, train_acc, train_loss, optimizer.param_groups[0]["lr"]))


            with torch.no_grad():
                val_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                model.eval()
                for batch_i, (imgs, targets) in enumerate(val_loader):
                    iter_cnt += 1
                    imgs = imgs.cuda()
                    outputs = model(imgs.cuda())
                    targets = targets.cuda().squeeze()

                    CE_loss = CE_criterion(outputs, targets)
                    loss = CE_loss
                    val_loss += loss

                    _, predicts = torch.max(outputs, 1)
                    correct_or_not = torch.eq(predicts, targets)
                    bingo_cnt += correct_or_not.sum().cpu()
                    
                val_loss = val_loss/iter_cnt
                val_acc = bingo_cnt.float()/float(val_num)
                val_acc = np.around(val_acc.numpy(), 4)
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, val_acc, val_loss))
                f_log.write("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f\n" % (i, val_acc, val_loss))

                if val_acc > best_acc:
                    best_acc = val_acc
                    print("best_acc:" + str(best_acc))
                    f_log.write("best_acc:" + str(best_acc) + '\n')
        time_end=time.time()
        time_elapsed = time_end - time_start
        print('Time cost:',time_elapsed,'s')
        f_log.write('Time cost:' + str(time_elapsed) + 's\n')
        print("best_acc:", best_acc)
        f_log.write("best_acc:" + str(best_acc) + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='', help='path for dataset')
    parser.add_argument('--label_path', type=str, default='', help='directory for image label file')
    parser.add_argument('--pretrained_model', type=str, default='../pretrained_models/resnet18_msceleb.pth', help='path for pretrained_weights')
    parser.add_argument('--p1', type=str, default='', help='path for train label file')
    parser.add_argument('--p2', type=str, default='', help='path for eval label file')
    parser.add_argument('-c', '--checkpoint', type=str, default=None, help='path for Pytorch checkpoint file')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--val_batch_size', type=int, default=32, help='Batch size for validation.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--log_dir', type=str, default='log', help='file path for logging trian details')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--weight_decay', default=1e-5, type=float, help='Weight_decay for Adam')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=70, help='Total training epochs.')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()
    run_train(args)

