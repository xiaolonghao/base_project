# -*- coding: utf-8 -*-
import os
import argparse

import torch
from torch import nn,optim
from torch.backends import cudnn
from torch.cuda.random import manual_seed_all
from torch.utils.data.dataloader import DataLoader

import utils
import models
from datas import datasets,transforms
from config import Default_config

def train(args, device, net):

    net.train(True)
    train_transform = transforms.Compose([transforms.IncreaseRandomCrop(256,128),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(),
                                          transforms.Normalize(mean = [0.1,0.1,0.1],std = [0.1,0.1,0.1])])
    train_dataset = datasets.Datasets(data_dir_path=args.train_data_path,
                                      transforms=train_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    # 创建优化器
    optimer = optim.SGD([{'params':net.base.parameters(),'lr':args.lr},
                   {'params':net.classifier.parameters(),'lr':args.lr*0.1}],
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    lr_schedualer = optim.lr_scheduler.StepLR(optimer,step_size=args.lr_decay_step,gamma=args.lr_decay_ratio)

    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # 是否接着训练
    target_num = -1
    if args.continue_train:
        model_root_path = os.path.join(os.path.curdir,args.save_model_path)
        model_paths = os.listdir(model_root_path)
        import re
        r = re.compile(r'epoch(\d+)$')
        try:
            target_num = max(list(map(int,[r.search(num).groups()[0] for num in model_paths])))
            target_model_path = os.path.join(model_root_path,'epoch%d'%target_num)
            net.load_state_dict(torch.load(target_model_path))
        except:
            print('No used epoch.pth')
    all_avg_loss = []
    all_avg_acc = []
    for epoch in range(target_num+1,args.all_epoch):
        losses = 0
        counts = 0
        for ith,(img, p_id, cam_id,_) in enumerate(train_dataloader):
            img, p_id, cam_id = img.to(device), p_id.to(device), cam_id.to(device)
            output = net(img)
            loss = loss_function(output,p_id)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            predict = output.argmax(dim=1)
            losses += loss.item()
            counts += predict.eq(p_id).sum().item()

            if ith % args.print_info_epoch ==0:
                data_count = args.batch_size * ith + len(p_id)
                print('               epoch%d-%dth,loss = %.2f, acc = %.2f' %(epoch,ith,
                                                               losses /data_count,
                                                               counts * 1.0 /data_count ))
        lr_schedualer.step(epoch)
        avg_loss = losses / len(train_dataset)
        avg_acc = counts * 1.0 / len(train_dataset)
        all_avg_loss.append(avg_loss)
        all_avg_acc.append(avg_acc)

        print('epoch = %d,loss = %.2f, acc = %.2f' %(epoch,avg_loss,avg_acc))

        if (epoch+1) % args.save_model_epoch == 0:
            save_path = os.path.join(os.path.curdir,args.save_model_path,'epoch%d'%epoch)
            torch.save(net.state_dict(),save_path)

    # 统计训练损失和准确度
    if args.display_train_statistics:
        utils.display_statistics(args.stage, args.all_epoch-target_num-1, all_avg_loss,all_avg_acc)
    
def train_val(args, device, net):
    # 训练集数据
    train_transform = transforms.Compose([transforms.IncreaseRandomCrop(256,128),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.RandomErasing(),
                                          transforms.Normalize(mean = [0.1,0.1,0.1],std = [0.1,0.1,0.1])])
    train_dataset = datasets.Datasets(data_dir_path=args.train_data_path,
                                      transforms=train_transform)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  # num_workers=args.num_workers,
                                  pin_memory=True)
    # 验证集数据
    val_transform = transforms.Compose([transforms.Resize((256,128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.1,0.1,0.1],std = [0.1,0.1,0.1])])
    val_dataset = datasets.Datasets(data_dir_path=args.val_data_path,
                                      transforms=val_transform)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  # num_workers=args.num_workers,
                                  pin_memory=True)
    # 创建优化器
    optimer = optim.SGD([{'params':net.base.parameters(),'lr':args.lr},
                   {'params':net.classifier.parameters(),'lr':args.lr*0.1}],
                    momentum=0.9,
                    weight_decay=args.weight_decay,
                    nesterov=True)
    lr_schedualer = optim.lr_scheduler.StepLR(optimer,step_size=args.lr_decay_step,gamma=args.lr_decay_ratio)

    loss_function = nn.CrossEntropyLoss(reduction='sum')

    # 是否接着训练
    target_num = -1
    if args.continue_train:
        model_root_path = os.path.join(os.path.curdir,args.save_model_path)
        model_paths = os.listdir(model_root_path)
        import re
        r = re.compile(r'epoch(\d+)$')
        try:
            target_num = max(list(map(int,[r.search(num).groups()[0] for num in model_paths])))
            target_model_path = os.path.join(model_root_path,'epoch%d'%target_num)
            net.load_state_dict(torch.load(target_model_path))
        except:
            print('No used epoch.pth')

    all_avg_loss = []
    all_avg_acc = []
    val_all_avg_loss = []
    val_all_avg_acc = []
    for epoch in range(target_num+1,args.all_epoch):
        # 训练集
        net.train(True)
        losses = 0
        counts = 0
        for ith,(img, p_id, cam_id,_) in enumerate(train_dataloader):
            img, p_id, cam_id = img.to(device), p_id.to(device), cam_id.to(device)
            output = net(img)
            loss = loss_function(output,p_id)

            optimer.zero_grad()
            loss.backward()
            optimer.step()

            predict = output.argmax(dim=1)
            losses += loss.item()
            counts += predict.eq(p_id).sum().item()

            if ith % args.print_info_epoch ==0:
                data_count = args.batch_size * ith + len(p_id)
                print('            Train: epoch%d-%dth,loss = %.2f, acc = %.2f' %(epoch,ith,
                                                               losses /data_count,
                                                               counts * 1.0 /data_count ))
        lr_schedualer.step(epoch)
        avg_loss = losses / len(train_dataset)
        avg_acc = counts * 1.0 / len(train_dataset)
        all_avg_loss.append(avg_loss)
        all_avg_acc.append(avg_acc)
        print('Train: epoch = %d,loss = %.2f, acc = %.2f' %(epoch,avg_loss,avg_acc))

        # 验证集
        net.eval()
        val_losses = 0
        val_counts = 0
        with torch.no_grad():
            for img, p_id, cam_id, _ in val_dataloader:
                img, p_id, cam_id = img.to(device), p_id.to(device), cam_id.to(device)
                output = net(img)
                loss = loss_function(output,p_id)

                predict = output.argmax(dim=1)
                val_losses += loss.item()
                val_counts += predict.eq(p_id).sum().item()
            avg_loss = val_losses / len(val_dataset)
            avg_acc = val_counts * 1.0 / len(val_dataset)
            val_all_avg_loss.append(avg_loss)
            val_all_avg_acc.append(avg_acc)
            print('Val: epoch = %d,loss = %.2f, acc = %.2f' %(epoch,avg_loss,avg_acc))

        # 保存模型
        if (epoch+1) % args.save_model_epoch == 0:
            save_path = os.path.join(os.path.curdir,args.save_model_path,'epoch%d'%epoch)
            torch.save(net.state_dict(),save_path)

    # 统计训练损失和准确度
    if args.display_train_statistics:
        utils.display_statistics('train_val-train', args.all_epoch-target_num-1, all_avg_loss, all_avg_acc)
        utils.display_statistics('train_val-val', args.all_epoch-target_num-1, val_all_avg_loss, val_all_avg_acc)

def test(args, device, net):

    net.eval()
    test_transform = transforms.Compose([transforms.Resize((256,128)),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.1,0.1,0.1],std = [0.1,0.1,0.1])])
    query_dataset = datasets.Datasets(data_dir_path=args.test_query_data_path,
                                      transforms=test_transform)
    gallery_dataset = datasets.Datasets(data_dir_path=args.test_gallery_data_path,
                                      transforms=test_transform)

    query_dataloader = DataLoader(query_dataset,
                                  batch_size=args.batch_size,
                                  # num_workers=args.num_workers,
                                  pin_memory=True)
    gallery_dataloader = DataLoader(gallery_dataset,
                              batch_size=args.batch_size,
                              # num_workers=args.num_workers,
                              pin_memory=True)
    # 加载模型
    model_root_path = os.path.join(os.path.curdir,args.save_model_path)
    target_model_path = os.path.join(model_root_path,args.load_model_path)
    net.load_state_dict(torch.load(target_model_path))

    # 提取query特征
    # 注：有两种方法 1.直接提取  2.直接提取+翻转提取
    '''
    def extract_feature(model, loader):
    features = torch.FloatTensor()

    for (inputs, labels) in loader:

        ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
        for i in range(2):
            if i == 1:
                inputs = inputs.index_select(3, torch.arange(inputs.size(3) - 1, -1, -1).long())
            input_img = inputs.to('cuda')
            outputs = model(input_img)
            f = outputs[0].data.cpu()
            ff = ff + f

        fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
        ff = ff.div(fnorm.expand_as(ff))
        features = torch.cat((features, ff), 0)
    return features
    '''
    query_feature = []
    query_labels = []
    query_cams = []
    query_paths = []
    with torch.no_grad():
        for img, p_id, cam_id, fpath in query_dataloader:
            img = img.to(device)
            output = net(img,output_feature = 'pool5')
            output = output.cpu()
            output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
            output = output.div(output_norm.expand_as(output))
            query_feature.append(output)
            query_labels.append(p_id)
            query_cams.append(cam_id)
            query_paths.append(fpath)
    query_f = torch.cat([f for f in query_feature], 0)
    query_l = torch.cat([l for l in query_labels], 0)
    query_c = torch.cat([c for c in query_cams], 0)
    query_p = [p  for path in query_paths for p in path]

    # 提取gallery特征
    gallery_feature = []
    gallery_labels = []
    gallery_cams = []
    gallery_paths = []
    with torch.no_grad():
        for img, p_id, cam_id,fpath in gallery_dataloader:
            img = img.to(device)
            output = net(img,output_feature = 'pool5')
            output = output.cpu()
            output_norm = torch.norm(output, p=2, dim=1, keepdim=True)
            output = output.div(output_norm.expand_as(output))
            gallery_feature.append(output)
            gallery_labels.append(p_id)
            gallery_cams.append(cam_id)
            gallery_paths.append(fpath)
    gallery_f = torch.cat([f for f in gallery_feature], 0)
    gallery_l = torch.cat([l for l in gallery_labels], 0)
    gallery_c = torch.cat([c for c in gallery_cams], 0)
    gallery_p = [p for path in gallery_paths for p in path]

    # 计算欧式距离
    q = query_f.size()[0]
    g = gallery_f.size()[0]
    query_ff = query_f.view(q,-1)
    gallery_ff = gallery_f.view(g,-1)

    dist = torch.pow(query_ff, 2).sum(dim=1, keepdim=True).expand(q, g) + \
            torch.pow(gallery_ff, 2).sum(dim=1, keepdim=True).expand(g, q).t()
    dist.addmm_(1, -2, query_ff, gallery_ff.t())

    # 计算cmc
    import numpy as np
    cmcs = utils.cmc(np.array(dist),
                     np.array(query_l),np.array(gallery_l),
                     np.array(query_c),np.array(gallery_c))
    maps = utils.mean_ap(np.array(dist),
                     np.array(query_l),np.array(gallery_l),
                     np.array(query_c),np.array(gallery_c))

    dist_q = torch.pow(query_ff, 2).sum(dim=1, keepdim=True).expand(q, q) + \
        torch.pow(query_ff, 2).sum(dim=1, keepdim=True).expand(q, q).t()
    dist_q.addmm_(1, -2, query_ff, query_ff.t())

    dist_g = torch.pow(gallery_ff, 2).sum(dim=1, keepdim=True).expand(g, g) + \
        torch.pow(gallery_ff, 2).sum(dim=1, keepdim=True).expand(g, g).t()
    dist_g.addmm_(1, -2, gallery_ff, gallery_ff.t())
    re_ranks = utils.re_ranking(np.array(dist),np.array(dist_q),np.array(dist_g))

    # visualization
    # rank_list, same_id = utils.get_rank_list(np.array(dist),
    #                                          np.array(query_l), np.array(query_c),
    #                                          np.array(gallery_l),np.array(gallery_c),
    #                                          rank_list_size=10)
    # utils.save_rank_list_to_im(rank_list, same_id,query_p,gallery_p,'')
    print('ok')

def test_(args, device, net):

    net.eval()
    test_transform = transforms.Compose([transforms.Resize(256,128),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean = [0.1,0.1,0.1],std = [0.1,0.1,0.1])])
    test_dataset = datasets.Datasets(data_dir_path=args.test_data_path,
                                      transforms=test_transform)
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=False,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    # 加载模型
    model_root_path = os.path.join(os.path.curdir,args.save_model_path)
    target_model_path = os.path.join(model_root_path,args.load_model_path)
    net.load_state_dict(torch.load(target_model_path))

    for img, p_id, cam_id in  test_dataloader:
        img, p_id, cam_id = img.to(device), p_id.to(device), cam_id.to(device)
        output = net(img,test = True)

if __name__=='__main__':
    
    # 配置命令行参数
    config = Default_config()
    parser = argparse.ArgumentParser(prog='Pytorch',description= 'Example of pytorch')
    parser.add_argument('--stage',type=str,default=config.stage,help='Train or val or test')
    parser.add_argument('--model',type=str,default=config.model,help='Name of model')
    parser.add_argument('--load_model_path',type=str,default=config.load_model_path,help='Load trained model path')
    parser.add_argument('--save_model_path',type=str,default=config.save_model_path,help='Save trained model path')
    parser.add_argument('--train_data_path',type=str,default=config.train_data_path,help='Train dataset path')
    parser.add_argument('--val_data_path',type=str,default=config.train_data_path,help='Val dataset path')
    parser.add_argument('--test_query_data_path',type=str,default=config.test_query_data_path,help='Test query dataset path')
    parser.add_argument('--test_gallery_data_path',type=str,default=config.test_gallery_data_path,help='Test gallery dataset path')
    parser.add_argument('--seed',type=int,default=config.seed,help='Set seed')
    parser.add_argument('--batch_size',type=int,default=config.batch_size,help='Batch size')
    parser.add_argument('--num_workers',type=int,default=config.num_workers,help='Number of workers')
    parser.add_argument('--print_info_epoch',type=int,default=config.print_info_epoch,help='How many epoch to print information')
    parser.add_argument('--all_epoch',type=int,default=config.all_epoch,help='All epoch')
    parser.add_argument('--save_model_epoch',type=int,default=config.save_model_epoch,help='Save model epoch')
    parser.add_argument('--lr',type=float,default=config.lr,help='Leaning rate')
    parser.add_argument('--lr_decay_ratio',type=float,default=config.lr_decay_ratio,help='Leaning rate decay  ratio')
    parser.add_argument('--lr_decay_step',type=int,default=config.lr_decay_step,help='Leaning rate decay step')
    parser.add_argument('--weight_decay',type=float,default=config.weight_decay,help='Weight decay ratio')
    parser.add_argument('--continue_train',type=bool,default=config.continue_train,help='Continue train or from zero')
    parser.add_argument('--display_train_statistics',type=bool,default=config.display_train_statistics,help='Display train statistics')
    args = parser.parse_args()

    cudnn.benchmark = True

    #选用设备 并设置随机种子数
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        manual_seed_all(args.seed)
    else:
        torch.manual_seed(args.seed)

    # 创建模型
    net = models.create(args.model).to(device)

    # 选择阶段
    if args.stage == 'train':
        train(args = args,device = device, net = net)
    elif args.stage == 'train_val':
        train_val(args = args,device = device, net = net)
    elif args.stage == 'test':
        test(args = args,device = device, net = net)
    else:
        raise KeyError("Unknown stage:", args.stage)
