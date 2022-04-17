import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pipeline.resnet_csra import ResNet_CSRA
from pipeline.vit_csra import VIT_B16_224_CSRA, VIT_L16_224_CSRA, VIT_CSRA
from pipeline.dataset import DataSet
from utils.evaluation.eval import evaluation
from utils.evaluation.warmUpLR import WarmUpLR
from tqdm import tqdm
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from time import time

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model default resnet101
    parser.add_argument("--model", default="resnet101", type=str)
    parser.add_argument("--num_heads", default=1, type=int)
    parser.add_argument("--lam",default=0.1, type=float)
    parser.add_argument("--load_from", default="models_local/resnet101_voc07_head1_lam0.1_94.7.pth", type=str)
    parser.add_argument("--cutmix", default=None, type=str) # the path to load cutmix-pretrained backbone
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop"], type=list)
    parser.add_argument("--test_aug", default=[], type=list)
    parser.add_argument("--img_size", default=448, type=int)
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--svm", default=False, type=bool)

    args = parser.parse_args()
    return args
    

def get_activation(name, activation):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def val(args, model, test_loader, test_file, svm):
    model.eval()
    print("Test on Pretrained Models")
    result_list = []

    # calculate logit
    for index, data in enumerate(tqdm(test_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']

        with torch.no_grad():
            logit = model(img)

        result = nn.Sigmoid()(logit).cpu().detach().numpy().tolist()
        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])

def val_svm_logit(args, model, test_loader, train_loader, test_file):
    model.eval()
    print("Test on Pretrained Models")
    result_list = []
    activation = {}
    pre_svm_features =[]
    pre_svm_label=[]
    # calculate the feature for svm
    for index, data in enumerate(tqdm(train_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']
        pre_svm_label.append(data['target'].numpy().tolist())
        model.classifier.multi_head[0].softmax.register_forward_hook(get_activation('feature',activation))
        with torch.no_grad():
            logit = model(img)
            pre_svm_features.append(logit.reshape((1,-1)).cpu().numpy().tolist())

    pre_svm_features = np.array(pre_svm_features).reshape((len(train_loader),-1))
    pre_svm_label = np.array(pre_svm_label).reshape((len(train_loader),-1))
    print('shape of output of pre_svm_features ',pre_svm_features.shape)
    print('shape of output of pre_svm_label ',pre_svm_label.shape)

    #Get svm model
    print('<SVM in training>')
    clf = OneVsRestClassifier(SVC(gamma=0.001,probability= True, C=0.1 ),n_jobs=-1).fit(pre_svm_features, pre_svm_label)
    
    # calculate prediction
    activation = {}
    print('Model predicting')
    post_svm_features = []
    post_svm_label = []
    for index, data in enumerate(tqdm(test_loader)):
        
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']
        post_svm_label.append(data['target'].numpy().tolist())
        model.classifier.multi_head[0].softmax.register_forward_hook(get_activation('feature',activation))

        with torch.no_grad():
            logit = model(img)
            post_svm_features.append(logit.reshape((1,-1)).cpu().numpy().tolist())
        result = clf.predict_proba(logit.reshape((1,-1)).cpu().numpy().tolist())

        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    post_svm_features = np.array(post_svm_features).reshape((len(test_loader),-1))
    post_svm_label = np.array(post_svm_label).reshape((len(test_loader),-1))
    post_svm_label[post_svm_label==-1] = 0
    score = clf.score(post_svm_features,post_svm_label)
    print("Acc : ", score)
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])

def val_svm(args, model, test_loader, train_loader, test_file):
    model.eval()
    print("Test on Pretrained Models")
    result_list = []
    activation = {}
    pre_svm_features =[]
    pre_svm_label=[]
    # calculate the feature for svm
    for index, data in enumerate(tqdm(train_loader)):
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']
        pre_svm_label.append(data['target'].numpy().tolist())
        model.classifier.multi_head[0].softmax.register_forward_hook(get_activation('feature',activation))
        with torch.no_grad():
            logit = model(img)
            pre_svm_features.append(activation['feature'].reshape((1,-1)).cpu().numpy().tolist())

    pre_svm_features = np.array(pre_svm_features).reshape((len(train_loader),-1))
    pre_svm_label = np.array(pre_svm_label).reshape((len(train_loader),-1))
    print('shape of output of pre_svm_features ',pre_svm_features.shape)
    print('shape of output of pre_svm_label ',pre_svm_label.shape)

    #Get svm model
    print('<SVM in training>')
    clf = OneVsRestClassifier(SVC(gamma=1,probability= True),n_jobs=-1).fit(pre_svm_features, pre_svm_label)
    
    # calculate prediction
    activation = {}
    print('Model predicting')
    post_svm_features = []
    post_svm_label = []
    for index, data in enumerate(tqdm(test_loader)):
        
        img = data['img'].cuda()
        target = data['target'].cuda()
        img_path = data['img_path']
        post_svm_label.append(data['target'].numpy().tolist())
        model.classifier.multi_head[0].softmax.register_forward_hook(get_activation('feature',activation))

        with torch.no_grad():
            logit = model(img)
            post_svm_features.append(activation['feature'].reshape((1,-1)).cpu().numpy().tolist())
        result = clf.predict_proba(post_svm_features[index])

        for k in range(len(img_path)):
            result_list.append(
                {
                    "file_name": img_path[k].split("/")[-1].split(".")[0],
                    "scores": result[k]
                }
            )
    post_svm_features = np.array(post_svm_features).reshape((len(test_loader),-1))
    post_svm_label = np.array(post_svm_label).reshape((len(test_loader),-1))
    post_svm_label[post_svm_label==-1] = 0
    score = clf.score(post_svm_features,post_svm_label)
    print("Acc : ", score)
    # cal_mAP OP OR
    evaluation(result=result_list, types=args.dataset, ann_path=test_file[0])

def main():
    args = Args()

    # model 
    if args.model == "resnet101": 
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix)
    if args.model == "vit_B16_224":
        model = VIT_B16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "vit_L16_224":
        model = VIT_L16_224_CSRA(cls_num_heads=args.num_heads, lam=args.lam, cls_num_cls=args.num_cls)
    if args.model == "resnet18":
        model = ResNet(num_classes=args.num_cls, depth=18)
    if args.model == "resnet34":
        model = ResNet(num_classes=args.num_cls, depth=34)
    if args.model == "resnet18_csra":
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix, depth=18, input_dim=512)
    if args.model == "resnet34_csra":
        model = ResNet_CSRA(num_heads=args.num_heads, lam=args.lam, num_classes=args.num_cls, cutmix=args.cutmix, depth=34, input_dim=512)

    model.cuda()
    print("Loading weights from {}".format(args.load_from))
    if torch.cuda.device_count() > 1:
        print("lets use {} GPUs.".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        model.module.load_state_dict(torch.load(args.load_from))
    else:
        model.load_state_dict(torch.load(args.load_from))

    # data
    if args.dataset == "voc07":
        test_file = ['data/voc07/test_voc07.json']
        train_file = ["data/voc07/trainval_voc07.json"]
    if args.dataset == "coco":
        test_file = ['data/coco/val_coco2014.json']
        train_file = ['data/coco/train_coco2014.json']
    if args.dataset == "wider":
        test_file = ['data/wider/test_wider.json']
        train_file = ['data/wider/trainval_wider.json']


    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    if args.svm:
        val_svm_logit(args, model, test_loader, train_loader, test_file)
    elif args.svm:
        val(args, model, test_loader, train_loader, test_file)


if __name__ == "__main__":
    main()
