import argparse
from pipeline.dataset import DataSet
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.filters import sobel
import cv2

def Args():
    parser = argparse.ArgumentParser(description="settings")
    # model
    parser.add_argument("--model", default="RF")
    # dataset
    parser.add_argument("--dataset", default="voc07", type=str)
    parser.add_argument("--num_cls", default=20, type=int)
    parser.add_argument("--img_size", default=62, type=int)
    parser.add_argument("--train_aug", default=["randomflip", "resizedcrop","greyscale"], type=list)
    parser.add_argument("--test_aug", default=["resizedcrop","greyscale"], type=list)
    parser.add_argument("--Extra_feature", default=False, type=bool) # extra feature extraction step
    args = parser.parse_args()
    return args
    
def OneVsRestRandomForest():
    clf = OneVsRestClassifier(RandomForestClassifier(random_state=8),n_jobs=-1)
    return clf

def MultiLabelRandomForest():
    clf = RandomForestClassifier(random_state=8)
    return clf

def feature_extractor(dataset):
    x_train = dataset
    image_dataset = pd.DataFrame()
    print("Start feature extraction")
    for image in tqdm(range(x_train.shape[0])):  #iterate through each file 
        #print(image)
        
        df = pd.DataFrame()  #Temporary data frame to capture information for each loop.
        #Reset dataframe to blank after each loop.
        
        input_img = x_train[image, :,:,:]
        img = input_img
    ################################################################
    #START ADDING DATA TO THE DATAFRAME
    #Add feature extractors, e.g. edge detection, smoothing, etc. 
            
         # FEATURE 1 - Pixel values
         
        #Add pixel values to the data frame
        pixel_values = img.reshape(-1)
        df['Pixel_Value'] = pixel_values   #Pixel value itself as a feature
        #df['Image_Name'] = image   #Capture image name as we read multiple images
        
        # FEATURE 2 - Bunch of Gabor filter responses
        
                #Generate Gabor features
        num = 1  #To count numbers up in order to give Gabor features a lable in the data frame
        kernels = []
        for theta in range(2):   #Define number of thetas
            theta = theta / 4. * np.pi
            for sigma in (1, 3):  #Sigma with 1 and 3
                lamda = np.pi/4
                gamma = 0.5
                gabor_label = 'Gabor' + str(num)  #Label Gabor columns as Gabor1, Gabor2, etc.
    #                print(gabor_label)
                ksize=9
                kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lamda, gamma, 0, ktype=cv2.CV_32F)    
                kernels.append(kernel)
                #Now filter the image and add values to a new column 
                fimg = cv2.filter2D(img, cv2.CV_8UC3, kernel)
                filtered_img = fimg.reshape(-1)
                df[gabor_label] = filtered_img  #Labels columns as Gabor1, Gabor2, etc.
                #print(gabor_label, ': theta=', theta, ': sigma=', sigma, ': lamda=', lamda, ': gamma=', gamma)
                num += 1  #Increment for gabor column label
                
         
        # FEATURE 3 Sobel
        edge_sobel = sobel(img)
        edge_sobel1 = edge_sobel.reshape(-1)
        df['Sobel'] = edge_sobel1
       
        #Add more filters as needed
        
        #Append features from current image to the dataset
        image_dataset = image_dataset.append(df)
        #print('numbers of images : ',image)
    image_dataset = np.expand_dims(image_dataset, axis=0)
    image_dataset = np.reshape(image_dataset, (x_train.shape[0], -1))

    return image_dataset

def main():
    args = Args()

    # data
    if args.dataset == "voc07":
        train_file = ["data/voc07/trainval_voc07.json"]
        test_file = ['data/voc07/test_voc07.json']
        step_size = 4
    if args.dataset == "coco":
        train_file = ['data/coco/train_coco2014.json']
        test_file = ['data/coco/val_coco2014.json']
        step_size = 5
    if args.dataset == "wider":
        train_file = ['data/wider/trainval_wider.json']
        test_file = ["data/wider/test_wider.json"]
        step_size = 5
        args.train_aug = ["randomflip"]
        
    train_dataset = DataSet(train_file, args.train_aug, args.img_size, args.dataset)
    test_dataset = DataSet(test_file, args.test_aug, args.img_size, args.dataset)
    train_loader = iter(DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=6))
    test_loader = iter(DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=6))

    print("<preparing dataset> \n")
    #train data prepare
    X = np.zeros((len(train_dataset),args.img_size,args.img_size,3))
    y = np.zeros((len(train_dataset),args.num_cls))
    for i in range(len(train_dataset)):
        data = next(train_loader)
        img, target = data['img'],data['target'].detach().numpy()
        img = img[0].permute(1, 2, 0)
        #plt.show()
        img = img.detach().numpy()
        X[i] = img
        y[i] = target

    #test data prepare
    X_test = np.zeros((len(test_loader),args.img_size,args.img_size,3))
    y_test = np.zeros((len(test_loader),args.num_cls))
    for i in range(len(test_loader)):
        data = next(test_loader)
        img, target = data['img'],data['target'].detach().numpy()
        img = img[0].permute(1, 2, 0)
        #plt.show()
        img = img.detach().numpy()
        X_test[i] = img
        y_test[i] = target
    y_test[y_test == -1] = 0

    print('<Training Models>')
    #model training
    if args.Extra_feature:
        image_features = feature_extractor(X)
        test_image_features = feature_extractor(X_test)
        if args.model == 'RF':
            forest_cla = MultiLabelRandomForest().fit(image_features,y)
            pred_score = forest_cla.score(test_image_features,y_test)
        elif args.model == 'BRRF':
            Binary_relevance_cla = OneVsRestRandomForest().fit(image_features,y)
            pred_score = Binary_relevance_cla.score(test_image_features,y_test)      


    else:
        #use only pixels as feature
        X_pixel = np.reshape(X,(len(X),-1))
        X_test_pixel = np.reshape(X_test,(len(X_test),-1))
        if args.model == 'RF':
            forest_cla = MultiLabelRandomForest().fit(X_pixel,y)
            pred_score = forest_cla.score(X_test_pixel,y_test)
        elif args.model == 'BRRF':
            Binary_relevance_cla = OneVsRestRandomForest().fit(X_pixel,y)
            pred_score = Binary_relevance_cla.score(X_test_pixel,y_test)      


    print(args.model,' Acc: ', pred_score)

if __name__ == "__main__":
    main()
