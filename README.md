# Comparative Study on CSRA with a SVM extension 
The following graph and link are taken for the original CSRA paper:<br>
[Residual Attention: A Simple But Effective Method for Multi-Label Recoginition](https://arxiv.org/abs/2108.02456)<br>

![attention](https://github.com/Kevinz-code/CSRA/blob/master/utils/pipeline.PNG)

### BRSVM extension on CSRA
![attention](https://github.com/This-is-joejoe/Comparative-Study-of-CSRA/blob/master/utils/CSRA_SVM.png)

This package was modified and developed based on the original code for learning purposes. 

## Requirements
- Python 3.9.7
- pytorch 1.11.0
- torchvision 0.12.0
- tqdm 4.63.0, pillow 9.0.1
- scikit-learn 1.0.2

## Dataset ( followed form the original Github)
Only VOC2007 was used, and the following structure is expected:
```
Dataset/
|-- VOCdevkit/
|---- VOC2007/
|------ JPEGImages/
|------ Annotations/
|------ ImageSets/

```
Then directly run the following command to generate json file (for implementation) of these datasets.
```shell
python utils/prepare/prepare_voc.py  --data_path  Dataset/VOCdevkit
```
which will automatically result in annotation json files in *./data/voc07*, *./data/coco* and *./data/wider*

## Demo
We provide prediction demos of our models. The demo images (picked from VCO2007) have already been put into *./utils/demo_images/*, you can simply run demo.py by using our CSRA models pretrained on VOC2007:
```shell
CUDA_VISIBLE_DEVICES=0 python demo.py --model resnet101 --num_heads 1 --lam 0.1 --dataset voc07 --load_from OUR_VOC_PRETRAINED.pth --img_dir utils/demo_images
```
which will output like this:
```shell
utils/demo_images/000001.jpg prediction: dog,person,
utils/demo_images/000004.jpg prediction: car,
utils/demo_images/000002.jpg prediction: train,
...
```


## Validation
Please download the pre-trained model form links proved by the author of the original CSRA paper. 
### origin links
ResNet101 trained on ImageNet with **CutMix** augmentation can be downloaded 
[here](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV).
|Dataset      | Backbone  |   Head nums   |   mAP(%)  |  Resolution     | Download   |
|  ---------- | -------   |  :--------:   | ------ |  :---:          | --------   |
| VOC2007     |ResNet-101 |     1         |  94.7  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=bXcv&id=1cQSRI_DWyKpLa0tvxltoH9rM4IZMIEWJ)   |
| VOC2007     |ResNet-cut |     1         |  95.2  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=otx_&id=1bzSsWhGG-zUNQRMB7rQCuPMqLZjnrzFh)  |
| COCO        |ResNet-101 |     4         |  83.3  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=EWtH&id=1e_WzdVgF_sQc--ubN-DRnGVbbJGSJEZa)   |
| COCO        |ResNet-cut |     6         |  85.6  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=uEcu&id=17FgLUe_vr5sJX6_TT-MPdP5TYYAcVEPF)   |
| COCO        |VIT_L16_224 |     8      |  86.5  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=1Rmm&id=1TTzCpRadhYDwZSEow3OVdrh1TKezWHF_)|
| COCO        |VIT_L16_224* |     8     |  86.9  |  448x448 |[download](https://drive.google.com/u/0/uc?export=download&confirm=xpbJ&id=1zYE88pmWcZfcrdQsP8-9JMo4n_g5pO4l)|
| Wider       |VIT_B16_224|     1         |  89.0  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1qkJgWQ2EOYri8ITLth_wgnR4kEsv0bfj&export=download)   |
| Wider       |VIT_L16_224|     1         |  90.2  |  224x224 |[download](https://drive.google.com/u/0/uc?id=1da8D7UP9cMCgKO0bb1gyRvVqYoZ3Wh7O&export=download)   |
### example run:
For voc2007, run the following validation example:
```shell
set CUDA_VISIBLE_DEVICES=0 & python val.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from MODEL.pth
set CUDA_VISIBLE_DEVICES=0 & python RF.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20  --load_from MODEL.pth

```
for RF.py, --model can be RF or BRRF 

Other variable options:
- --svm activate BRSVM if True
- --Extra_feature activate extra feature filtering if True.

## Training (No changes form original code)
#### VOC2007
You can run either of these two lines below 
```shell
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20
CUDA_VISIBLE_DEVICES=0 python main.py --num_heads 1 --lam 0.1 --dataset voc07 --num_cls 20 --cutmix CutMix_ResNet101.pth
```
Note that the first command uses the Official ResNet-101 backbone while the second command uses the ResNet-101 pretrained on ImageNet with CutMix augmentation
[link](https://drive.google.com/u/0/uc?export=download&confirm=kYfp&id=1T4AxsAO2tszvhn62KFN5kaknBtBZIpDV) (which is supposed to gain better performance).


## Acknowledgement
This extension study was modified and developed by:
 Yunan Zhou( Master Student at Department of System Design Engineering, University of Waterloo, Canada )

CSRA Authors:
Ke Zhu (http://www.lamda.nju.edu.cn/zhuk/)
Jianxin Wu(wujx2001@gmail.com)
Lin Sui (http://www.lamda.nju.edu.cn/suil/)


