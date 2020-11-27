# MEBOW
### Human Body Orientation Estimation
## Introduction
This is an official pytorch implementation of [*MEBOW: Monocular Estimation of Body Orientation In the Wild*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_MEBOW_Monocular_Estimation_of_Body_Orientation_in_the_Wild_CVPR_2020_paper.pdf). 
In this work, we present COCO-MEBOW (Monocular Estimation of Body Orientation in the Wild), a new large-scale dataset for orientation estimation from a single in-the-wild image. Based on COCO-MEBOW, we established a simple baseline model for human body orientation estimation. This repo provides the code.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
5. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

6. Download pretrained models from the model zoo provided by [HRnet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch)([GoogleDrive](https://drive.google.com/drive/folders/1hOTihvbyIxsm5ygDpbUuJ7O_tzv4oXjC?usp=sharing) or [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blW231MH2krnmLq5kkQ))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pose_hrnet_w32_256x192.pth
   ```
   
### Data preparation
**For MEBOW dataset**, please download images, bbox and keypoints annotation from [COCO download](http://cocodataset.org/#download). Please email <czw390@psu.edu> to get access to human body orientation annotation.
Put them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- train_hoe.json
        |   |-- val_hoe.json
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```
**For TUD dataset**, in progress...

### Training and Testing

#### Traing on MEBOW dataset
```
python tools/train.py --cfg experiments/coco/segm-4_lr1e-3.yaml
```
### Acknowledgement
This repo is based on [HRnet](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch).

### Citation
If you use our dataset or models in your research, please cite with:
```
@inproceedings{wu2020mebow,
  title={MEBOW: Monocular Estimation of Body Orientation In the Wild},
  author={Wu, Chenyan and Chen, Yukun and Luo, Jiajia and Su, Che-Chun and Dawane, Anuja and Hanzra, Bikramjot and Deng, Zhuo and Liu, Bilan and Wang, James Z and Kuo, Cheng-hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3451--3461},
  year={2020}
}
```
