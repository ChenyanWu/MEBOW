# MEBOW
### Human Body Orientation Estimation
## Introduction
This is an official pytorch implementation of [*MEBOW: Monocular Estimation of Body Orientation In the Wild*](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wu_MEBOW_Monocular_Estimation_of_Body_Orientation_in_the_Wild_CVPR_2020_paper.pdf). 
In this work, we present COCO-MEBOW (Monocular Estimation of Body Orientation in the Wild), a new large-scale dataset for orientation estimation from a single in-the-wild image. Based on COCO-MEBOW, we established a simple baseline model for human body orientation estimation. This repo provides the code.

## Quick start
### Installation
1. Install pytorch >= v1.0.0 following [official instruction](https://pytorch.org/).
2. Clone this repo, and we'll call the directory that you cloned as ${HBOE_ROOT}.
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
   ${HBOE_ROOT}
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
   ${HBOE_ROOT}
    `-- models
        `-- pose_hrnet_w32_256x192.pth
   ```
  
### Data preparation
**For MEBOW dataset**, please download images, bboxes and keypoints from [COCO download](http://cocodataset.org/#download). Please email <czw390@psu.edu> to get access to the human body orientation annotations. **Note**: For academic researchers, please use your educational email address. You will directly get access to the annotations via your educational email. For researchers in business companies, please send a formal email (with company title and signature) to promise that you will not use the annotations for commercial purposes. Sorry for the inconvenience.

Put images and all the annotations under {HBOE_ROOT}/data, and make them look like this:
```
${HBOE_ROOT}
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
**For TUD dataset**, please download images from [the web page of TUD](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/people-detection-pose-estimation-and-tracking/monocular-3d-pose-estimation-and-tracking-by-detection). The page also provides 8-bin orientation annotation. Continuous orientation annotation for TUD dataset can be found from [here](http://www.kotahara.com/publications.html). We provide our precessed TUD annotation from [here](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/czw390_psu_edu/EqU8hWh-NgFOoNmIBEgE5RYBn61ZsFudKHCgbEH9-_V9DA?e=PZzshY).
Put TUD images and our processed annotation under {HBOE_ROOT}/data, and make them look like this:
```
${HBOE_ROOT}
|-- data
`-- |-- tud
    `-- |-- annot
        |   |-- train_tud.pkl
        |   |-- val_tud.pkl
        |   `-- test_tud.pkl
        `-- images
            |-- train
            |-- validate
            `-- test
```
### Trained HBOE model
We also provide the trained HBOE model (MEBOW as training set). ([OneDrive](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/czw390_psu_edu/EoXLPTeNqHlCg7DgVvmRrDgB_DpkEupEUrrGATpUdvF6oQ?e=CQQ2KY))
### Training and Testing

#### Training on MEBOW dataset
```
python tools/train.py --cfg experiments/coco/segm-4_lr1e-3.yaml
```
#### Training on TUD dataset
```
python tools/train.py --cfg experiments/tud/lr1e-3.yaml
```
#### Testing on MEBOW dataset
You should change TEST:MODEL_FILE to your own in "experiments/coco/segm-4_lr1e-3.yaml". If you want to test with our trained HBOE model, specify TEST:MODEL_FILE with the downloaded model path.
```
python tools/test.py --cfg experiments/coco/segm-4_lr1e-3.yaml
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
