## MEBOW
### Monocular Estimation of Body Orientation In the Wild
### Overview
Body orientation estimation provides crucial visual cues in many applications, including robotics and autonomous driving. It is particularly desirable when 3D pose estimation is difficult to infer due to poor image resolution, occlusion, or indistinguishable body parts. 

We present COCO-MEBOW (Monocular Estimation of Body Orientation in the Wild), a new large-scale dataset for orientation estimation from a single in-the-wild image. The body-orientation labels for 133380 human bodies within 55K images from the COCO dataset have been collected using an efficient and high-precision annotation pipeline. There are 127844 human instance in training set and 5536 human instance in validation set.

Based on MEBOW, we established a simple baseline model for human body orientation estimation. The code and trained models are available on [Github](https://github.com/ChenyanWu/MEBOW).
### Description
![Image of MEBOW](/images/data_examples.png)
### Downloads
<!-- MEBOW dataset belongs to Amazon Inc. The disclosure of the dataset requires Amazon's approval. It is currently in the final stage of approval. The dataset will be public at around August 12. -->
<font color="#dd0000">MEBOW dataset is only for research purposes. Commercial use is not allowed.</font>
Images of MEBOW all come from COCO dataset. Please download images from <https://cocodataset.org/#download>. Click "2017 Train images [118K/18GB]" and "2017 Val images [5K/1GB]" in the COCO download page.
Please email <czw390@psu.edu> to get access to human body orientation annotation. You will usually get a reply within 24 hours (no more than 72 hours). **Note**: For academic researchers, please use your educational email address. You will directly get access to the dataset via your educational email. For researchers in business companies, please send a formal letter (with company title and signature) to promise that you will not use the dataset for commercial purposes. Sorry for the inconvenience.

### Citation
If you use our code or models in your research, please cite with:
```
@inproceedings{wu2020mebow,
  title={MEBOW: Monocular Estimation of Body Orientation In the Wild},
  author={Wu, Chenyan and Chen, Yukun and Luo, Jiajia and Su, Che-Chun and Dawane, Anuja and Hanzra, Bikramjot and Deng, Zhuo and Liu, Bilan and Wang, James Z and Kuo, Cheng-hao},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3451--3461},
  year={2020}
}
```
