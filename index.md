## MEBOW
### Monocular Estimation of Body Orientation In the Wild
### Overview
Body orientation estimation provides crucial visual cues in many applications, including robotics and autonomous driving. It is particularly desirable when 3D pose estimation is difficult to infer due to poor image resolution, occlusion, or indistinguishable body parts. 

We present COCO-MEBOW (Monocular Estimation of Body Orientation in the Wild), a new large-scale dataset for orientation estimation from a single in-the-wild image. The body-orientation labels for 133380 human bodies within 55K images from the COCO dataset have been collected using an efficient and high-precision annotation pipeline. There are 127844 human instance in training set and 5536 human instance in validation set.

Based on MEBOW, we established a simple baseline model for human body orientation estimation. The code for baseline model will be available on [Github](https://github.com/ChenyanWu/MEBOW).
### Description
![Image of MEBOW](/images/data_examples.png)
### Downloads
<!-- MEBOW dataset belongs to Amazon Inc. The disclosure of the dataset requires Amazon's approval. It is currently in the final stage of approval. The dataset will be public at around August 12. -->
<font color="#dd0000">MEBOW dataset is only for research purposes. Commercial use is not allowed.</font>
Images of MEBOW all come from COCO dataset. Please download all the images from <https://cocodataset.org/#download>. 
Please email <czw390@psu.edu> to get access to human body orientation annotation. You will usually get a reply within 24 hours (no more than 72 hours).
<!-- [Dataset Download](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/czw390_psu_edu/EpdIRxS3_4hBpo9MvlWSiUcBEwAwjw5QgZ2kKFXH0T5hUw?e=U7wIEO)
[Dataset Readme](/descirption.txt) -->
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






<!-- This is the website for MEBOW: Monocular Estimation of Body Orientation In the Wild
You can use the [editor on GitHub](https://github.com/ChenyanWu/MEBOW/edit/master/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/ChenyanWu/MEBOW/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out. -->
