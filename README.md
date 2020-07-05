# hyperIQA
Pytorch version of the CVPR 2020 paper: "[Blindly Assess Image Quality in the Wild Guided by A Self-Adaptive Hyper Network](https://openaccess.thecvf.com/content_CVPR_2020/html/Su_Blindly_Assess_Image_Quality_in_the_Wild_Guided_by_a_CVPR_2020_paper.html)"

# Note
1. Since the author of this paper has not yet released the code to the [GitHub](https://github.com/SSL92/hyperIQA)(the code is still empty there). I implemented the network according to my understanding. As the details of the network are not precise, I have made some changes in the network.
2. I didn't follow the training process in the original paper, and we can tell from the Implementation Details that "We randomly sample and horizontally flipping 25 patches with size 224x224 pixels from each training image for augmentation.", "25 patches with 224x224 pixels from test image are randomly sampled and their corresponding prediction scores are average pooled to get the final quality score.". During the training progress in LIVE Challenge, I only randomly sample and horizontally flipping patches once from each training image, and resize the image to 224x224 pixels during the testing progress. During the training and testing progress in LIVE, I crop the patches of each image, and stride I set is 80 pixels, I calculate the average score of each patch during the testing progress.
3. Once the author of the paper publishes the [code](https://github.com/SSL92/hyperIQA), I will adjust it as soon as possible.
4. The dataset mat file I used is from LiDingQuan, and you can find it from here [LiDingQuan](https://github.com/lidq92/CNNIQA).

# Train
`python train.py`
