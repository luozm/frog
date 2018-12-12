# Mask R-CNN with watershed head

## Data
1. Download
    * Download stage1_train and stage2_test data from [here](https://www.kaggle.com/c/data-science-bowl-2018/data)
    * Download stage1_test data from [here](https://github.com/yuanqing811/DSB2018_stage1_test)
2. Unzip data and put them into `/data/__download__`:
    * stage1_train
    * stage1_test
    * stage2_test

## Instruction
1. change `ROOT_DIR` in `/code/common.py`
2. run `make.sh` in `/code/net/lib` to build custom operations
3. run `preprocess.py` to make multi masks for data sets, and normalize images
4. training scripts:
    * `train.py` for training normal mask rcnn
    * `train_depth.py` for training depth mask rcnn
    * `train_rpn.py` for training only RPN
5. training settings (for any specific training scripts)
    * setup training in main function
    * change `out_dir` to setup result folder
    * setup pre-trained model in `pretrain_file` (train from start) or `resume_checkpoint` (resume training)
    * change specific training settings in `# training settings` part in `run_train` function
    * augmentation: change `train_augmen` function in train script
6. weighted loss
    * rewrite weight function to `/code/net/weighted_loss.py`
    * rewrite `make_mask_target` function in `/code/net/layer/mask_target.py` to output corresponding weights for each mask
    * change `mask_loss` function in `/code/net/layer/mask_loss.py` to use `weighted_binary_cross_entropy_with_logits` function, and add `mask_weights` as one input for the function

## Requirements
* Pytorch 0.4 or above
* OpenCV (cv2)
* sklearn
* tqdm
* numpy/scipy/skimage
* matplotlib
* ...

## Citations
* Thanks to [Heng CherKeng](https://www.kaggle.com/c/data-science-bowl-2018/discussion/47686) for his outstanding work!
