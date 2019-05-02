# Human Pose Transfer for ReID

## Implemented paper

- [x] Pose Guided Person Image Generation (NIPS2017)
- [ ] Disentangled Person Image Generation (CVPR2018 Oral)
- [ ] Progressive Pose Attention Transfer for Person Image Generation (CVPR2019 Oral)

## Prepare

### Requirement

* pytorch 1.0
* ignite
* torchvision
* numpy
* scipy
* scikit-image
* pillow
* pandas
* tqdm

### Download data

we need Market1501 and `market-pairs-train.csv`, `market-pairs-test.csv`, `market-annotation-train.csv`, `market-annotation-train.csv`
provided by [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)

1. download Market1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html)
2. download 18 key points pose data from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)
3. download train and test pair from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)
4. copy&rename above pair and annotation file to `data`

Finally, `data` folder looks like:

```text
data
├── market
│   ├── annotation-test.csv # PLEASE USE THE SAME FILENAME
│   ├── annotation-train.csv # PLEASE USE THE SAME FILENAME
│   ├── test # WILL BE GENERATED IN NEXT STEP
│   │   ├── pose_map_image
│   │   └── pose_mask_image
│   └── train # WILL BE GENERATED IN NEXT STEP
│       ├── pose_map_image
│       └── pose_mask_image
├── market-pairs-test.csv
├── market-pairs-train.csv
```

### Generate Pose 18-channels image and corresponding mask


1. `python3 tool/generate_pose_map_add_mask.py --type train`
1. `python3 tool/generate_pose_map_add_mask.py --type test`


## Train

1. use `python3 train.py -h` to see support train process.
2. use `python3 train.py --name <TrainName> -h` to see train option for `TrainName`.
3. use `python3 train.py --name <TrainName>` to train.


## Test 

`python3 test.py`

## Thanks

[@tengteng95](https://github.com/tengteng95) - [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer) 
for clear code and his paper(Progressive Pose Attention Transfer for Person Image Generation).
