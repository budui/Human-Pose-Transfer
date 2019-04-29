# Human Pose Transfer for ReID

## Implemented paper

- [x] Pose Guided Person Image Generation (NIPS2017)
- [ ] Disentangled Person Image Generation (CVPR2018 Oral)
- [ ] Progressive Pose Attention Transfer for Person Image Generation (CVPR2019 Oral)

## Prepare

### Download data

1. download Market1501 dataset
2. download 18 key points pose data from here
3. download train and test pair from here

### Generate Pose 18-channels image and corresponding mask

1. `mv ./market-annotation-<type>.csv data/market/annotation-<type>.csv`
2. `python3 tool/generate_pose_map.py_add_mask.py`

## Train

1. use `python3 train.py -h` to see support train process.
2. use `python3 train.py --name <TrainName> -h` to see train option for <TrainName>.
3. use `python3 train.py --name <TrainName>` to train.


## Test 

`python3 test.py`