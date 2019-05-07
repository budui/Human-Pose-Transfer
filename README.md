# Human Pose Transfer for ReID

## Implemented paper

- [x] Pose Guided Person Image Generation (NIPS2017)
- [ ] Disentangled Person Image Generation (CVPR2018 Oral)
- [ ] Progressive Pose Attention Transfer for Person Image Generation (CVPR2019 Oral)

## Prepare

### Requirement

* pytorch **1.0+**
* ignite
* torchvision
* numpy
* scipy
* scikit-image
* pandas
* tqdm

### Download data

we need Market1501 and `market-pairs-train.csv`, `market-pairs-test.csv`, `market-annotation-train.csv`, `market-annotation-train.csv`
provided by [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)

1. download Market1501 dataset from [here](http://www.liangzheng.com.cn/Project/project_reid.html)
2. download 18 key points pose data from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)
3. download train and test pair from [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer#data-preperation)
4. copy&rename above pair and annotation file to `data`
5. download Market1501 attribute from [Market-1501_Attribute](https://github.com/vana77/Market-1501_Attribute)

Finally, `data` folder looks like:

```text
data
├── market
│   ├── annotation-test.csv
│   ├── annotation-train.csv
│   ├── pairs-test.csv
│   ├── pairs-train.csv
│   ├── attribute
│   │   ├── evaluate_market_attribute.m
│   │   ├── gallery_market.mat
│   │   ├── market_attribute.mat
│   │   ├── README.md
│   │   └── sample_image.jpg
│   ├── test # WILL BE GENERATED IN NEXT STEP
│   │   ├── pose_map_image
│   │   └── pose_mask_image
│   └── train # WILL BE GENERATED IN NEXT STEP
│       ├── pose_map_image
│       └── pose_mask_image
```

### Generate Pose 18-channels image and corresponding mask


1. `python3 tool/generate_pose_map_add_mask.py --type train`
1. `python3 tool/generate_pose_map_add_mask.py --type test`


## Train

1. use `python3 train.py -h` to see support train process.
2. use `python3 train.py --name <TrainName> -h` to see train option for `TrainName`.
3. use `python3 train.py --name <TrainName>` to train.


## Test 

1. use `python3 test.py -h` to see support test process.
2. use `python3 test.py --name <TestName> -h` to see test option for `TestName`.
3. use `python3 test.py --name <TestName>` to test.


## Eval

For fair comparisons, I just copy&use the same evaluation codes in previous works Deform, PG2 and PATN 
which used some outdated frameworks, like `Tensorflow 1.4.1`(python3)

I recommend using docker to evaluate the results:

```
docker run -v <project path>:/tmp -w /tmp --runtime=nvidia -it --rm tensorflow/tensorflow:1.4.1-gpu-py3 bash
# now in docker:
$ pip install scikit-image tqdm 
$ python tool/getMetrics_market.py
``` 



## Thanks

[@tengteng95](https://github.com/tengteng95) - [Pose-Transfer](https://github.com/tengteng95/Pose-Transfer) 
for clear code and his paper(Progressive Pose Attention Transfer for Person Image Generation).
