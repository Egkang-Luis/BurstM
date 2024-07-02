# BurstM

## Requirement
### Python packages
```
conda env create --file environment.yaml
conda activate BurstM
```

## Track 1 SyntheticBurst dataset
### Training
- Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python BurstM_Track_1_train.py --image_dir=<Input DIR>
```

### Evaluation
```
python BurstM_Track_1_evaluation.py
```
