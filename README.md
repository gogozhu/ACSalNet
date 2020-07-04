# Attentive and Context-aware Network for Saliency Prediction (ACSalNet)
## Preparation


First of all, clone the code
```
git clone https://github.com/gogozhu/ACSalNet.git
```

Then, create a folder:
```
cd ACSalNet && mkdir data
```

### prerequisites

* Python 3.6
* Pytorch 0.4.1
* CUDA 9
* matlab

### Data Preparation

Please prepare the required datasets in advance, including SALICON, MIT1003, MIT300, Silent360, etc.

Then, before running the following commands, please check and modify the dataset's path in the relevant script by yourself.

```
python tools/prepare_data.py
python tools/prepare_data_mit1003.py
cd matlab_script
matlab -nodesktop -nosplash -logfile matlab_`date +%Y_%m_%d-%H_%M_%S`.log -r preprocessing_trainingdata
matlab -nodesktop -nosplash -logfile matlab_`date +%Y_%m_%d-%H_%M_%S`.log -r preprocessing_global_test_data
```

### Compilation

Compile the cuda dependencies using following simple commands:

```
cd lib/models/dcn
python setup.py develop
```

## Train

The configuration file is placed under the "experiments" folder. Modify them according to experimental requirements.

### Train SALICON

In "ACSalNet" floder, run the following simple commands:

```
python tools/train.py --cfg experiments/salicon17/ACSalNet.yaml --m "exp mentions"
```

### Train MIT1003

To train the model on MIT1003, please place the pre-trained model's path under the "PRETRAINED" key in "mit1003_ACSalNet.yaml" file, and then simply run commands:

```
python tools/train.py --cfg experiments/salicon17/mit1003_ACSalNet.yaml --m "exp mentions"
```

### Train Silent360

To train the model on Silent360, please place the pre-trained model's path under the "PRETRAINED" key in "ACSalNet4_s360_cube_train.yaml" file, and then simply run commands:

```
python tools/train.py --cfg experiments/salicon17/ACSalNet4_s360_cube_train.yaml --m "exp mentions"
```

## Validation

The command format is as follows:

```
python tools/test.py --cfg [experiments' yaml file] --s [the index of the session] --mt best --mode val
```

## Test

The command format is as follows:

```
python tools/test.py --cfg [experiments' yaml file] --s [the index of the session] --mt best --mode test
```

**Note**: When testing Silent360 data, you need to replace the content of the "ACSalNet4_s360_cube_train.yaml" file with the content of the "ACSalNet4_s360_global_test.yaml" file, and then run the above command.