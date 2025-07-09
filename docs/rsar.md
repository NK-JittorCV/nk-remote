# Using JRS with RSAR
## Data Preparing
Downloading RSAR at https://github.com/zhasion/RSAR , and save to `$RSAR_PATH$` as:
```
$RSAR_PATH$
├── train
|     ├──images
|     └──annfiles
├── val
|     ├──images
|     └──annfiles
└── test
      └──images
      └──annfiles
```
## Data Preprocessing
You need set the `$RSAR_PATH$` and run the following script for preprocessing：
```
python configs/preprocess/rsar_preprocess_config.py
```
