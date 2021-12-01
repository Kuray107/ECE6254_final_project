## Data preprocessing
### Data Download and decompress
- First we need to download Coswara and coughbid datasets.
    ```sh
    mkdir datasets
    cd datasets
    git clone https://github.com/iiscleap/Coswara-Data.git # Coswara
    ./decompress.sh # Decompress the dataset. Only Coswara needs it. 
    git clone https://github.com/virufy/virufy-cdf-coughvid.git # coughvid from virufy
    ```
### Data preparation
- After finishing data download, we will need to do data preparation. 
    ```sh
    python data_preprocessing.py --dataset coswara ## preprocess coswara dataset, or
    python data_preprocessing.py --dataset coughvid ## preprocess coughvid dataset
    ```
- This will extract features of these wav files and generate a json file (stored at datasets/) according to the dataset being specified.
    ```
    # Coswara dataset
    {
        "iV3Db6t1T8b7c5HQY2TwxIhjbzD3": { # patient ID
            "feature_paths": [ # list of the patient's sound file
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3counting-normal.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-o.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-a.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-e.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3breathing-shallow.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3counting-fast.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3breathing-deep.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3cough-heavy.npy",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3cough-shallow.npy"
            ],
            "pct_test_result": "untested" # patient's pcr test result
        },
        "AxuYWBN0jFVLINCBqIW5aZmGCdu1": { # another patient ID
            ...
        }
        ...
    }
    ```
- There are 417/1331/398 positive/negative/untested speakers in Coswara dataset.
- There are 3734/11926/3574 positive/negative/untested speakers in Coswara dataset.
- There are 588/7129/5400 positive/negative/untested speakers in Coughvid dataset. 

## Model training:
- To train the model, use the following command:
    ```sh
    python train.py --dataset <dataset> --semi <bool> --split_type <split_type>
    ```
- Note that we can only choose "default" as the split_type when training on Coughvid dataset.


