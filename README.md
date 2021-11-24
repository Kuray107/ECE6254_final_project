## Stage 0: data preprocessing
### Data Download
- First we need to download Coswara and coughbid datasets.
    ```sh
    mkdir datasets
    cd datasets
    git clone https://github.com/iiscleap/Coswara-Data.git # Coswara
    git clone https://github.com/virufy/virufy-cdf-coughvid.git # coughvid from virufy
    ```
    
### Data preparation
- After finishing data download, we will need to do data preparation. 
    ```sh
    ./run.sh --stage 0 --stop_stage 0 --dataset coswara --decompress true ## preprocess coswara dataset, or
    ./run.sh --stage 0 --stop_stage 0 --dataset coughvid ## preprocess coughvid dataset
    ```
- This will generate a json file (stored at datasets/) according to the dataset being specified.
    ```json
    # Coswara dataset
    {
        "iV3Db6t1T8b7c5HQY2TwxIhjbzD3": { # patient ID
            "wav_paths": [ # list of the patient's sound file
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3counting-normal.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-o.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-a.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3vowel-e.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3breathing-shallow.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3counting-fast.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3breathing-deep.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3cough-heavy.wav",
                "datasets/Coswara-Data/20200424/iV3Db6t1T8b7c5HQY2TwxIhjbzD3cough-shallow.wav"
            ],
            "pct_test_result": "untested" # patient's pcr test result
        },
        "AxuYWBN0jFVLINCBqIW5aZmGCdu1": { # another patient ID
            ...
        }
        ...
    }
    ```
- There are 429/1375/424 positive/negative/untested speakers in Coswara dataset.
- There are 588/7129/5400 positive/negative/untested speakers in Coughvid dataset. 

## Stage 1: 
