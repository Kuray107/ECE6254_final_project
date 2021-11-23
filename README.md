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
   ./run.sh --stage 0 --stop_stage 0 --dataset coswara ## preprocess coswara dataset, or
   ./run.sh --stage 0 --stop_stage 0 --dataset coughvid ## preprocess coughvid dataset
   ```

