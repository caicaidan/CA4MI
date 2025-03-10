<!-- PROJECT LOGO -->

<!-- ABOUT THE PROJECT -->

<!-- Setup -->

## Setup
 * Set up conda environment with python 3.9, ex: `conda create -n ca4mi python=3.9 anaconda`
 * `conda activate ca4mi`
 * `cd ca4mi/`
 * `pip install -r requirements.txt` 

## Datasets
Download the datasets and place them in the `../Datasets/` directory, ex: `../Datasets/BCICompetition-IV2a`, `../Datasets/BCICompetition-IV2b`, `../Datasets/openBMI/`
#### Public datasets links:
##### *  **BCI Competition IV-2a & BCI Competition IV-2b**: (https://www.bbci.de/competition/iv/)
##### * **OpenBMI**: retrieve from: https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html#moabb.datasets.Lee2019_MI
 
 * Note: The datasets should be in the following format:
   ```sh
    ./Datasets/BCICompetition-IV2a/
    ├── A1.mat
    ├── A3.mat
    ├── A3.mat
    ├── A4.mat
    ├── ...
    ├── A8.mat
    └── A9.mat
    ```
    ```sh
    ./Datasets/BCICompetition-IV2b/
    ├── B1.mat
    ├── B2.mat
    ├── B3.mat
    ├── B4.mat
    ├── ...
    ├── B5.mat
    └── B6.mat
    ```
    ```sh  
    ./Datasets/openBMI/
    ├── s1.mat
    ├── s2.mat
    ├── s3.mat
    ├── s4.mat
    ├── ...
    ├── s53.mat
    └── S54.mat
    ```

## Running
1. Clone this repository.
 
2. Set your dataset paths in the configuration files.

    Set paths to your datasets in ./configs/openBMI.yml, ./configs/bcicomp2a.yml, ./configs/bcicomp2b.yml, respectively.

3. Run the following command to train the model:
   ```sh
   python main.py --config ./configs/bcicomp2a.yml
   ```
   ```sh
   python main.py --config ./configs/bcicomp2b.yml
   ```
   ```sh
   python main.py --config ./configs/openBMI.yml
   ```
   The trained model will be saved in the `./checkpoints/` directory.



