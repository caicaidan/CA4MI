<!-- PROJECT LOGO -->
# Towards Continual Adaptation of Motor Imagery BCIs using Domain-Incremental Learning
<p align="center">
<image src='ca4mi.jpg' width='80%' align="center">
</p>

<!-- ABOUT THE PROJECT -->
## Abstract
Non-invasive electroencephalogram (EEG)-based brain-computer interfaces (BCIs) hold significant potential for enhancing automation and human-machine interaction in industrial applications. However, continual adaptation to new users is challenging due to inter-subject variability and limited extensive data access from privacy restrictions. Traditional transfer learning methods suffer from catastrophic forgetting and often require continuous access to large datasets, which is impractical in industrial settings. In this study, we propose a novel domain-incremental learning framework for subject-incremental continual adaptation task (SI-CA) in EEG-based BCIs. Our method features a dynamically extendable architecture with incremental Euclidean alignment and adversarial training that separates subject-invariant and -specific features, enabling rapid adaptation to new subjects without accessing prior data. We further employ prototype consistency regularization and selective sample replay to minimize forgetting. Experiments on BCI benchmarks demonstrate the effectiveness of our approach in the SI-CA task, paving the way for stable and lifelong neural interfaces in industrial environments.

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



