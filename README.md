<!-- PROJECT LOGO -->
# Towards Continual Adaptation of Motor Imagery BCIs using Domain-Incremental Learning
<p align="center">
<image src='ca4mi.jpg' width='80%' align="center">
</p>

<!-- ABOUT THE PROJECT -->
## Abstract
Non-invasive electroencephalography (EEG) brain-computer interfaces (BCIs) enable intuitive human-machine interaction and automation in industrial environments. However, adapting to new users remains a major challenge due to inter-subject variability. Conventional EEG decoding methods rely on pre-training a general model on a centralized dataset, but accessing past data is often infeasible due to memory and privacy constraints. Moreover, these methods struggle with catastrophic forgetting (CF) when adapting to a continuous stream of new users, making it difficult to retain previously learned knowledge while integrating new subjects. To address this, we introduce Subject-Incremental Continual Adaptation (SI-CA), a practical continual learning paradigm for EEG-based BCIs, where new subjects arrive incrementally, and previously seen data is inaccessible. Our proposed solution, CA4MI combines an extensible model architecture with a regularization-based forgetting prevention strategy. By employing adversarial feature extraction, it decomposes compact subject-specific representations, facilitating more effective adaptation. This approach enables model expansion through a lightweight modular structure while leveraging adversarial training and orthogonal constraints to maintain efficiency and prevent interference. Additionally, it incorporates prototype-guided regularization with prototypes replay, reducing reliance on stored raw data while mitigating catastrophic forgetting via consistency regularization. Furthermore, Incremental Euclidean Alignment (IEA) and data augmentation facilitate domain adaptation while maintaining computational efficiency. 
Unlike traditional methods requiring extensive pre-training, CA4MI enables on-the-fly adaptation using only the subject’s own data, allowing effective learning from sequentially arriving users while preserving knowledge of previously encountered subjects. Extensive experiments on the BCI benchmark have verified the effectiveness of our method in the SI-CA task, demonstrating its potential for enabling stable, lifelong neural interfaces in industrial environments.

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



