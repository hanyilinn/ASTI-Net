# ASTI-Net
## Overview
This repository contains the implementation of ASTI-Net, a novel deep learning-based architecture designed for artifact removal in EEG signals with arbitrary channel settings. ASTI-Net leverages a dual-branch network to capture both spatial and temporal features of EEG signals, enabling effective denoising across varying numbers of EEG channels. The model is particularly useful for applications in neuroscience, brain-computer interfaces, and clinical studies where EEG signals are often contaminated by artifacts such as EOG (electrooculography) and EMG (electromyography).

![ASTI-Net Architecture](pictures/Graphical_abstract.jpg)

## Network Structure

The overall architecture of ASTI-Net is illustrated below. It consists of three components: a multi-channel spatial feature extraction branch (MSFE Branch), a single-channel temporal feature extraction branch (STFE Branch), and a fusion reconstruction module (FR Module). The two branches process the EEG input in different ways: the multi-channel branch takes the entire two-dimensional EEG signal, while the single-channel branch processes each channel independently.

![ASTI-Net Architecture](pictures/Architecture_of_ASTI-Net.jpg)

## Results
![ASTI-Net Architecture](pictures/Result_2s.jpg)

## Environment Requirements

### Python Version
- Python >= 3.8

### Dependencies

We recommend using `pip` or `conda` to install the required dependencies. Below is the list of key packages and their versions:

#### Main Packages

- PyTorch >= 2.3.0
- torchmetrics >= 1.5.1
- torchaudio >= 2.3.0
- torchvision >= 0.18.0
- numpy >= 1.24.3
- pandas >= 2.0.3
- scikit-learn >= 1.3.0
- scipy >= 1.10.1
- matplotlib >= 3.7.2
- seaborn >= 0.13.2
- tqdm >= 4.66.4
- requests >= 2.31.0

## Citation
For more details, you can refer to the [DOI: 10.1109/JBHI.2025.3555813](https://doi.org/10.1109/JBHI.2025.3555813).

