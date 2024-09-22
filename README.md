# Uncertainty Quantification via Deep Ensemble and MC-Dropout

This repository contains my personal practice code for studying Bayesian deep learning, with a focus on uncertainty quantification. I referenced the paper [*Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*](https://arxiv.org/abs/1612.01474) and its accompanying [code and dataset](https://github.com/Kyushik/Predictive-Uncertainty-Estimation-using-Deep-Ensemble). While the original implementation was in TensorFlow, I have re-implemented the methods in **PyTorch**.

## Main Python Scripts

| Python File Name           | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `DEnsembleClass.py`        | Trains a single model and a Deep Ensemble on the MNIST dataset |
| `DE-unc-basd-ood.py`       | Performs uncertainty evaluation on OOD tests, comparing single models and ensembles |
| `DE-single-picture.py`     | Tests and compares single models and ensembles on randomly chosen images (in the `./pic/` folder) |
| `MCDrop-Class.py`          | Trains a single MC-Dropout model on the MNIST dataset        |
| `MCDrop-unc-basd-ood.py`   | Performs uncertainty evaluation on OOD tests with MC-Dropout models |
| `MCDrop-single-picture.py` | Tests and compares MC-Dropout models on the same chosen images (in the `./pic/` folder) |

**Note.**
Since the MNIST dataset consists of black digits on a white background, while Omniglot has white digits on a black background, we inverted the colors of the Omniglot dataset to improve the reliability of the OOD tests.

## Results

Several result plots are available in the `/DeepEnsemble Result/` and `/MCDropout Result/` directories.

**Notes on Performance.**
For better results, especially with the MC-Dropout model, you might want to use more complex models and train for more epochs. Under the current settings, the MC-Dropout results are not as strong as expected.


### Deep Ensemble

1. OOD Test and Evaluations
Uncertainty-based OOD Detection AUROC (Total Uncertainty, Omniglot): 0.9878, AUPR: 0.9898
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, Omniglot): 0.9881, AUPR: 0.9897
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, Omniglot): 0.9825, AUPR: 0.9825
Uncertainty-based OOD Detection AUROC (Total Uncertainty, EMNIST): 0.9631, AUPR: 0.9806
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, EMNIST): 0.9619, AUPR: 0.9791
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, EMNIST): 0.9598, AUPR: 0.9778
Uncertainty-based OOD Detection AUROC (Total Uncertainty, KMNIST): 0.9891, AUPR: 0.9878
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, KMNIST): 0.9859, AUPR: 0.9824
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, KMNIST): 0.9876, AUPR: 0.9859

2. Single Image Test with randomly chosen pictures

Single Model predictions:
1th sample (G.png) with Single Model: label = 3, probability = 0.9983
2th sample (E.png) with Single Model: label = 8, probability = 0.5474
3th sample (J.png) with Single Model: label = 1, probability = 0.9992
4th sample (D.png) with Single Model: label = 0, probability = 0.9977
5th sample (1F.png) with Single Model: label = 4, probability = 0.6319
6th sample (6I.png) with Single Model: label = 1, probability = 0.9997
7th sample (2H.png) with Single Model: label = 7, probability = 0.4463
8th sample (C.png) with Single Model: label = 1, probability = 0.9913
9th sample (3B.png) with Single Model: label = 8, probability = 0.9616
10th sample (A.png) with Single Model: label = 4, probability = 0.7672

Ensemble Model predictions:
1th sample (G.png) with Ensemble Model: label = 3, probability = 0.5188
2th sample (E.png) with Ensemble Model: label = 8, probability = 0.7905
3th sample (J.png) with Ensemble Model: label = 1, probability = 0.9974
4th sample (D.png) with Ensemble Model: label = 0, probability = 0.5697
5th sample (1F.png) with Ensemble Model: label = 4, probability = 0.8692
6th sample (6I.png) with Ensemble Model: label = 1, probability = 0.9995
7th sample (2H.png) with Ensemble Model: label = 4, probability = 0.3324
8th sample (C.png) with Ensemble Model: label = 1, probability = 0.6432
9th sample (3B.png) with Ensemble Model: label = 4, probability = 0.5934
10th sample (A.png) with Ensemble Model: label = 4, probability = 0.9511





### MC Drop-out
1. OOD Test

Uncertainty-based OOD Detection AUROC (Total Uncertainty, Omniglot): 0.9602, AUPR: 0.9686
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, Omniglot): 0.9632, AUPR: 0.9718
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, Omniglot): 0.9465, AUPR: 0.9432
Uncertainty-based OOD Detection AUROC (Total Uncertainty, EMNIST): 0.9283, AUPR: 0.9613
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, EMNIST): 0.9275, AUPR: 0.9600
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, EMNIST): 0.9242, AUPR: 0.9576
Uncertainty-based OOD Detection AUROC (Total Uncertainty, KMNIST): 0.9562, AUPR: 0.9551
Uncertainty-based OOD Detection AUROC (Aleatoric Uncertainty, KMNIST): 0.9548, AUPR: 0.9526
Uncertainty-based OOD Detection AUROC (Epistemic Uncertainty, KMNIST): 0.9511, AUPR: 0.9452

2. Single Image Test with randomly chosen pictures

MC-Dropout Model predictions:
1th sample (G.png) with MC-Dropout Model: label = 5, probability = 0.3144
2th sample (E.png) with MC-Dropout Model: label = 8, probability = 0.6371
3th sample (J.png) with MC-Dropout Model: label = 1, probability = 0.9831
4th sample (D.png) with MC-Dropout Model: label = 4, probability = 0.5639
5th sample (1F.png) with MC-Dropout Model: label = 4, probability = 0.9991
6th sample (6I.png) with MC-Dropout Model: label = 1, probability = 0.8647
7th sample (2H.png) with MC-Dropout Model: label = 4, probability = 0.9640
8th sample (C.png) with MC-Dropout Model: label = 1, probability = 0.4944
9th sample (3B.png) with MC-Dropout Model: label = 4, probability = 0.9990
10th sample (A.png) with MC-Dropout Model: label = 4, probability = 0.9999