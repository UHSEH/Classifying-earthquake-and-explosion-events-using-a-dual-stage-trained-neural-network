Our evaluation on a balanced, chronologically partitioned test set from the PNSN ComCat catalog demonstrates that the event-level model achieves 96.41% accuracy and an F1 score of 0.9635, significantly reducing false positives and negatives compared to trace-level baselines. Moreover, the model exhibits strong generalization to out-of-distribution data, including surface military explosions in Ukraine and quarry blasts in Utah, maintaining accuracies above 85% without retraining.
Installation Instructions
Environment Dependencies
Python 3.8+
PyTorch 
ObsPy (Seismic data processing)
NumPy, SciPy, Pandas
Matplotlib, Seaborn (Visualization)
Usage Instructions：Due to the large size of the data file, it cannot be uploaded to GitHub, so users need to download the data themselves. First, run the data code to download the data, then run the dataset code to split the dataset. Finally, save the model as well and run 'The original training code for the model'; if the model already exists, the training part will be skipped and testing will be performed directly.
