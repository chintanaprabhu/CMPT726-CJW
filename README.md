# CMPT726-CJW - Kaggle IEEE-CIS Fraud detection challenge

This project is a collaborated work by Chintana Prabhu - cprabhu@sfu.ca, Jason Lee - jason_lee_19@sfu.ca and  Wesley Romey - wes_romey@sfu.ca as a part of course project towards completion of course CMPT419/726:Machine Learning under the guidance of Dr.Mo Chen in the Spring-2020 term.

We used Google Colab platform to train the model and run the code. The data files train_transaction.csv and test_transaction.csv could not be uploaded in the repository because of file size limit. 
If using Google colab to run the code, the data files could be found at https://drive.google.com/drive/folders/1y6qWfN1syKy7Q2uyQXPNnr_yHyiGLZ1Z?usp=sharing
We need to mount the directory containing the data files prior to proceeding with data read.
The data files are sourced from https://www.kaggle.com/c/ieee-fraud-detection/data

Additional instruction while running the notebook final-code:
1. Set GOOGLE_COLAB to either True or False depending on if you are using Google Colab
2. Change the paths for the data files under the "ieee-fraud-detection" folder to whatever the actual paths are, if applicable
3. See what the values of all the constants should be. If you leave them at their defaults, that is also fine
4. Just run the entire file from start to finish each time you want to run it. It should work perfectly and skip over all the lengthy parts after the first complete run.
5. Save the code as a .tex along with all plots. Also, save the file as .ipynb.
6. Copy and paste "textCNN_submission.csv" to a separate folder for storage of Kaggle test data predictions. Hand this into Kaggle to get an accuracy score.

Contents

* The ieee-fraud-detection folder contains few od the data files sourced from https://www.kaggle.com/c/ieee-fraud-detection/data for the project.
* The notebook final_code.ipynb traverses through the process of reading the data, merging data, cleaning and preprocessing followed by feature extraction and unnecessary feature removal. Some experimental code in comments could also be seen in the notebook (which did not yeild good results). The last part of the notebook contains training and prediction of a CNN network which has been explainined in detail in the final-report.pdf file.
* final_code.py contains the modules of python code used in the notebook final_code.ipynb.
* fianl-report.pdf is the project report submitted towards the completion of the course project. The report follows the guidelines of NIPS-2015 Style

Credits:

1. https://www.kaggle.com/c/ieee-fraud-detection/discussion/100400
2. https://www.kaggle.com/c/ieee-fraud-detection/discussion/99982
3. https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
4. https://keras.io/examples/imdb_cnn/
5. https://glassboxmedicine.com/2019/05/05/how-computers-see-intro-to-convolutional-neural-networks/
6. https://richliao.github.io/supervised/classification/2016/11/26/textclassifier-convolutional/
7. https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
8. https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
9. https://github.com/krishnaik06/Handle-Imbalanced-Dataset/blob/master/Handling%20Imbalanced%20Data-%20Over%20Sampling.ipynb
