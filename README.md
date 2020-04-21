# CMPT726-CJW - Kaggle IEEE-CIS Fraud detection challenge

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
