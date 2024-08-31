# Ensemble Learning: Boosting #

## By Alexander Nemecek ##

## Overview ##
This repository contains a review and exploration into ensemble methods tailored to dealing with imbalanced data. The primary ensemble method of focus is XGBoost, and the implementation called Imbalanced-XGBoost which shows better results in binary classification than normal methods. The file "Alexander_Nemecek_Project_Report.md" contains an exploration into the XGBoost initial algorithm as well as the implementation of Imbalanced-XGBoost. The paper will also discuss possible extensions which allow for better results than the proposed method.

## Running the Code ##
Navigate to the "ajn98-boosting" directory and install the required packages using the "requirements.txt" file.

```console
user@pc:~$ cd /PATH/TO/csds440project-f23-4/ajn98-boosting
user@pc:~$ pip install -r requirements.txt
```

To be able to use the tested datasets in this project, you can download the datasets from the **DataSet Drive** section of this document.

Place this outside the repository, "csds440project-f23-4" and extract it in order for the get_x_data() methods in "src/data.py" to work.

- ./../../440-project-data

Below contains the following four datasets allowed for the user to explore with the ensemble models implemented in this library:

- Heart Failure Prediction
    - https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- Pima Indians Diabetes
    - https://github.com/npradaschnor/Pima-Indians-Diabetes-Dataset
- Sonar Mines
    - https://github.com/ksatola/Sonar-Mines-vs-Rocks/tree/master
- Parkinson
    - https://archive.ics.uci.edu/dataset/174/parkinsons

The ./src/ directory contains the following files:

- **data.py** - A collection of data importers and cleaning operations as well as a cv_split function that can create both stratified and non-stratified folds of data with the added option of bootstrapping.
- **main.py** - A consolidation of the models Imbalanced-XGBoost, Imbalanced-DNN, and Imbalanced-DNN-Boost which allows for the user to interface with the program through the command line.
- **deepNeuraNetwork.py** - Houses a class which uses a Deep Neural Network as the primary model, allowing for custom loss functions as described by C. Wang, C. Deng, and S. Wang.
- **imbalancedXGBoost.py** - Main class implementation of the Imbalanced-XGBoost model proposed by C. Wang, C. Deng, and S. Wang in the paper "Imbalance-XGBoost: Leveraging Weighted and Focal Losses for Binary Label-Imbalanced Classification with XGBoost".
- **focalBinaryLoss.py** - Customized focal loss function class as described above.
- **weigtedBinaryCrossEntropy.py** - Customized weighted cross entropy loss function class as described above.
- **imbalancedNNBoost.py** - Houses a class that combines the Imbalanced-DNN model and the Imbalanced-XGBoost model.

### Running Main ###
In order to run **main.py**, the user needs to "_cd_" to be in the directory "/PATH/TO/csds440project-f23-4/ajn98-boosting/src"

```console
user@pc:~$ cd /PATH/TO/csds440project-f23-4/ajn98-boosting/src
```

The user can then run **main.py** in the following format:

```
user@pc:~$ python main.py [dataset] [training-rounds] [model-flag] [custom-loss-function]
```

The parameters are as follows:
- **dataset** - The allowed datasets are the four mentioned above and must be input as one word, with the options being: [heart, sonar, diabetes, parkinson]. "heart" for the heart disease dataset, "sonar" for the sonar mine dataset, "diabetes" for the pima diabetes dataset, and "parkinson" for the parkinson disease dataset.
- **training-rounds** - Takes any integer that is greater than or equal to 10. This parameter allows for the number of training rounds for the XGBoost model and the number of epochs for the DNN model.
- **model-flag** - There are only three options for this parameter: [--use-xgboost-model, --use-dnn-model, --use-dnn-boost-model]. "--use-xgboost-model" uses only the XGBoost model, "--use-dnn-model" uses only the DNN model, and "--use-dnn-boost-model" uses a combination of the two models, DNN and XGBoost.
- **custom-loss-function** - The "custom-loss-function" parameter is tailored to the "model-flag" parameter. The flag gives the options of choosing "weighted" for the weighted loss function, "focal" for the focal loss function, and "default" for the default loss function specific to the model type. The options per flag are stated below (NOTE: This flag needs to be surrounded by quotation marks the CLI may mix up the '|' character):
    - --use-xgboost-model
        - "weighted"
        - "focal"
        - "default"
    - --use-dnn-model
        - "weighted"
        - "focal"
        - "default"
    - --use-dnn-model (The combination flag is structured by: "dnn-loss|xgboost-loss")
        - "default|default" 
        - "weighted|weighted"
        - "focal|focal"
        - "default|weighted" 
        - "default|focal"
        - "weighted|default" 
        - "focal|default"
        - "weighted|focal" 
        - "focal|weighted"

Example:
```
user@pc:~$ python main.py heart 10 --use-xgboost-model "focal"
```
Expected output would be the accuracy of the model based on the specified parameters.

## DataSet Drive ##
https://drive.google.com/file/d/1LNYoVx44XoPrmqsXhmfslxRWEh4nQCYO/view?usp=sharing

## Sources ##
The following sourses were referenced throughout the duration of the project (not including the main papers). The sources are listed in the following format:
- Author. _Title_. Source Link.
---
- dmlc XGBoost. _Custmo Objective and Evaluation Metric_. https://xgboost.readthedocs.io/en/stable/tutorials/custom_metric_obj.html. 

- dmlc XGBoost. _Python API Reference_. https://xgboost.readthedocs.io/en/stable/python/python_api.html

- dmlc XGBoost. _XGBoost Parameters_. https://xgboost.readthedocs.io/en/stable/parameter.html

- GeeksforGeeks. _XGBoost_. https://www.geeksforgeeks.org/xgboost/.

- Packt. _From Gradient Boosting to XGBoost_. https://dev.to/packt/from-gradient-boosting-to-xgboost-2ba2.

- StatQuest with Josh Starer. _Gradient Boost Part 1 (of 4): Regression Main Ideas_. https://www.youtube.com/watch?v=3CC4N4z3GJc.

- StatQuest with Josh Starer. _Gradient Boost Part 2 (of 4): Regression Details_. https://www.youtube.com/watch?v=2xudPOBz-vs&t=32s.

- StatQuest with Josh Starer. _Gradient Boost Part 3 (of 4): Classification_. https://www.youtube.com/watch?v=jxuNLH5dXCs.

- StatQuest with Josh Starer. _Gradient Boost Part 4 (of 4): Classification Details_. https://www.youtube.com/watch?v=StWY5QWMXCw.

- StatQuest with Josh Starer. _Regularization Part 1: Ridge (L2) Regression_. https://www.youtube.com/watch?v=Q81RR3yKn30.

- StatQuest with Josh Starer. _Regularization Part 2: Lasso (L1) Regression_. https://www.youtube.com/watch?v=NGf0voTMlcs.

- StatQuest with Josh Starer. _XGBoost Part 1 (of 4): Regression_. https://www.youtube.com/watch?v=OtD8wVaFm6E&t=39s.

- StatQuest with Josh Starer. _XGBoost Part 2 (of 4): Classification_. https://www.youtube.com/watch?v=8b1JEDvenQU.

- StatQuest with Josh Starer. _XGBoost Part 3 (of 4): Mathematical Details_. https://www.youtube.com/watch?v=ZVFeW798-2I.

- StatQuest with Josh Starer. _XGBoost Part 4 (of 4): Crazy Cool Optimizations_. https://www.youtube.com/watch?v=oRrKeUCEbq8.