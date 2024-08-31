import argparse
import data
import pandas as pd
from imbalancedXGBoost import ImbalancedXGBoost
from deepNeuralNetwork import ImbalancedDeepNeuralNetwork
from imbalancedDNNBoost import ImbalancedDNNBoost
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    """
    Permits Imbalanced-XGBoost, Imbalanced-DNN and Imbalanced-DNNBoost to be trained from the command line. 
    Intakes the following possible arguments:

    Command Line Args:
        data: Which dataset to use. Assumed that the only datasets fed into each model are those used in the project.
              Consisting of: [heart, sonar, diabetes, parkinson]
        training_rounds: Number of training rounds the chosen model is supposed to run. For XGBoost they are called rounds,
                         but for nerual networks, they are referred to as epochs.
        --use-xgboost-model: The use of the XGBoost model with the specified parameters of the loss function and training rounds.
        --use-dnn-model: The use of the DNN model with the specified parameters of the loss function and epochs.
        --use-dnn-boost-model: The use of the DDN-Boost model with the specified parameters of the two loss functions.

    Returns:
        None
    """
    # Set up the argparse arguments.
    parser = argparse.ArgumentParser(description='Runs a model sepcified by the user and the arguements.')
    parser.add_argument('data', metavar='DATA', type=str, help='Usage of dataset.')
    parser.add_argument('training_rounds', metavar='ROUNDS', type=int, help='Number training rounds of specified algorithm')
    parser.add_argument('--use-xgboost-model', dest='xgboost', help='Uses the XGBoost model.')
    parser.add_argument('--use-dnn-model', dest='dnn', help='Uses the Deep Neural Network model.')
    parser.add_argument('--use-dnn-boost-model', dest='dnnboost', help='Uses the combination of DNN and XGBoost models.')
    parser.set_defaults(xgboost=False, dnn=False, dnnboost=False)
    args = parser.parse_args()

    # Checks if the correct dataset type was entered
    # Each statement uses functions from the data method 
    # specified to the chosen dataset
    if args.data == 'heart':
        X, y = data.get_heart_failure()
    elif args.data == 'sonar':
        X, y = data.get_sonar_mines()
    elif args.data == 'diabetes':
        X, y = data.get_pima_diabetes()
    elif args.data == 'parkinson':
        df = pd.read_csv("./../../440-project-data/parkinson/pd_speech_features.csv")
        X = df.iloc[:,:754]
        y = df['class']
        S = StandardScaler()
        X = S.fit_transform(X)
    else:
        parser.error("Must input a valid dataset. [heart, sonar, diabetes, parkinson]")

    # Split chosen dataset into suubsets using cv_split from data.py
    X_train, y_train, X_test, y_test = data.cv_split(X=X, y=y, folds=5, stratified=True)[0]

    # Checks to see if th specified training rounds are larger than 10
    if args.training_rounds < 10:
        parser.error("Number of training rounds cannot be less than 10.")

    # Checks to see which model the user chooses if the variable is true
    if args.xgboost != False:
        if args.xgboost == 'weighted': # weighted function
            model = ImbalancedXGBoost(objective='weighted', weighted_alpha=0.4, n_estimators=args.training_rounds)
        elif args.xgboost == 'focal': # focal function
            model = ImbalancedXGBoost(objective='focal', focal_gamma=2.0, n_estimators=args.training_rounds)
        elif args.xgboost == 'default': # default paramter set for XGBoost (binary:logistic)
            model = ImbalancedXGBoost(n_estimators=args.training_rounds)
        else:
            parser.error("Unrecognized arguement for imbalanced deep neural network. [weighted, focal, default]")

        # Fit the model
        model.fit(X_train, y_train)
        # Find predictions and use them to compute the accuracy of the model
        predictions = model.predict(X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model Accuracy: \t{accuracy*100:.2f}%')

    elif args.dnn != False:
        if args.dnn == 'weighted': # weighted function
            model = ImbalancedDeepNeuralNetwork(loss='weighted', weighted_alpha=0.4, epochs=args.training_rounds)
        elif args.dnn == 'focal': # focal function
            model = ImbalancedDeepNeuralNetwork(loss='focal', focal_gamma=2.0, epochs=args.training_rounds)
        elif args.dnn == 'default': # default paramter set for DNN (binary_crossentropy)
            model = ImbalancedDeepNeuralNetwork(epochs=args.training_rounds)
        else:
            parser.error("Unrecognized arguement for imbalanced deep neural network. [weighted, focal, default]")

        # Fit and compile the model based on the loss function
        model.compile_fit(X_train, y_train)

    elif args.dnnboost != False:
        if args.dnnboost == 'default|default': # default for both xgboost and dnn
            model = ImbalancedDNNBoost(epochs=args.training_rounds)
        elif args.dnnboost == 'weighted|weighted': # weighted for xgboost and dnn
            model = ImbalancedDNNBoost(dnn_loss='weighted', dnn_weighted_alpha=0.4, xgb_loss='weighted', xgb_weighted_alpha=0.4)
        elif args.dnnboost == 'focal|focal': # focal for both xgboost and dnn
            model = ImbalancedDNNBoost(dnn_loss='focal', dnn_focal_gamma=2.0, xgb_loss='focal', xgb_focal_gamma=0.4)
        elif args.dnnboost == 'default|weighted': # defualt for dnn and weighted for xgboost
            model = ImbalancedDNNBoost(xgb_loss='weighted', xgb_weighted_alpha=0.4)
        elif args.dnnboost == 'default|focal': # default for dnn and focal for xgboost
            model = ImbalancedDNNBoost(xgb_loss='focal', xgb_focal_gamma=2.0)
        elif args.dnnboost == 'weighted|default': # weighted for dnn and default for xgboost
            model = ImbalancedDNNBoost(dnn_loss='weighted', dnn_weighted_alpha=0.4)
        elif args.dnnboost == 'focal|default': # focal for dnn and default for xgboost
            model = ImbalancedDNNBoost(dnn_loss='focal', dnn_focal_gamma=2.0)
        elif args.dnnboost == 'weighted|focal': # weighted for dnn and focal for xgboost
            model = ImbalancedDNNBoost(dnn_loss='weighted', dnn_weighted_alpha=0.4, xgb_loss='focal', xgb_focal_gamma=2.0)
        elif args.dnnboost == 'focal|weighted': # focal for dnn and weighted for xgboost
            model = ImbalancedDNNBoost(dnn_loss='focal', dnn_focal_gamma=2.0, xgb_loss='weighted', xgb_weighted_alpha=0.4)
        else:
            parser.error("Unrecognized arguement for imbalanced dnn-boost. [default|default, weighted|weighted, focal|focal, default|weighted, default|focal, weighted|default, focal|default, weighted|focal, focal|weighted]")

        # Fit and make predictions using DNN and XGBoost models
        # Output allows use to calculate accuracy
        predictions = model.combo(X_train, y_train, X_test, y_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f'Model Accuracy: \t{accuracy*100:.2f}%')

    # If any of the models specified aren't among the allowed type
    else:
        parser.error("Unrecognized model. Must choose between: [--use-xgboost-model, --use-dnn-model, --use-dnn-boost-model]")