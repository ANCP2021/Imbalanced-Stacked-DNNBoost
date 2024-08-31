import numpy as np
from deepNeuralNetwork import ImbalancedDeepNeuralNetwork
from imbalancedXGBoost import ImbalancedXGBoost


class ImbalancedDNNBoost:
    def __init__(self, epochs=10, dnn_loss=None, dnn_weighted_alpha=0.4, dnn_focal_gamma=2.0, 
                 xgb_loss=None, xgb_weighted_alpha=0.4, xgb_focal_gamma=2.0) -> None:
        
        """
        ImbalancedDNNBoost is a class that combines a deep neural network algorithm (imported) with the xgboost algorithm along
        with the custom weighted and focal loss functions. The model works with the networks predictions being input values for 
        the xgboost model. Thinking about the process in steps, the deep neural network is step 1 and the xgboost model is step 2
        to see if the different combinations of parameters results in better accuracy with imbalanced data.

        Args:
            epochs: Total number of iterations of all the training data in one cycle.
            dnn_loss: Loss function of the deep neural network.
            dnn_weighted_alpha: The alpha value for weight cross-entropy loss function tailored to the deep neural network model.
            dnn_focal_gamma: The gamma value for focal loss function tailored to the deep neural network model.
            xgb_loss: Loss function of the xgboost model.
            xgb_weighted_alpha: The alpha value for weight cross-entropy loss function tailored to the xgboost model.
            xgb_focal_gamma: The gamma value for focal loss function tailored to the xgboost model.
        
        Return:
            None
        """
        
        self.epochs = epochs
        self.dnn_loss = dnn_loss
        self.dnn_weighted_alpha = dnn_weighted_alpha
        self.dnn_focal_gamma = dnn_focal_gamma
        self.xgb_loss = xgb_loss
        self.xgb_weighted_alpha = xgb_weighted_alpha
        self.xgb_focal_gamma = xgb_focal_gamma


    def combo(self, X_train, y_train, X_test, y_test):
        """
        The combo method allows for the combination of the deep neural network model and the xgboost model. 
        The goal is to use the DNN model's training predictions, concatinate them with the input of the
        xgboost model to see if it results in a better accuracy.

        Args:
            X_train: The training dataset.
            y_train: The training labels.
            X_test: The testing dataset.
            y_test: The testing labels.

        Return:
            predictions: Output of the concatination of the DNN and XGBoost's .predict functions, giving an array of predictions.
        """
        
        # If the specified loss function is 'weighted'
        if self.dnn_loss == "weighted":
            # Use the weighted loss function defined in the ImbalancedDeepNeuralNetwork
            dnn_model = ImbalancedDeepNeuralNetwork(epochs=self.epochs, loss='weighted', weighted_alpha=self.dnn_weighted_alpha)

        # If the specified loss function is 'focal'
        elif self.dnn_loss == "focal":
            # Use the focal loss function defined in the ImbalancedDeepNeuralNetwork
            dnn_model = ImbalancedDeepNeuralNetwork(epochs=self.epochs, loss='focal', focal_gamma=self.dnn_focal_gamma)
        
        # If there is no specified custom loss function, then use the default (binary_crossentropy)
        else:
            dnn_model = ImbalancedDeepNeuralNetwork(epochs=self.epochs)

        # Compile and fit the model using the trainin data
        dnn_model = dnn_model.compile_fit(X_train, y_train)

        # Use the predict methods for the Neural Networks
        # The two lists allow for the networks predictions to be used
        # as inputs in the XGBoost model
        dnn_model_predictons_train = dnn_model.predict(X_train)
        dnn_model_predictons_test = dnn_model.predict(X_test)

        # Concatinate both the network's training and testing predictions
        # to the trainin data to use for future inputs
        extended_train = np.concatenate((X_train, dnn_model_predictons_train), axis=1)
        extended_test = np.concatenate((X_test, dnn_model_predictons_test), axis=1)

        # If the specified loss function is 'weighted'
        if self.xgb_loss == "weighted":
            # Use the weighted loss function defined in the ImbalancedXGBoost
            xgb_model = ImbalancedXGBoost(objective='weighted', weighted_alpha=self.xgb_weighted_alpha)

        # If the specified loss function is 'focal'
        elif self.xgb_loss == "focal":
            # Use the focal loss function defined in the ImbalancedXGBoost
            xgb_model = ImbalancedXGBoost(objective='focal', focal_gamma=self.xgb_focal_gamma)
        
        # If there is no specified custom loss function, then use the default (binary:logistic)
        else:
            xgb_model = ImbalancedXGBoost()

        # Fit the XGBoost model using the extended training data and the y labels
        xgb_model.fit(extended_train, y_train)

        # Get the predictions using the extended testing dataset and the testing y labels
        predictions = xgb_model.predict(extended_test, y_test)

        return predictions