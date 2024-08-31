import numpy as np

class WeightedBinaryCrossEntropy():
    def __init__(self, weighted_alpha:float) -> None:
        """
        WeightedBinaryCrossEntropy is a class used for the weighted cross-entropy loss function. The 
        class is used in imbalancedXGBoost.py by the ImbalancedXGBoost class.

        Args:
            weighted_alpha: The alpha value for weight cross-entropy loss function specified by the user.

        Return:
            None
        """
        self.weighted_alpha = weighted_alpha

    def weightedCrossEntropyLoss(self, pred, dtrain):
        """
        Function used for the weighted cross-entropy loss calculation.

        Args:
            dtrain: The training dataset output of the DMatrix() function (XGBoost's data matrix).
            pred: Predictions made by the classifier.
        
        Return:
            gradient: First-order derivative.
            hessian: Second-order derivative.
        """
        # User assigned weighted_alpha passed in
        weighted_alpha = self.weighted_alpha
        # Sigmoid is selected as activation with the basic property used in the derivatives
        sigmoid = 1 / (1 + np.exp(-pred))
        # Calculation for residual (true_predictions - sigmoid)
        residual = dtrain.get_label() - sigmoid

        # First-order derivative (Gradient)
        # -alpha^y_i * (y_i - yhat_i)
        gradient = -(weighted_alpha ** dtrain.get_label()) * residual
        # Second-order derivative (Hessian)
        # alpha^y_i * (1 - yhat_i) * (yhat_i)
        hessian = (weighted_alpha ** dtrain.get_label()) * (1 - sigmoid) * sigmoid

        return gradient, hessian