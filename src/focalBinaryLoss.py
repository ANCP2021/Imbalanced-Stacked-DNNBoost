import numpy as np

class FocalBinaryLoss():
    def __init__(self, focal_gamma:float) -> None:
        """
        FocalBinaryLoss is a class used for the focal loss function. The class is used in 
        imbalancedXGBoost.py by the ImbalancedXGBoost class.

        Args:
            focal_gamma: The gamma value for focal loss function specified by the user.

        Return:
            None
        """
        self.focal_gamma = focal_gamma

    def focalLoss(self, pred, dtrain):
        """
        Function used for the focal loss calculation.

        Args:
            dtrain: The training dataset output of the DMatrix() function (XGBoost's data matrix).
            pred: Predictions made by the classifier.
        
        Return:
            gradient: First-order derivative.
            hessian: Second-order derivative.
        """
        # User assigned focal_gamma passed in
        focal_gamma = self.focal_gamma
        # Sigmoid is selected as activation with the basic property used in the derivatives
        sigmoid = 1 / (1 + np.exp(-pred))

        # Simplification of equations using short-and variables described
        n_1 = sigmoid * (1 - sigmoid) # yhat_i*(1-yhat_i)
        n_2 = dtrain.get_label() + ((-1) ** dtrain.get_label()) * sigmoid # y_i + (-1)^y_i * yhat_i
        n_3 = sigmoid + dtrain.get_label() - 1 # yhat_i + y_i + 1
        n_4 = 1 - dtrain.get_label() - ((-1) ** dtrain.get_label()) * sigmoid # 1 - y_i - (-1)^y_i * yhat_i

        # First-order derivative (Gradient)
        # gamma * n3 * n2^gamma * log(n4) + (-1)^y_i * n2^(gamma + 1)
        gradient = focal_gamma * n_3 * (n_2 ** focal_gamma) * np.log(n_4 + 1e-9) + \
               ((-1) ** dtrain.get_label()) * (n_2 ** (focal_gamma + 1))
            
        # Second-order derivative (Hessian)
        # n1 * {gamma * [(n2^gamma + gamma * (-1)^y_i * n3 * n2^(gamma - 1)) * log(n4) - ((-1)^y_i * n3 * n2^gamma) / n4] + ((gamma + 1) * n2^gamma)}
        hessian = n_1 * ( \
                    focal_gamma * ( \
                        ( ((n_2 ** focal_gamma) + focal_gamma * ((-1) ** dtrain.get_label()) * n_3 * (n_2 ** (focal_gamma-1))) * np.log(n_4) ) - \
                            ( ( ((-1) ** dtrain.get_label()) * n_3 * (n_2 ** focal_gamma) ) / n_4 ) \
                    ) + \
                    ((focal_gamma + 1) * (n_2 ** focal_gamma)) \
                )

        return gradient, hessian