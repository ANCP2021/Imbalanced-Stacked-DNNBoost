from weightedBinaryCrossEntropy import WeightedBinaryCrossEntropy
from focalBinaryLoss import FocalBinaryLoss
from sklearn.model_selection import train_test_split
import xgboost as xgb

class ImbalancedXGBoost():
    def __init__(self, n_estimators:int=10, max_depth:int=10, max_leaves:int=0, eta_learning_rate:float=0.1, 
                 verbosity:int=0, objective:str='binary:logistic', booster:str='gbtree', gamma:float= 0.0, 
                 min_child_weight:float=1.0, early_stopping_rounds:int=None, eval_metric:str='logloss', 
                 l1_reg_alpha:float=0.0, l2_reg_lambda:float=1.0, focal_gamma:float=None, weighted_alpha:float=None) -> None:
        
        """
        ImbalancedXGBoost is a class that combines the XGBoost algorithm (imported) with weighted and focal loss functions. The
        goal for the new loss functions provided is to handle label-imbalanced scenarios.

        Args:
            n_estimators: Number of boosting rounds; default=10.
            max_depth: Maximum tree depth for base learners; default=10.
            max_leaves: Maximum number of leaves; default=0 (indicating no limit).
            eta_learning_rate: Boosting learning rate also known as "eta"; defualt=0.1.
            verbosity: The degree of verbosity; Valid values are 0(silent) - 3(debug); default=0.
            objective: Specification of learning task and the corresponding learning objective or custom function used; 
                       defualt='binary:logistic'; there are many options allowed for this parameter;
                       'binary:logitraw' stands for logistic regression for binary classification, output score before
                       logistic transformation.
            booster: Specify which booster to use: gbtree, gblinear, r dart; default='gbtree'.
            gamma: Minimum loss reduction requiured to make a further parition on a leaf node of the tree; default=0.0.
            min_child_weight: Minimum sum of instance weight(hessian) needed in a child; default=1.0.
            early_stopping_rounds: Activates early stopping with a validation metric needs to improve at least once in 
                                   every round to coontinue training; default=None.
            eval_metric: Evaluatin metrics for validation data; default='logloss'; many different choices are available.
            l1_reg_alpha: L1 regularization term on weights; default=0.0.
            l2_reg_lambda: L2 regularization term on weights; default=1.0.
            focal_gamma: The gamma value for focal loss function.
            weighted_alpha: The alpha value for weight cross-entropy loss function.

        Return:
            None

        Sources:
            https://xgboost.readthedocs.io/en/stable/python/python_api.html
            https://xgboost.readthedocs.io/en/stable/parameter.html
        """
        
        self.boosted_model = 0
        self.eval_list = []
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_leaves = max_leaves
        self.eta_learning_rate = eta_learning_rate
        self.verbosity = verbosity
        self.objective = objective
        self.booster = booster
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.l1_reg_alpha = l1_reg_alpha
        self.l2_reg_lambda = l2_reg_lambda
        self.focal_gamma = focal_gamma
        self.weighted_alpha = weighted_alpha
        

    def fit(self, X, y) -> None:
        """
        This methd trains the ImbalancedXGBoost classifier. The function takes the X and y training sets and uses XGBoost's
        DMatrix() function. The same goes for X_val and y_val varaiables which need to be used when using early stopping. The params 
        dictionary involves parameters stated in the class for general algorithm purposes such as limit of tree depth and leaves,
        alpha, lambda, and gamma values, etc. Using the 'objective' variable, the function wil use XGBoost's train() function to
        train the classifier based on whether the objective function was specified (focal or weight) or not (binary:logitraw).

        Args:
            X: The training dataset.
            y: The training labels.

        Return:
            None
        """

        # Split the incoming X and y datasets into training and 
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.25, random_state=42)

        # Conversion to data matrix used in XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_valid, label=y_valid)

        # Gives training to estimator
        # used in .train() function
        self.eval_list = [(dtrain, 'train'), (dval, 'val')]

        # parameter list for .train() function
        params = {
                'max_depth': self.max_depth,
                'max_leaves': self.max_leaves,
                'eta': self.eta_learning_rate,
                'verbosity': self.verbosity,
                'objective': self.objective,
                'booster': self.booster,
                'eval_metric': self.eval_metric,
                'gamma': self.gamma,
                'min_child_weight': self.min_child_weight,
                'alpha': self.l1_reg_alpha,
                'lambda': self.l2_reg_lambda
            }

        # If the objective function is specified for the weighted cross-entropy loss function
        if self.objective == "weighted":
            # Do not need the previous objective fuction
            del params['objective']   

            # weighted_loss object is initialized using the class weightedBinaryCrossEntropy and weighted_alpha value
            weighted_loss = WeightedBinaryCrossEntropy(weighted_alpha=self.weighted_alpha) 
            # Trains boosted model using the weighted cross-entropy loss function
            self.boosted_model = xgb.train(params, dtrain, num_boost_round=self.n_estimators, evals=self.eval_list, 
                                           obj=weighted_loss.weightedCrossEntropyLoss,
                                           early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)        
        # If the objective function is specified for the focal loss function
        elif self.objective == "focal":
            # Do not need the previous objective fuction
            del params['objective']

            # focal loss object is initialized using the class focalBinaryLoss and focal_gamma value
            focal_loss = FocalBinaryLoss(focal_gamma=self.focal_gamma)
            # Trains boosted model using the focal loss function
            self.boosted_model = xgb.train(params, dtrain, num_boost_round=self.n_estimators, evals=self.eval_list, 
                                           obj=focal_loss.focalLoss,
                                           early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        # If the objective function is not specified,
        # use the default function and parameters
        else:
            self.boosted_model = xgb.train(params, dtrain, num_boost_round=self.n_estimators, evals=self.eval_list, 
                                           early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
    

    def predict(self, X, y=None) -> None:
        """
        This method is called for prediction of the classifier. The function is passed a set of examples (wit their features).
        Again, using the DMatrix() function for using the matrix used in XGBoost, The function uses the models own prediction 
        function witin.

        Args:
            X: The testing dataset.
            y: The testing labels.

        Return:
            predictions: Output of XGBoost's .predict function, giving an array of predictions.
        """
        # Check if y is type "None"
        if y is not None:
            try:
                dtest = xgb.DMatrix(X, label=y)
            except:
                raise ValueError('Test data is not working.')
        else:
            # If there is no lables, then we just use X
            dtest = xgb.DMatrix(X)

        # Using XGBoost's predict function, we get the prediction output
        y_pred = self.boosted_model.predict(dtest)
        # Compile predictions using the testing datasets
        predictions = [round(value) for value in y_pred]

        return predictions
