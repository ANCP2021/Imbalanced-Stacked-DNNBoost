import tensorflow as tf

class ImbalancedDeepNeuralNetwork:
    def __init__(self, epochs=10, batch_size=32, input_layer=12, input_layer_activation='relu', hidden_layer=8, hidden_layer_activation='relu', 
                 output_layer=1, output_layer_activation='sigmoid', optimizer='adam', loss='binary_crossentropy', 
                 focal_gamma:float=None, weighted_alpha:float=None) -> None:
        
        """
        ImbalancedDeepNeuralNetwork is a class that combines a deep neural network algorithm (imported) with 
        weighted and focal loss functions. The goal for the new loss functions provided is to handle label-imbalanced scenarios.

        Args:
            epochs: Total number of iterations of all the training data in one cycle.
            batch_size: The number of training examples in one forward or backward pass.
            input_layer: Number of nodes in the first layyer of the network.
            input_layer_activation: Type of activiation function in the first layer of the network (relu).
            hidden_layer: Number of nodes in the hidden layer of the network. In this network there is only one hidden layer.
            hidden_layer_activation: Type of activation function in the first hidden layer if the network (relu). 
                                     First layer after the input layer.
            output_layer: Number of nodes in the output layer. Since we are primarily dealing with binary classification, 1 node
                          is the default.
            output_layer_activation: Type of activation function in the output (last) layer of the network.
            optimizer: Used to change the attrubutes of the network to reduce losses.
            loss: The loss function compares the target and predicted output values and measurees how well the network is training.
            focal_gamma: The gamma value for focal loss function.
            weighted_alpha: The alpha value for weight cross-entropy loss function.
        
        Return:
            None
        """
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_layer = input_layer
        self.input_layer_activation = input_layer_activation
        self.hidden_layer = hidden_layer
        self.hidden_layer_activation = hidden_layer_activation
        self.output_layer = output_layer
        self.output_layer_activation = output_layer_activation
        self.optimizer = optimizer
        self.loss = loss
        self.focal_gamma = focal_gamma
        self.weighted_alpha = weighted_alpha
        self.metrics = []


    def compile_fit(self, X, y):
        """
        Function used for compiling and fitting the defined neural network.

        Args:
            X: The training dataset.
            y: The training labels.
        
        Return:
            Model: The defined, compiled, and fitted DNN model with the customized loss function.
        """

        # Definition of model
        # Model is considered a deep neural network due to the amount of layers
        # Model involves one input, one hidden, and one output layer
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(self.input_layer, activation=self.input_layer_activation, input_shape=X.shape[1:]),
            tf.keras.layers.Dense(self.hidden_layer, activation=self.hidden_layer_activation),
            tf.keras.layers.Dense(self.output_layer, activation=self.output_layer_activation)
        ])

        # If the specified loss function is 'weighted'...
        if self.loss == 'weighted':
            # Compile the model based on the weighted loss function defined 
            model.compile(optimizer=self.optimizer, loss=self.weightedLossFunct, metrics=['accuracy'])
        elif self.loss == 'focal': # If the specified loss function is 'focal'...
            # Compile the model based on the focal loss function defined
            model.compile(optimizer=self.optimizer, loss=self.focalLossFunct, metrics=['accuracy'])
        else: # If there is no specified loss function...
            # Compile the model based on the default value (binary_crossentropy)
            model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])

        # Fit the DNN
        model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size)

        # Return the model
        return model


    def focalLoss(self, pred, labels):
        """
        Function used for the focal loss calculation.

        Args:
            labels: The training dataset output.
            pred: Predictions made by the classifier.
        
        Return:
            gradient: First-order derivative.
            hessian: Second-order derivative.
        """

        focal_gamma = self.focal_gamma
        # Sigmoid is selected as activation with the basic property used in the derivatives
        sigmoid = 1 / (1 + tf.exp(-pred))

        # Simplification of equations using short-and variables described
        n_1 = sigmoid * (1 - sigmoid) # yhat_i*(1-yhat_i)
        n_2 = labels + ((-1) ** labels) * sigmoid # y_i + (-1)^y_i * yhat_i
        n_3 = sigmoid + labels - 1 # yhat_i + y_i + 1
        n_4 = 1 - labels - ((-1) ** labels) * sigmoid # 1 - y_i - (-1)^y_i * yhat_i

        # First-order derivative (Gradient)
        # gamma * n3 * n2^gamma * log(n4) + (-1)^y_i * n2^(gamma + 1)
        gradient = focal_gamma * n_3 * (n_2 ** focal_gamma) * tf.math.log(n_4 + 1e-9) + \
                ((-1) ** labels) * (n_2 ** (focal_gamma + 1))
            
        # Second-order derivative (Hessian)
        # n1 * {gamma * [(n2^gamma + gamma * (-1)^y_i * n3 * n2^(gamma - 1)) * log(n4) - ((-1)^y_i * n3 * n2^gamma) / n4] + ((gamma + 1) * n2^gamma)}
        hessian = n_1 * ( \
                    focal_gamma * ( \
                        ( ((n_2 ** focal_gamma) + focal_gamma * ((-1) ** labels) * n_3 * (n_2 ** (focal_gamma-1))) * tf.math.log(n_4) ) - \
                            ( ( ((-1) ** labels) * n_3 * (n_2 ** focal_gamma) ) / n_4 ) \
                    ) + \
                    ((focal_gamma + 1) * (n_2 ** focal_gamma)) \
                )

        return gradient, hessian


    def weightedCrossEntropyLoss(self, pred, labels):
        """
        Function used for the weighted cross-entropy loss calculation.

        Args:
            labels: The training dataset output.
            pred: Predictions made by the classifier.
        
        Return:
            gradient: First-order derivative.
            hessian: Second-order derivative.
        """

        weighted_alpha = self.weighted_alpha
        # Sigmoid is selected as activation with the basic property used in the derivatives
        sigmoid = 1 / (1 + tf.exp(-pred))
        # Calculation for residual (true_predictions - sigmoid)
        residual = labels - sigmoid

        # First-order derivative (Gradient)
        # -alpha^y_i * (y_i - yhat_i)
        gradient = -(weighted_alpha ** labels) * residual
        # Second-order derivative (Hessian)
        # alpha^y_i * (1 - yhat_i) * (yhat_i)
        hessian = (weighted_alpha ** labels) * (1 - sigmoid) * sigmoid

        return gradient, hessian


    def weightedLossFunct(self, labels, predictions):
        """
        Function used for calling main weighted loss function and calculating the gradient and hessians
        into a scalar value.

        Args:
            labels: The training dataset output.
            predictions: Predictions made by the classifier.
        
        Return:
            Returns the square of the input tensor (gradient / hessian) and
            then computes the mean value of elements across the tensor.
        """
        gradient, hessian = self.weightedCrossEntropyLoss(predictions, float(labels))
        return tf.reduce_mean(tf.square(gradient / (hessian + 1e-9)))


    def focalLossFunct(self, labels, predictions):
        """
        Function used for calling main focal loss function and calculating the gradient and hessians
        into a scalar value.

        Args:
            labels: The training dataset output.
            predictions: Predictions made by the classifier.
        
        Return:
            Returns the square of the input tensor (gradient / hessian) and
            then computes the mean value of elements across the tensor.
        """
        gradient, hessian = self.focalLoss(predictions, float(labels))
        return tf.reduce_mean(tf.square(gradient / (hessian + 1e-9)))
    

    def predict(self, X):
        """
        Function used for generating predictions for the input.

        Args:
            X: Input data used to make predictions.
        
        Return:
            predictions
        """
        # Calls the predict method stored in the DNN model
        predictions = self.model.predict(X)
        return predictions