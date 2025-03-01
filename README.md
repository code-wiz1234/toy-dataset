 1. GENERATING TOY DATA: 

The purpose of the toy problem is to train the ANN on learning the sine function, (y = sin(x)), for x in the range (-2 pi <= x <= 2 pi).

Training Data:  
 generates 1000 equally spaced values between -2 pi & 2 pi .These values are passed through the sine function to generate corresponding training values This creates the training dataset for the ANN to learn the mathematical function

Validation Data:  
  The validation set is generated using 300 randomly selected x values within the same range ( -2 pi <= x <= 2 pi ) The corresponding sine values are calculated for these random x_val values. This validation dataset ensures that the model is tested on different input points, not seen during training.

2. DEFINING ANN STRUCTURE AND INITIALISING WEIGHTS

 Class Initialization:  
  The ANN class constructor initialises the network structure by defining the number of nodes in the input layer, hidden layer, and output layer. Additionally, a default learning rate of 0.001 is specified, which controls how much the weights are adjusted during training.

 Weight and bias initialization  
  The weights for the connections between the input and hidden layers are initialised using a normal distribution. Helps prevent issues with gradients vanishing or exploding.  
  The biases for the hidden layer are initialised to zero.  



  
3 . ANN STRUCTURE

Class Constructor: The constructor initialises the parameters of the network, including the number of input, hidden, and output neurons, the learning rate, and momentum. The weights are initialised with random values following a normal distribution, scaled by the square root of the number of input neurons to manage gradient propagation effectively. Biases for both layers are initialised to zero.

 Forward Pass:
The forward pass computes the output of the network for a given input. It involves two steps:

 Hidden Layer Calculation: The input is multiplied by the input-to-hidden weights, added to the bias, and passed through the Tanh activation function.
 Output Layer Calculation: The hidden layer output is then multiplied by the hidden-to-output weights, added to the bias, producing the network’s output for regression tasks. 

4.  BACK-PROPAGATION

Δwji(n)=ηδi(n)yi(n)
 weight update formula 
η is the learning rate, δj(n) is the error term, and yi(n) is the input to the weight.

δj​(n)=ϕj′​(vj​(n))ej​(n)
 error 

The backpropagation algorithm is a fundamental part of training artificial neural networks. It enables the model to adjust its weights and biases based on the error of its predictions. This report outlines the implementation of the backpropagation algorithm in a fully vectorized manner, emphasising efficiency and accuracy.

 Error Calculation: The output error is computed as the difference between the predicted output and the true output. This error is crucial as it indicates how far off the predictions are from the actual values.

    Hidden Layer Error:
 The error for the hidden layer is calculated by backpropagating the output error through the network. This ensures that the weights associated with the hidden layer are updated based on their contribution to the output error.

   Weight and Bias Updates:
 The weights and biases are updated using the calculated errors. This is done using batch gradients to ensure that the updates are based on the mean error across the batch
In our backpropagation implementation, we have effectively applied the principles described in equations (D) and (E):
Equation (D) describes the relationship between the output layer and the inputs to the hidden layer:

V(l)=W(l)⋅Y(l−1)
In our code, this corresponds to how the weights between the hidden and output layers are updated based on the contributions of the hidden layer's activations.

Equation (E) relates to the summation of errors in the hidden layer:
δj(n)=ϕj′(Vj(n))⋅∑kδk(n)⋅wkj(n) 
This equation captures the essence of backpropagation through the hidden layer, allowing us to compute the hidden errors based on the output errors and the weights connecting to the next layer.

6. TRAINING
 Data Shuffling:  
At the beginning of each epoch, the training data is shuffled to ensure that the model does not learn any order-dependent patterns in the data, which helps in achieving better generalisation.
  Mini-Batch Training:
The function checks if `batch_size` is set to `'full'`, in which case it uses the entire training dataset as a single batch. It then iterates over the training data in increments of the specified batch size, creating mini-batches of inputs and labels. For each mini-batch, the function performs a forward pass followed by a backpropagation using vectorized operations.
Error Calculation: 
After processing all mini-batches for an epoch, the function calculates the Mean Squared Error (MSE) for both the training and validation datasets. This is done by performing a full pass through the training and validation sets and comparing the predicted outputs with the true labels.
Progress Reporting:
 Every 100 epochs, the function prints the current epoch number along with the training and validation errors to monitor the learning process.
Return Values: 
The function returns two lists: `train_errors` and `val_errors`, which contain the MSE values for the training and validation datasets over the course of training.
The choice of mini-batch size significantly affects the training dynamics and convergence behaviour of the neural network. In this implementation, we explored several batch sizes, including:

 7.  ACTIVATION FUNCTIONS: 

Tanh and Sigmoid.

These are fundamental to the operation of the artificial neural network, as they allow for non-linear mappings between inputs and outputs.

 Tanh Function:  
  The tanh(x) function applies the hyperbolic tangent function element-wise to the input x, squashing values into the range between -1 and 1.
 
Sigmoid Function:  
  The sigmoid(x) function applies the logistic sigmoid function, transforming values into the range [0, 1]. 

7. MAIN EXECUTION:
Data normalisation:
A toy dataset is generated and normalised between -1 and 1 to ensure inputs are suitable for the network.
We have used one hidden layer in our code and have tried different learning rates, such as 0.01, 0.001 and 0.0001. 
8. TRAINING STOPPED
Early stopping is used to prevent overfitting. We monitor the performance of the model on a validation set during training and stop the training when the model’s performance on the validation set starts to degrade. The reason this is done is that the network may start to memorise the training data and capture noise and irrelevant patterns rather than generalising to new data.  
We also use MAPE to evaluate the performance of the model. It calculates the average absolute percentage difference between actual and predicted values, giving insight into the accuracy of models. We have done so for both the training and validation set. 
9. ADAM
We have also implemented ADAM, which is basically the combination of SGD-with-momentum and RMSProp. While SGD-with-momentum smooths out the learning process by accumulating an exponentially decaying average of past gradients, RMSProp maintains the exponentially decaying average of squared gradients, to help adjust the learning rate for each parameter. 
The Adam optimization algorithm was also implemented to improve the training process:
Initialization:
The neural network parameters, including weights and biases, as well as the Adam-related parameters, are initialised.
Forward Pass:
The forward pass computes predictions by feeding input data through the network.
Backward Pass:
Errors and gradients (deltas) are computed using backpropagation. The moment estimates (previous weight updates and biases) are updated using squared gradients.
Adam Update Step:
The corrected first and second moment estimates are calculated, and weights and biases are updated using Adam's update rule. This rule adapts the learning rates for each parameter based on the magnitudes of the first and second moment estimates, making the optimization process more efficient.
Training Loop and Early Stopping:
The forward pass, backward pass, and update steps are repeated for each epoch and batch. Training and validation errors are tracked, with early stopping applied when validation errors no longer improve.
Difference from Momentum Optimization:
Unlike momentum optimization, which maintains only one moment estimate, Adam maintains both first and second moment estimates for each parameter. Adam adapts learning rates dynamically based on these estimates and includes bias correction for stabilising the optimization process, particularly in the initial iterations.
Benefits of Adam Optimization:
Adam optimization combines the advantages of both momentum and RMSprop, making it highly effective for training neural networks. It overcomes the challenges of setting fixed learning rates and handles sparse gradients more efficiently.

COMBINED CYCLE POWER PLANT DATASET

1. CCPP : 
The dataset consists of 9568 data points, the key features in this dataset include four ambivalent variables: temperature (T), Ambient Pressure (AP), Relative Humidity (RH), and Exhaust Vacuum (V), used to predict the plant’s net hourly electrical energy output (EP). These variables are critical to the operation of a CCPP, which generates electricity through a combination of gas turbines (GT) and steam turbines (ST), with heat recovery steam generators playing a pivotal role in the process. 

2. CODE :
We have followed a similar approach to the toy problem code used earlier, 
to predict the energy output based on the given features. We split the dataset into training and validation sets. This allows to effectively monitor the model’s learning process and prevent overfitting. The ANN consists of an input layer, a hidden layer and an output layer and the forward pass uses the tanh activation in the hidden layer. The training function performs mini batch gradient descent by shuffling the training data , dividing it into batches and iterating over the batches to update network parameters. After each epoch, the training and validation MSE are calculated and recorded. 

3. ADAM:
 The network employs the Adam optimizer for backpropagation, which adjusts the weights and biases using first and second moment estimates of gradients. These moments are corrected for bias to improve training performance. The backpropagation method computes the error at the output and hidden layers, adjusts weights and biases using Adam steps, and updates the weights accordingly. 

BEST PARAMETERS


Input size - 4
Learning rate- 0.0001
Epochs- 5000
Lambda -0
Batch size - 100
Hidden activation function- ReLU
Output- sigmoid 
Early stopping - 220
For Adam - Beta1- 0.9 and Beta2-0.99
CONCLUSION
The implementation of this ANN project covered multiple phases, including data preprocessing, model development, hyperparameter tuning, and evaluation. The dataset was carefully preprocessed to normalise inputs and outputs, followed by the construction of a flexible neural network capable of handling various configurations of hidden layers, neurons, and activation functions.
The training process was closely monitored using both training and validation error metrics, with early stopping employed to prevent overfitting. The final evaluation on the test set demonstrated the model’s ability to predict accurately, with the resulting MAPE reflecting reasonable generalisation performance.
In conclusion, this project successfully demonstrated the effectiveness of Artificial Neural Networks (ANNs) in regression tasks. It highlighted the importance of hyperparameter tuning and validation techniques to achieve optimal model performance. Adam optimization further enhanced the training process, showing how advanced optimization techniques can lead to faster and more reliable convergence.
