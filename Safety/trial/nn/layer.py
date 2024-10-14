"""
Fully custom Linear Layer
Using only numpy  
"""

# Import libraries
import numpy as np 

class FCLayer():
    def __init__(self, input_size, output_size, activation):
        """
        Args:
            input_size (int): Input shape of the layer 
            output_size (int): Output of the layer
            activation (str): activation function 
        """
        # initialize weights and biases 
        # The weights are initialized using HE-Initialization 
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        # define the activation 
        self.activation = activation
        
        # Define m & v for weights and biases 
        # These variables are used in Adam optimization 
        self.m_weights = np.zeros((input_size, output_size))
        self.v_weights = np.zeros((input_size, output_size))
        self.m_biases = np.zeros((1, output_size))
        self.v_biases = np.zeros((1, output_size))
        
        # Define hyperparameters for Adam optimizer 
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        
    def forward(self, x):
        """
        Forward pass 

        Args:
            x (Tensor): Numerical values of the data
        """
        self.x = x
        
        # Calculate the output (z)
        z = np.dot(self.x, self.weights) + self.biases 
        
        # Apply activation functions 
        if self.activation == "relu":
            self.output = np.maximum(0, z)
            
        elif self.activation == "softmax":
            exp_values = np.exp(z - np.max(z, axis=-1, keepdims=True)) 
            self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
            
        else: 
            print(f"The activation function entered does not exist!\nUse either relu or softmax")
            
        return self.output 
    
    def backward(self, d_values, learning_rate, t):
        """
        Backpropagation 

        Args:
            d_values (float): Derivative of the output 
            learning_rate (float): Learning rate for gradient descent 
            t (int): Timestep 
        """
        # Get the derivative of the softmax Function 
        if self.activation == "softmax":
            for i, gradient in enumerate(d_values):
                if len(gradient.shape) == 1:  # For single instance
                    gradient = gradient.reshape(-1, 1)
                jacobian_matrix = np.diagflat(gradient) - np.dot(gradient, gradient.T)
                d_values[i] = np.dot(jacobian_matrix, self.output[i])
                
        # Calculate the derivative of the ReLU function 
        elif self.activation == "relu":
            d_values = d_values * (self.output > 0)
            
        # Calculate the derivative with respect to the weight and bias (one with weight and one with bias)
        d_weights = np.dot(self.x.T, d_values)
        d_biases = np.sum(d_values, axis=0, keepdims=True)
        # Limit the derivative to avoid really big or small numbers 
        d_weights = np.clip(d_weights, -1.0, 1.0)
        d_biases = np.clip(d_biases, -1.0, 1.0)
        
        # Calculate the gradient with respect to the input 
        d_inputs = np.dot(d_values, self.weights.T)
        
        # Update the weights and biases using the learning rate and their derivatives 
        self.weights -= learning_rate * d_weights
        self.biases -= learning_rate * d_biases 
        
        # Update weights using m and v values 
        m_weights = self.beta1 * self.m_weights + (1 - self.beta1) * d_weights
        v_weights = self.beta2 * self.v_weights + (1 - self.beta2) * (d_weights ** 2)
        m_hat_weights = m_weights / (1 - self.beta1 ** t)
        v_hat_weights = v_weights / (1 - self.beta2 ** t)
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + self.epsilon)
        
        # Update biases using m and v values 
        m_biases = self.beta1 * self.m_biases + (1 - self.beta1) * d_biases
        v_biases = self.beta2 * self.v_biases + (1 - self.beta2) * (d_biases ** 2)
        m_hat_biases = m_biases / (1 - self.beta1 ** t)
        v_hat_biases = v_biases / (1 - self.beta2 ** t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + self.epsilon)
        
        return d_inputs