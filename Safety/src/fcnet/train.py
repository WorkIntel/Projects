

# import libraries
import numpy as np
from fcnet.layer import FCLayer # Linear Layer 
import matplotlib.pyplot as plt # For plotting the loss and accuracy

 # importing common Use modules 
import sys 
sys.path.append(r'..\Utils\src')
from common import log

# The following libraries are loaded for data loading and preprocessing 
#from keras.datasets import mnist
#from keras.utils import to_categorical

#%% Main
class Model():
    def __init__(self, input_size, output_size, hidden_size):
        """
        Creating the model using layers (written from scratch)
        Args:
            input_size (int): Input shape of the data
            output_size (int): Output size of the layer
            hidden_size (int): size of hidden layers
        """
        self.layer1 = FCLayer(input_size=input_size, output_size=hidden_size[0], activation="relu")
        self.layer2 = FCLayer(input_size=hidden_size[0], output_size=hidden_size[1], activation="relu")
        self.layer3 = FCLayer(input_size=hidden_size[1], output_size=output_size, activation="softmax")
        
    def forward(self, inputs):
        """
        Forward propagation

        Args:
            x (Tensor): A tensor consist of neumerical values of the data
        """
        # Calculate the output
        output1 = self.layer1.forward(inputs)
        output2 = self.layer2.forward(output1)
        output3 = self.layer3.forward(output2)
        
        return output3
        
    def train(self, inputs, targets, n_epochs, initial_learning_rate, decay, plot_training_results=False):
        """
        This function does the training process of the model,
            First forward propagation is done, then the loss and accuracy are calculated,
            After that the backpropagation is done. 
        
        Args:
            inputs (Tensor): _description_
            targets (Tensor): _description_
            initial_learning_rate (float): _description_
            decay (float): _description_
            plot_training_results (bool, optional): _description_. Defaults to False.
        """        
        # Define timestep
        t = 0
        
        # Define lists for loss and accuracy
        loss_log = []
        accuracy_log = []
        
        # Loop over number of epochs 
        for epoch in range(n_epochs):
            # calculate the forward pass  
            output  = self.forward(inputs=inputs)
            
            # Calculate the loss (Categorical Crossentropy)
            epsilon = 1e-10
            loss    = -np.mean(targets * np.log(output + epsilon))
            
            # calculate the accuracy 
            predicted_labels = np.argmax(output, axis=1)
            true_labels = np.argmax(targets, axis=1)
            accuracy = np.mean(predicted_labels == true_labels)
            
            # backward 
            output_grad = 6 * (output - targets) / output.shape[0]
            t += 1
            learning_rate = initial_learning_rate / (1 + decay * epoch)
            grad_3 = self.layer3.backward(output_grad, learning_rate, t)
            grad_2 = self.layer2.backward(grad_3, learning_rate, t)
            grad_1 = self.layer1.backward(grad_2, learning_rate, t)
            
            # Add the loss and accuracy to the list 
            if plot_training_results == True:
                loss_log.append(loss)
                accuracy_log.append(accuracy)
            
            # print training results 
            self.tprint(f"Epoch {epoch} // Loss: {loss:.4f} // Accuracy: {accuracy:.4f}")
            
    def show(self, n_epochs, loss_log, accuracy_log, plot_training_results=False):
        # Draw plot if needed 
        if plot_training_results == True:
            plt.plot(range(n_epochs), loss_log, label='Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training Loss Curve')
            plt.legend()
            plt.show()

            plt.plot(range(n_epochs), accuracy_log, label='Training Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.title('Training Accuracy Curve')
            plt.legend()
            plt.show()


    def tprint(self, txt = '', level = 'I'):
        if level == "I":
            log.info(txt)
        elif level == "W":
            log.warning(txt)
        elif level == "E":
            log.error(txt)
        else:
            log.info(txt) 
            

""" 
Model Training 
"""

if __name__ == "__main__":   
    # Define hyperparameters for training 
    INPUT_SIZE = 784
    HIDDEN_SIZE = [512, 512]
    OUTPUT_SIZE = 10            
                
                
    # load the dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Flatten the images
    x_train = x_train.reshape((60000, 784))

    # Normalize 
    x_train = x_train.astype("float32") / 255.0

    # Preprocess the labels 
    y_train = to_categorical(y_train)

    # Create the Neural Network model
    nn = Model(input_size=INPUT_SIZE, output_size=OUTPUT_SIZE, hidden_size=HIDDEN_SIZE)
    nn.train(x_train, y_train, initial_learning_rate=0.001, decay=0.001, n_epochs=100, plot_training_results=True)