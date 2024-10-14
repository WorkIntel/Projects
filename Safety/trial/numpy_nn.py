import numpy as np
import matplotlib.pyplot as plt
import json
from activations import activation_function, derivate_function



class NN:
   
    def __init__(self, input_shape:int=None, load:str=None):
        self.layers = [input_shape]
        self.activation = []
        self.weights = []
        self.biases = []
        if load is not None:
            with open(load, 'r') as m:
                model = json.loads(m.read())

            self.layers = model['layers']
            self.activation = model['activations']
            self.weights = np.load(model['weights'], allow_pickle=True)
            self.biases = np.load(model['biases'], allow_pickle=True)

            
    def add(self, layer:int, activation_function:str):
        #Get number of layers to index the next layer
        layers_count = len(self.layers)
        self.layers.append(layer)
        self.activation.append(activation_function)

        self.weights.append(np.random.uniform(-1,1,(self.layers[layers_count], self.layers[layers_count-1]) ))
        self.biases.append(np.random.uniform(-1,1 ,(self.layers[layers_count], 1)))


    def forward(self, X):
        inputs = X

        # Keep track of each layers for backprop
        layers_outputs = [inputs] 
        outputs = []

        for i in range(len(self.weights)):
            activation_name = self.activation[i]

            #Calculate the dot product of inputs and weights and add bias
            outputs.append(self.weights[i].dot(inputs) + self.biases[i])

            #Calculate the next input, ( activation(x) )
            inputs = activation_function(activation_name)(outputs[-1])
            layers_outputs.append(inputs)

        return (outputs, layers_outputs)

      
    def back_propagate(self, target_output, weighted_sum, layers_outputs):
        dweights = []
        dbias = []
        deltas = [0] * len(self.weights)

        # calculate deltas
        # Delta_W(j,i) = lr*(target_output - layers_output) * g'(weightet_sum)*Xi(input) 
        deltas[-1] = ((target_output - layers_outputs[-1]) *
                    derivate_function(self.activation[-1])(weighted_sum[-1])
                )
        # Loop backwards for back propagation
        for i in reversed(range(len(deltas)-1)):

            #calculate deltas for all the weights
            deltas[i] = self.weights[i+1].T.dot(
                deltas[i+1])* (derivate_function(self.activation[i])(weighted_sum[i])
                )
            batch_size = target_output.shape[1]

        
        dbias = [x.dot(np.ones((batch_size, 1))) / float(batch_size) for x in deltas]
        dweights = [x.dot(layers_outputs[i].T) / float(batch_size) for i,x in enumerate(deltas)]

        return dweights, dbias

      
    def train(self, x, y, batch_size, epochs, lr=0.01):
        for epoch in range(epochs):
            i = 0
            while i < y.shape[1]:
                x_batch = x[ :, i : i + batch_size ]
                y_batch = y[ :, i : i + batch_size ]
                i += batch_size
                outputs, layers_outputs = self.forward(x_batch)
                dweights, dbiases = self.back_propagate(y_batch, outputs, layers_outputs)

                #Update weights and biases after back propagation
                self.weights = [weight + lr*dweight for weight, dweight in zip(self.weights, dweights)]
                self.biases = [bias + lr*dbias for bias, dbias in zip(self.biases, dbiases)]

                #Print the loss of each epoch
                loss = round(np.linalg.norm(layers_outputs[-1]-y_batch), 2)
                print(f"Epoch: {epoch + 1 }. Loss: {loss}", end="\r")
        print(f"Epoch: {epoch + 1 }. Loss: {loss}")
        
        
    def save(self):
        #Save weights and biases in .npy file
        np.save("weights.npy", self.weights, allow_pickle=True)
        np.save("biases.npy", self.biases, allow_pickle=True)

        model = {
            'weights' : "weights.npy",
            "biases" : "biases.npy",
            "layers" : self.layers,
            "activations": self.activation
        }
        json_str = json.dumps(model)
        with open("model.json", "w") as m:
            m.write(json_str)

        
if __name__ == "__main__":

    nn = NN(input_shape=1)
    nn.add(100, 'sigmoid')
    nn.add(100, 'sigmoid')
    nn.add(1, 'relu') #Output layer

    X = np.pi*np.linspace(1,100,2000).reshape(1,-1)
    Y = 1 + np.log(X)
    nn.train(X,Y, epochs=100, batch_size=16, lr=0.001)

    predictions = nn.forward(X)[1][-1]
    plt.scatter(X.flatten(), Y.flatten(), color='red', label="Actual")
    plt.scatter(X.flatten(), predictions.flatten(), color='green', label="Predicted")
    plt.legend()
    plt.show()