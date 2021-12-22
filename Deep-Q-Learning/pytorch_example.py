import torch.nn as nn  # Layers
import torch.nn.functional as F  # Activation functions
import torch.optim as optim  # Optimizers
import torch as T

class LinearClassifier(nn.Module): # Access to deep learning parameters
    # nn.Module allows you to call methods like to("cuda:0"), .eval(), .parameters(), .zero_grad(), .forward()
    def __init__(self, lr, n_classes, input_dims):
        super(LinearClassifier, self).__init__()

        self.fc1 = nn.Linear(*input_dims, 128) #*input_dims : * to unpack the list of dims
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, n_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device) # Two types of tensors. One that lives on the CPU and one on the GPU (CUDA Tensors). Make sure to send it to the GPU.

    def forward(self, data):
        layer1 = F.sigmoid(self.fc1(data))
        layer2 = F.sigmoid(self.fc2(layer1))
        layer3 = self.fc3(layer2)

        return layer3

    def learn(self, data, labels):
        self.optimizer.zero_grad() # Sets gradients of all model parameters to zero.
        data = T.tensor(data).to(self.device) # torch needs tensors and not any other data like numpy arrays
        labels = T.tensor(labels).to(self.device) # Must do it for even labels

        predictions = self.forward(data)

        cost = self.loss(predictions, labels)

        # The gradients are "stored" by the tensors themselves (they have a grad and a requires_grad attributes) once you
        # call backward() on the loss. After computing the gradients for all tensors in the model, calling optimizer.step()
        # makes the optimizer iterate over all parameters (tensors) it is supposed to update and use their internally stored
        # grad to update their values.
        cost.backward()
        self.optimizer.step()
