# sigmoid_output.py.....
from activation_functions import sigmoid
import numpy as np

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
print("Sigmoid Output:")
for val in random_values:
    print(sigmoid(val))


input_shape = (28, 28, 1) # Example input shape for MNIST dataset
num_classes = 10          # Example number of classes for MNIST dataset

model = create_model(input_shape, num_classes)   # Original model
model = config_1(input_shape, num_classes)       # Configuration 1
model = config_2(input_shape, num_classes)       # Configuration 2
model = config_3(input_shape, num_classes)       # Configuration 3
model = config_4(input_shape, num_classes)       # Configuration 4
