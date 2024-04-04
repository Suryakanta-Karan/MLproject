# sigmoid_output.py.....
from activation_functions import sigmoid
import numpy as np

random_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
print("Sigmoid Output:")
for val in random_values:
    print(sigmoid(val))
