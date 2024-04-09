import numpy as np

def sigmod(x):
    return 1.0 / (1.0 + np.exp(-x))

traning_inputs = np.array([[0,0,1],
                           [1,1,1],
                           [1,0,1],
                           [0,1,1],
                           [0,1,0],
                           [1,1,0],
                           [1,0,0],
                           [0,0,0]])

traning_outputs = np.array([[0,1,1,1,0,0,0,0]]).T

np.random.seed(1)

synaptic_weights = 2 * np.random.random((3,4)) - 1

print('Случайные инициализированные веса:')
print(synaptic_weights)
for i in range(2000):
    input_layer = traning_inputs
    outputs = sigmod(np.dot(input_layer,synaptic_weights))

    err = traning_outputs - outputs
    adjustments = np.dot( input_layer.T, err * (outputs) * (1 - sigmod(1 - outputs)) )
    
    synaptic_weights += adjustments

print('Веса после обучения:')
print(synaptic_weights)

print('Результат после обучения:')
print(outputs)