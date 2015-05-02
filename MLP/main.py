import network
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# neurons 1st layer, neurons 2nd layer, neurons final layer
net = network.Network([2, 3, 1])
net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data=test_data)