from numpy import exp, array, random, dot

class SingleLayerNeuralNetwork:

    def __init__(self):
        random.seed(1)
        self.synaptic_weights = 2 * random.random((3, 1)) - 1

    def __sigmoid(self, x):
        return 1/(1 + exp(-x))

    def __sigmoid_derivative(self, x):
        return x * (1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        for i in xrange(number_of_iterations):
            output = self.predict(training_set_inputs)
            error = training_set_outputs - output
            print 'error: {}'.format(error)
            adjustment = dot(training_set_inputs.T, error * self.__sigmoid_derivative(output))
            print 'adjustment: {}'.format(adjustment)
            self.synaptic_weights = adjustment + self.synaptic_weights


    def predict(self, inputs):
        return self.__sigmoid(dot(inputs, self.synaptic_weights))

if __name__ == '__main__':

    neural_network = SingleLayerNeuralNetwork()

    print 'Random starting synaptic weights: {}'.format(neural_network.synaptic_weights)

    training_set_inputs = array([ [0,0,1], [1,1,1], [1,0,1], [0,1,1] ])
    training_set_outputs = array([[0,1,1,0]])

    neural_network.train(training_set_inputs, training_set_outputs, number_of_iterations = 10000)

    print 'New synaptic weights after training: {}'.format(neural_network.synaptic_weights)

    print 'predicting: {}'.format(neural_network.predict(array([1,0,0])))


