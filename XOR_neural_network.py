import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


training_inputs = numpy.array([[1, 1],
                               [1, 0],
                               [0, 1]])
training_outputs = numpy.array([[0, 1, 1]]).T

numpy.random.seed(1)

w = 2 * numpy.random.random((2, 1)) - 1

for i in range(10 ** 5):
    input_layer = training_inputs
    outputs = sigmoid(numpy.dot(input_layer, w))

    err = training_outputs - outputs
    adjustments = numpy.dot(input_layer.T, err * (outputs * (1 - outputs)))

    w += adjustments


two_nums = list(map(int, input('Введите два числа для использования оператора XOR:').split()))

input_layer = [two_nums]
result = sigmoid(numpy.dot(input_layer, w))

print(f'Результат: {round(result[0][0])}')
