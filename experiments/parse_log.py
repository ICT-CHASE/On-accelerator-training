import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_log(log_file):
    if not os.path.exists(log_file):
        raise ValueError('cannot find log file: {}'.format(log_file))
    with open(log_file, 'r') as f:
        lines = f.readlines()
    
    training_loss = []
    testing_loss = []
    testing_acc = []

    for line in lines:
        line = line.strip()
        if line.find('Iteration') != -1 and line.find('loss') != -1:
            ## training loss
            start = line.index('Iteration') + len('Iteration')
            end = line.find('(')
            if end == -1:
                end = line.find(',')

            iteration = line[start: end]
            start = line.index('loss = ') + len('loss = ')
            end = len(line)
            loss = line[start:end]
            try:
                iteration = int(iteration)
                loss = float(loss)
            except ValueError:
                print('cannot convert {} to iteration value(int) or {} to loss value(float)'.format(
                    iteration, loss))
                raise ValueError
            else:
                training_loss.append((iteration, loss))

        elif line.find('Iteration') != -1 and line.find('Testing net') != -1:
            ## testing iteration
            start = line.index('Iteration') + len('Iteration')
            end = line.index(',')
            test_iteration = line[start: end]
            try:
                test_iteration = int(test_iteration)
            except ValueError:
                print('cannot convert {} to test iteration value(int)'.format(test_iteration))
        elif line.find('Test net output') != -1 and line.find('accuracy') != -1:
            ## testing accuracy
            start = line.index('accuracy = ') + len('accuracy = ')
            end = len(line)
            accuracy = line[start: end]
            try:
                accuracy = float(accuracy)
            except ValueError:
                print('cannot convert {} to accuracy value(float)',format(accuracy))
            else:
                assert test_iteration is not None
                testing_acc.append((test_iteration, accuracy))
        elif line.find('Test net output') != -1 and line.find('loss') != -1:
            ## testing accuracy
            start = line.index('loss = ') + len('loss = ')
            end = line.index('(')
            test_loss = line[start: end]
            try:
                test_loss = float(test_loss)
            except ValueError:
                print('cannot convert {} to loss value(float)',format(test_loss))
            else:
                assert test_iteration is not None
                testing_loss.append((test_iteration, test_loss))

    return training_loss, testing_loss, testing_acc

def plot(training_loss, testing_loss, testing_acc):
    fig = plt.figure()
    """
    iterations = [x[0] for x in training_loss]
    loss = [x[1] for x in training_loss]
    ax1 = fig.add_subplot(111)
    ax1.plot(iterations, loss, 'r-', label='loss')

    iterations = [x[0] for x in testing_loss]
    loss = [x[1] for x in testing_loss]
    ax1.plot(iterations, loss, 'g--')
    """
    iterations = [x[0] for x in testing_acc]
    acc = [x[1] for x in testing_acc]
    ax1 = fig.add_subplot(111)
    ax1.plot(iterations, acc, 'c-', label='acc')

if __name__ == '__main__':
    log_file = sys.argv[1]
    training_loss, testing_loss, testing_acc = parse_log(log_file)

    plot(training_loss, testing_loss, testing_acc)
    plt.show()




