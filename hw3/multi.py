import csv
import numpy
import time
from multiprocessing import Process, Queue


class Regression:
    def __init__(self, filename, step_size):
        self.filename = filename
        self.step_size = step_size
        self.Queue = Queue()
        with open(self.filename, 'r') as data:
            csvreader = csv.reader(data)
            l = len(next(csvreader))
            self.theta = numpy.zeros(l - 1)
            self.theta_trans = self.theta.transpose()
            self.gradient_sum = numpy.zeros(l - 1)

    def _convergence(self):
        for grad in self.gradient_sum:
            if abs(grad) > 0.01:
                return False
        return True

    def main_driver(self):
        i = 0
        s = time.time()
        while True:
            if self._convergence() and i != 0:
                print('Convergence Reached')
                self.print_progress(i)
                break

            self.gradient_sum = numpy.zeros(6)

            processes = []

            p0 = Process(target=self.calc_gradient, args=[0, 300])
            p1 = Process(target=self.calc_gradient, args=[301, 600])
            p2 = Process(target=self.calc_gradient, args=[601, 887])

            p0.start()
            p1.start()
            p2.start()

            processes.append(p0)
            processes.append(p1)
            processes.append(p2)

            for p in processes:
                p.join()

            while not self.Queue.empty():
                grad = self.Queue.get()
                self.gradient_sum += grad

            self.calc_new_theta()

            if i % 10 == 0:
                self.print_progress(i)

            if i % 1000 == 0 and i != 0:
                print(f'TIME: {time.time() - s}')
                break
            i += 1

    def calc_gradient(self, start, end):
        with open(self.filename, 'r') as data:
            csvreader = csv.reader(data)
            l = len(next(csvreader))
            for _ in range(start):
                next(csvreader)

            grad = numpy.zeros(l - 1)

            i = start
            for row in csvreader:
                if i > end:
                    break
                d = numpy.array(row).astype(numpy.float)
                features = d[1:]
                label = d[0]
                self._gradient(label, features, grad)
                i += 1
        self.Queue.put(grad)

    def _gradient(self, y_i, x_i, grad):
        frac = None
        ex = -self.theta_trans.dot(x_i)
        try:
            denom = numpy.exp(ex)
        except:
            if ex > 0:
                frac = 0
            else:
                frac = 1
        else:
            frac = 1 / (1 + denom)
        gradient = (y_i - frac) * x_i
        grad += gradient

    def calc_new_theta(self):
        self.theta += (self.step_size * self.gradient_sum)
        self.theta_trans = self.theta.transpose()

    def print_progress(self, i):
        print(f'ITERATION: {i}:')
        # print(f'THETA: {self.theta}')
        print(f'GRADIENT: {self.gradient_sum}')


if __name__ == '__main__':
    r = Regression('titanic_data.csv', 0.000001)
    r.main_driver()

