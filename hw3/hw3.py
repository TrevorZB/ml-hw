# hw3
import csv
import numpy
import time


class Regression:
    def __init__(self, filename, step_size, theta=None):
        self.filename = filename
        self.step_size = step_size
        self.f = open(self.filename, 'r')
        self.csvreader = csv.reader(self.f)
        if theta.any():
            self.theta = theta
        else:
            self.theta = numpy.zeros(6)
        self.theta_trans = self.theta.transpose()
        self.gradient_sum = numpy.zeros(6)

    def _convergence(self):
        for grad in self.gradient_sum:
            if abs(grad) > 0.01:
                return False
        return True

    def main_driver(self):
        i = 0
        while True:
            if i > 30000 and self._convergence():
                print('Convergence Reached')
                self.print_progress(i)
                break

            self.calc_gradient()
            self.calc_new_theta()

            if i % 500 == 0:
                self.print_progress(i)
            i += 1

        self.f.close()

    def calc_gradient(self):
        self.gradient_sum = numpy.zeros(6)
        self.f.seek(0)
        next(self.csvreader)
        for row in self.csvreader:
            d = numpy.array(row).astype(numpy.float)
            features = d[1:]
            label = d[0]
            self._gradient(label, features)


    def _gradient(self, y_i, x_i):
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
        self.gradient_sum += gradient

    def calc_new_theta(self):
        self.theta += (self.step_size * self.gradient_sum)
        self.theta_trans = self.theta.transpose()

    def print_progress(self, i):
        print(f'ITERATION: {i}:')
        print(f'THETA: {self.theta}')
        print(f'GRADIENT: {self.gradient_sum}')
        
    def calc_log_likelihood(self):
        self.f.seek(0)
        next(self.csvreader)
        summation = 0
        self.theta_trans = self.theta.transpose()
        for row in self.csvreader:
            d = numpy.array(row).astype(numpy.float)
            x_i = d[1:]
            y_i = d[0]
            expon_a = -self.theta_trans.dot(x_i)
            expon_b = self.theta_trans.dot(x_i)
            a = (1 / (1 + numpy.exp(expon_a))) ** y_i
            b = (1 / (1 + numpy.exp(expon_b))) ** (1 - y_i)
            summation += numpy.log(a * b)
        print(summation)

if __name__ == '__main__':
    theta = numpy.array([-0.5334428,   2.75141467, -0.01618011, -0.33762346, -0.1458185,   0.00960426])
    r = Regression('titanic_data.csv', 0.000004, theta)
    # r.main_driver()
    r.calc_log_likelihood()

