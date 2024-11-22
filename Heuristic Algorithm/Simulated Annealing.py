# -*-coding: Utf-8 -*-
# @File : Simulated Annealing .py
# author: 薛煜殿
# email: xue_yu_dian@163.com
# Time：2024/11/20

"""
模拟退火算法包
1.支持自定义目标函数
2、自带多种接受新解的概率，支持自定义接受新解的概率函数
3. 支持自定义约束函数
4. 支持自定义变量扰动方案
"""

import random
import math

class SimulatedAnnealing:
    def __init__(self, initial_solution, objective_function, prob_type='Metropolis', temperature=1000, cooling_rate=0.99, num_iterations=1000, coefficient=1):
        """
        :param initial_solution: 初始解，以列表的形式
        :param objective_function: 目标函数
        :param prob_type: 概率函数【可选‘Metropolis’或者自定义一个函数
        :param temperature: 初始温度
        :param cooling_rate: 冷却系数
        :param num_iterations: 每次退火的迭代次数
        :param coef: 生成邻域解的时候乘的系数【在乘上温度值之后的系数】
        """
        self.initial_solution = initial_solution
        self.objective_function = objective_function
        self.temperature = temperature
        self.cooling_rate = cooling_rate
        self.num_iterations = num_iterations
        self.current_solution = initial_solution
        self.best_solution = initial_solution
        self.best_fitness = objective_function(initial_solution)
        self.prob_type = prob_type
        self.coefficient = coefficient

    def neighbor_solution(self, solution):
        """
        生成当前解的一个邻域解。根据变量的维度和类型，采用不同的扰动方式。
        对于一维数组（多个单变量），对每个元素添加随机扰动。
        对于多维数组（如矩阵形式的变量），对每个子数组或元素根据其维度进行合适的扰动。
        邻域解的大小跟温度正相关，需要根据具体量纲大小乘上对应系数coef
        """
        neighbor = solution.copy()

        if isinstance(solution, list) and all(isinstance(x, (int, float)) for x in solution):
            # 一维数组情况，多个单变量
            for i in range(len(solution)):
                neighbor[i] += random.uniform(-1, 1) * self.temperature * self.coefficient
        elif isinstance(solution, list) and all(isinstance(x, list) for x in solution):
            # 多维数组情况
            for i in range(len(solution)):
                for j in range(len(solution[i])):
                    neighbor[i][j] += random.uniform(-1, 1) * self.temperature * self.coefficient
        else:
            raise ValueError("Unsupported solution type：暂时只支持一维变量")
        return neighbor

    def acceptance_probability(self, new_fitness, current_fitness):
        """
        默认的接受概率函数，基于Metropolis准则。
        用户可以根据需要替换为其他选择概率函数。
        """
        if self.prob_type == 'Metropolis':
            if new_fitness < current_fitness:
                return 1
            return math.exp((current_fitness - new_fitness) / self.temperature)

        else:
            return self.prob

    def run(self, acceptance_probability_func=None):
        if acceptance_probability_func is None:
            acceptance_probability_func = self.acceptance_probability

        for i in range(self.num_iterations):
            new_solution = self.neighbor_solution(self.current_solution)
            new_fitness = self.objective_function(new_solution)
            current_fitness = self.objective_function(self.current_solution)

            ap = acceptance_probability_func(new_fitness, current_fitness, self.temperature)
            if random.random() < ap:
                self.current_solution = new_solution
                if new_fitness < self.best_fitness:
                    self.best_solution = new_solution
                    self.best_fitness = new_fitness

            self.temperature *= self.cooling_rate

        return self.best_solution, self.best_fitness



if __name__ == '__main__':
    import math
    from random import random
    import matplotlib.pyplot as plt


    def func(x, y):  # 函数优化问题
        res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
        return res


    # x为公式里的x1,y为公式里面的x2
    class SA:
        def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
            self.func = func
            self.iter = iter  # 内循环迭代次数,即为L =100
            self.alpha = alpha  # 降温系数，alpha=0.99
            self.T0 = T0  # 初始温度T0为100
            self.Tf = Tf  # 温度终值Tf为0.01
            self.T = T0  # 当前温度
            self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
            self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
            self.most_best = []
            """
            random()这个函数取0到1之间的小数
            如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
            该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
            （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
            """
            self.history = {'f': [], 'T': []}

        def generate_new(self, x, y):  # 扰动产生新解的过程
            while True:
                x_new = x + self.T * (random() - random())
                y_new = y + self.T * (random() - random())
                if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
                    break  # 重复得到新解，直到产生的新解满足约束条件
            return x_new, y_new

        def Metrospolis(self, f, f_new):  # Metropolis准则
            if f_new <= f:
                return 1
            else:
                p = math.exp((f - f_new) / self.T)
                if random() < p:
                    return 1
                else:
                    return 0

        def best(self):  # 获取最优目标函数值
            f_list = []  # f_list数组保存每次迭代之后的值
            for i in range(self.iter):
                f = self.func(self.x[i], self.y[i])
                f_list.append(f)
            f_best = min(f_list)

            idx = f_list.index(f_best)
            return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

        def run(self):
            count = 0
            # 外循环迭代，当前温度小于终止温度的阈值
            while self.T > self.Tf:

                # 内循环迭代100次
                for i in range(self.iter):
                    f = self.func(self.x[i], self.y[i])  # f为迭代一次后的值
                    x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 产生新解
                    f_new = self.func(x_new, y_new)  # 产生新值
                    if self.Metrospolis(f, f_new):  # 判断是否接受新值
                        self.x[i] = x_new  # 如果接受新值，则把新值的x,y存入x数组和y数组
                        self.y[i] = y_new
                # 迭代L次记录在该温度下最优解
                ft, _ = self.best()
                self.history['f'].append(ft)
                self.history['T'].append(self.T)
                # 温度按照一定的比例下降（冷却）
                self.T = self.T * self.alpha
                count += 1

                # 得到最优解
            f_best, idx = self.best()
            print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")


    sa = SA(func)
    sa.run()

    plt.plot(sa.history['T'], sa.history['f'])
    plt.title('SA')
    plt.xlabel('T')
    plt.ylabel('f')
    plt.gca().invert_xaxis()
    plt.show()

