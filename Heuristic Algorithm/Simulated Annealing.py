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
import numpy as np

class SimulatedAnnealing:
    def __init__(self, **kwargs):
        """
        :param initial_solution: 初始解，以列表的形式
        :param sol_cnt: 候选解集的解数量
        :param objective_function: 目标函数
        :param opt_type: 'max', 'min'
        :param constraint: 约束条件【输入自变量形式，返回True or False】
        :param integer: 变量是否是整数
        :param prob_type: 概率函数【可选‘Metropolis’或者自定义一个函数
        :param temperature: 初始温度
        :param cooling_rate: 冷却系数
        :param num_iterations: 每次退火的迭代次数
        :param coef: 生成邻域解的时候乘的系数【在乘上温度值之后的系数】
        :param check_num: 在搜索邻域解时候允许自变量搜索满足约束解的次数，若是超过次数则舍弃这个解，将之前的一个解进行搜索
        """

        # 支持传入字典参数，也可以传入值参数修改默认的字典参数
        default_params = {
            "initial_solution": None,
            'sol_cnt': 10,
            "objective_function": None,
            'constraint': None,
            'integer': 0,
            'opt_type': 'min',
            "temperature": 1000,
            'cooling_rate': 0.99,
            'num_iterations': 1000,
            'current_solution': None,
            'best_solution': None,
            'best_fitness': None,
            'prob_type': 'Metropolis',
            'coefficient': 0.5,
            'check_num': 100000
        }
        if 'opt_type' in kwargs and kwargs['opt_type'] not in ['max', 'min']:
            raise ValueError(f"请输入正确的opt_type")

        if 'num_iterations' in kwargs and not isinstance(kwargs['num_iterations'], int):
            raise ValueError(f"请输入正确的num_iterations")

        if 'temperature' in kwargs and not isinstance(kwargs['temperature'], (int, float)):
            raise ValueError(f"请输入正确的temperature")

        if 'cooling_rate' in kwargs and not isinstance(kwargs['cooling_rate'], (int, float)):
            raise ValueError(f"请输入正确的cooling_rate")

        if 'coefficient' in kwargs and not isinstance(kwargs['coefficient'], (int, float)):
            raise ValueError(f"请输入正确的coefficient")

        if 'check_num' in kwargs and not isinstance(kwargs['check_num'], (int, float)):
            raise ValueError(f"请输入正确的check_num")

        for key in ['objective_function', 'constraint']:
             if key not in kwargs or kwargs[key] is None:
                 raise ValueError(f"参数{key}未定义，请重新输入")

        # 先将默认参数设置到实例上
        for key, value in default_params.items():
            setattr(self, key, value)

        # 再根据传入的关键字参数更新实例属性
        if kwargs:
            for key, value in kwargs.items():
                setattr(self, key, value)

        if self.initial_solution  is None:
            self.initial_solution = self.initial_sol()

        # 查看输入的初始变量和给定是否是整数的约束条件是否一致
        if self.integer:
            if all(isinstance(x, int) for x in np.nditer(np.array(self.initial_solution))):
                pass
            else:
                raise ValueError(f"定义自变量为整数类型，但是初始值内存在浮点数，请重新定义初始值或修改自变量类型")

        if kwargs:
            if kwargs['initial_solution'] is not None and ('best_solution' not in kwargs or kwargs['best_solution'] is None):
                setattr(self, 'best_solution', kwargs['initial_solution'])

            if kwargs['initial_solution'] is not None and ('current_solution' not in kwargs or kwargs['current_solution'] is None):
                setattr(self, 'current_solution', kwargs['initial_solution'])

            if kwargs['initial_solution'] is not None and kwargs["objective_function"] is not None and ('best_fitness' not in kwargs or kwargs['best_fitness'] is None):
                setattr(self, 'best_fitness', kwargs["objective_function"](kwargs['initial_solution']))




    def initial_sol(self):
        """
        初始化候选解集
        :return: initial_solutions
        """
        sol_cnt = self.sol_cnt
        check_num = self.check_num # 直接共用一个参数
        candidate_sol_cnt = sol_cnt * 3

        candidate_sol = []
        while len(candidate_sol) <= candidate_sol_cnt:











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
            i = 0
            while i <= len(solution):
                flag = 0

                # 更改自变量值之后需要检查是否满足约束

                for j in range(self.check_num):
                    if self.integer:
                        new_sol = neighbor[i] + int(random.uniform(-1, 1) * self.temperature * self.coefficient)
                    else:
                        new_sol = neighbor[i] + random.uniform(-1, 1) * self.temperature * self.coefficient

                    if self.constraint(new_sol):
                        neighbor[i] += new_sol
                        flag = 1
                        break

                # 若是没有找到邻域内的可行解，那么这个解删除，用前一个解代替，如果是第一个解，那么保持不变
                if flag == 0:
                    if i == 0:
                        new_sol = neighbor[i]
                    else:
                        # 如果尝试多次都无法满足约束，那么有一半的概率用前一个值代替，也有一半的概率保持当前值不变
                        if random.random() < 0.5:
                            new_sol = neighbor[i - 1]
                        else:
                            new_sol = neighbor[i]
                    neighbor[i] = new_sol

                i += 1

        elif isinstance(solution, list) and all(isinstance(x, list) for x in solution):
            # 多维数组情况
            i = 0
            while i <= len(solution):
                flag = 0
                new_sol = [0]* len(solution[i])

                for j in range(self.check_num):
                    if self.integer:
                        for j in range(len(solution[i])):
                            new_sol[j] = neighbor[i][j] + int(random.uniform(-1, 1) * self.temperature * self.coefficient)
                    else:
                        for j in range(len(solution[i])):
                            new_sol[j] = neighbor[i][j] + random.uniform(-1, 1) * self.temperature * self.coefficient

                    if self.constraint(new_sol):
                        neighbor[i] += new_sol
                        flag = 1
                        break

                if flag == 0:
                    if i == 0:
                        new_sol = neighbor[i]
                    else:
                        if random.random() < 0.5:
                            new_sol = neighbor[i - 1]
                        else:
                            new_sol = neighbor[i]
                    neighbor[i] = new_sol
                i += 1

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
            return self.prob(new_fitness, current_fitness)

    def best(self):  # 获取最优目标函数值
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in range(self.iter):
            f = self.func(self.x[i], self.y[i])
            f_list.append(f)
        f_best = min(f_list)

        idx = f_list.index(f_best)
        return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标

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


def function(x):
    return x


if __name__ == '__main__':

    a = SimulatedAnnealing(initial_solution=2, objective_function=function)
    print(a.best_fitness)
    print(a.current_solution)

    # import math
    # from random import random
    # import matplotlib.pyplot as plt
    #
    #
    # def func(x, y):  # 函数优化问题
    #     res = 4 * x ** 2 - 2.1 * x ** 4 + x ** 6 / 3 + x * y - 4 * y ** 2 + 4 * y ** 4
    #     return res
    #
    #
    # # x为公式里的x1,y为公式里面的x2
    # class SA:
    #     def __init__(self, func, iter=100, T0=100, Tf=0.01, alpha=0.99):
    #         self.func = func
    #         self.iter = iter  # 内循环迭代次数,即为L =100
    #         self.alpha = alpha  # 降温系数，alpha=0.99
    #         self.T0 = T0  # 初始温度T0为100
    #         self.Tf = Tf  # 温度终值Tf为0.01
    #         self.T = T0  # 当前温度
    #         self.x = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个x的值
    #         self.y = [random() * 11 - 5 for i in range(iter)]  # 随机生成100个y的值
    #         self.most_best = []
    #         """
    #         random()这个函数取0到1之间的小数
    #         如果你要取0-10之间的整数（包括0和10）就写成 (int)random()*11就可以了，11乘以零点多的数最大是10点多，最小是0点多
    #         该实例中x1和x2的绝对值不超过5（包含整数5和-5），（random() * 11 -5）的结果是-6到6之间的任意值（不包括-6和6）
    #         （random() * 10 -5）的结果是-5到5之间的任意值（不包括-5和5），所有先乘以11，取-6到6之间的值，产生新解过程中，用一个if条件语句把-5到5之间（包括整数5和-5）的筛选出来。
    #         """
    #         self.history = {'f': [], 'T': []}
    #
    #     def generate_new(self, x, y):  # 扰动产生新解的过程
    #         while True:
    #             x_new = x + self.T * (random() - random())
    #             y_new = y + self.T * (random() - random())
    #             if (-5 <= x_new <= 5) & (-5 <= y_new <= 5):
    #                 break  # 重复得到新解，直到产生的新解满足约束条件
    #         return x_new, y_new
    #
    #     def Metrospolis(self, f, f_new):  # Metropolis准则
    #         if f_new <= f:
    #             return 1
    #         else:
    #             p = math.exp((f - f_new) / self.T)
    #             if random() < p:
    #                 return 1
    #             else:
    #                 return 0
    #
    #     def best(self):  # 获取最优目标函数值
    #         f_list = []  # f_list数组保存每次迭代之后的值
    #         for i in range(self.iter):
    #             f = self.func(self.x[i], self.y[i])
    #             f_list.append(f)
    #         f_best = min(f_list)
    #
    #         idx = f_list.index(f_best)
    #         return f_best, idx  # f_best,idx分别为在该温度下，迭代L次之后目标函数的最优解和最优解的下标
    #
    #     def run(self):
    #         count = 0
    #         # 外循环迭代，当前温度小于终止温度的阈值
    #         while self.T > self.Tf:
    #
    #             # 内循环迭代100次
    #             for i in range(self.iter):
    #                 f = self.func(self.x[i], self.y[i])  # f为迭代一次后的值
    #                 x_new, y_new = self.generate_new(self.x[i], self.y[i])  # 产生新解
    #                 f_new = self.func(x_new, y_new)  # 产生新值
    #                 if self.Metrospolis(f, f_new):  # 判断是否接受新值
    #                     self.x[i] = x_new  # 如果接受新值，则把新值的x,y存入x数组和y数组
    #                     self.y[i] = y_new
    #             # 迭代L次记录在该温度下最优解
    #             ft, _ = self.best()
    #             self.history['f'].append(ft)
    #             self.history['T'].append(self.T)
    #             # 温度按照一定的比例下降（冷却）
    #             self.T = self.T * self.alpha
    #             count += 1
    #
    #             # 得到最优解
    #         f_best, idx = self.best()
    #         print(f"F={f_best}, x={self.x[idx]}, y={self.y[idx]}")
    #
    #
    # sa = SA(func)
    # sa.run()
    #
    # plt.plot(sa.history['T'], sa.history['f'])
    # plt.title('SA')
    # plt.xlabel('T')
    # plt.ylabel('f')
    # plt.gca().invert_xaxis()
    # plt.show()

