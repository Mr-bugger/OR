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
from samples import ObjectiveFunctions, Constraints

class SimulatedAnnealing:
    def __init__(self, **kwargs):
        """
        :param initial_solution: 初始解，以列表的形式
        :param x_num: 变量个数
        :param minx: 变量最小值
        :param maxx: 变量最大值
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
            "x_num": None,
            "minx": 0,
            "maxx": 9999,
            "objective_function": None,
            'constraint': None,
            'integer': 0,
            'opt_type': 'min',
            "temperature": 100,
            'temperature_end': 10,
            'cooling_rate': 0.99,
            'num_iterations': 10,
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

        if self.initial_solution is None:
            self.initial_solution = self.initial_sol()

        # 查看输入的初始变量和给定是否是整数的约束条件是否一致
        if self.integer:
            if all(isinstance(x, int) for x in np.nditer(np.array(self.initial_solution))):
                pass
            else:
                raise ValueError(f"定义自变量为整数类型，但是初始值内存在浮点数，请重新定义初始值或修改自变量类型")

        if kwargs:
            if 'best_solution' not in kwargs or kwargs['best_solution'] is None:
                setattr(self, 'best_solution', self.initial_solution)

            if 'current_solution' not in kwargs or kwargs['current_solution'] is None:
                setattr(self, 'current_solution', self.initial_solution)

            if kwargs["objective_function"] is not None and ('best_fitness' not in kwargs or kwargs['best_fitness'] is None):
                setattr(self, 'best_fitness', min([kwargs["objective_function"](i) for i in self.initial_solution]))


    def initial_sol(self):
        """
        初始化候选解集
        :return: initial_solutions
        """
        sol_cnt = self.num_iterations
        check_num = self.check_num # 直接共用一个参数
        candidate_sol_cnt = sol_cnt * 2

        candidate_sol = []

        try:
            if self.integer:
                while len(candidate_sol) <= candidate_sol_cnt:
                    i = 0
                    while i <= check_num:
                        a = random.random(self.minx, self.maxx)
                        if self.constraint(a):
                            candidate_sol.append(a)
                            break
                return random.sample(candidate_sol, sol_cnt)
            else:
                while len(candidate_sol) <= candidate_sol_cnt:
                    i = 0
                    while i <= check_num:
                        a = random.uniform(self.minx, self.maxx)
                        if self.constraint(a):
                            candidate_sol.append(a)
                            break
                return random.sample(candidate_sol, sol_cnt)
        except:
            if self.integer:
                while len(candidate_sol) <= candidate_sol_cnt:
                    i = 0
                    while i <= check_num:
                        a = [random.random(self.minx, self.maxx) for _ in range(self.x_num)]
                        if self.constraint(a):
                            candidate_sol.append(a)
                            break
                return random.sample(candidate_sol, sol_cnt)
            else:
                while len(candidate_sol) <= candidate_sol_cnt:
                    i = 0
                    while i <= check_num:
                        a = [random.uniform(self.minx, self.maxx) for _ in range(self.x_num)]
                        if self.constraint(a):
                            candidate_sol.append(a)
                            break
                return random.sample(candidate_sol, sol_cnt)

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
            while i < len(solution):
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
            else:
                # print(math.exp((current_fitness - new_fitness) / self.temperature))
                return max(math.exp((current_fitness - new_fitness) / self.temperature), 0.01)

        else:
            return self.prob(new_fitness, current_fitness)

    def best(self):
        """
        获得当前候选解集中最优解的函数值以及最优解
        :return:
        """
        f_list = []  # f_list数组保存每次迭代之后的值
        for i in self.current_solution:
            f = self.objective_function(i)
            f_list.append(f)

        f_best = min(f_list)
        idx = f_list.index(f_best)
        return f_best, self.current_solution[idx]  # f_best,idx分别为在该温度下的函数最优值以及对应的变量的取值

    def solve(self):
        """
        模拟退火主函数入口
        :return:
        """
        result = []
        acceptance_probability_func = self.acceptance_probability
        count = 0
        # 外循环迭代，当前温度小于终止温度的阈值
        while self.temperature > self.temperature_end:

            # 内循环迭代
            for i in range(self.num_iterations):
                f = min([self.objective_function(self.current_solution[j]) for j in range(len(self.current_solution))])
                new_sol = self.neighbor_solution(self.current_solution)  # 产生新解

                f_new = min([self.objective_function(new_sol[j]) for j in range(len(new_sol))])  # 产生新值

                if random.random() <= acceptance_probability_func(f_new, f):
                    self.current_solution = new_sol

            # 迭代L次记录在该温度下最优解
            best_fitness, best_sol = self.best()
            result.append(best_fitness)

            print(f'当前优化轮数：{count+1}，当前温度值：{self.temperature}, 当前迭代最优解：{best_sol}，最优函数值：{best_fitness}')
            # 温度按照一定的比例下降（冷却）
            self.temperature = self.temperature * self.cooling_rate
            count += 1

            # 得到最优解
        best_fitness, best_sol = self.best()
        return best_fitness, best_sol


if __name__ == '__main__':

    obj = ObjectiveFunctions.mixed_function
    cons = Constraints.box_constraint
    a = SimulatedAnnealing(objective_function=obj, constraint=cons, x_num=1)
    a.solve()
    print(a.best_fitness)
    print(a.current_solution)
