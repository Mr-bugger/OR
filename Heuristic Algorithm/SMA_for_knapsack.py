# -*-coding: Utf-8 -*-
# @File : SMA .py
# author: 薛煜殿
# email: xue_yu_dian@163.com
# Time：2024/11/20
import random
from functools import partial
import numpy as np
from typing import Union, List, Callable
from math import floor, ceil
from samples import ObjectiveFunctions, Constraints


class SMAforknapsack:
    def __init__(self, objective_function, values, constraint, weights, max_num: List[int],
                 max_weight, population_size=1000, max_iter=1000, try_num_max=100):
        """
        objective_function: 目标函数
        constraint: 约束条件
        max_num: 列表：存储所有物品的最多数量
        max_weight: 背包的最大重量
        population_size: 种群的大小
        max_iter: 迭代轮数
        """
        self.objective_function = partial(objective_function, values=values)
        self.constraint = partial(constraint, weights=weights, max_num=max_num, max_weight=max_weight)
        self.max_num = max_num
        self.max_weight = max_weight
        self.population_size = population_size
        self.max_iter = max_iter
        self.initial_population = self.initial_sol()
        self.current_population = self.initial_population
        self.try_num_max = try_num_max
        self.best_solution = None
        self.best_objective_function = None

    def initial_sol(self) -> List[List[int]]:
        """
        初始化种群函数，初始化population_size个解形成一个解集
        """
        population = []
        for _ in range(self.population_size):
            solution = [np.random.randint(0, self.max_num[i]+1) for i in range(len(self.max_num))]
            while not self.constraint(solution):
                solution = [np.random.randint(0, self.max_num[i]+1) for i in range(len(self.max_num))]
            population.append(solution)
        return population

    def fitness(self, x) -> float:
        """
        适应度函数
        由于是最大化问题，默认直接将目标函数作为适应度函数
        """
        if not self.constraint(x):
            return -999
        return self.objective_function(x)
    @staticmethod
    def crossover(parent1, parent2) -> List[int]:
        """
        交叉操作
        """
        crossover_point = random.randint(1, len(parent1)-1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        return child

    def mutation(self, x) -> List[int]:
        """
        变异操作
        """
        mutation_point = random.randint(0, len(x)-1)
        x[mutation_point] = random.randint(0, self.max_num[mutation_point]+1)
        while not self.constraint(x):
            x[mutation_point] = random.randint(0, self.max_num[mutation_point]+1)
            x = self.mutation(x)

        return x





    def solve(self):
        """
        主函数
        """
        current_population = self.current_population

        fitness_list = [self.fitness(x) for x in current_population]
        self.best_objective_function = max(fitness_list)
        self.best_solution = current_population[fitness_list.index(max(fitness_list))]

        iter = 0
        while iter < self.max_iter:

            update_population = []
            while len(update_population) < len(current_population):

                try_num = 0  # 判断新增的解是否符合约束条件
                while try_num <= self.try_num_max:
                    parents = random.choices(current_population, weights=fitness_list, k=2)
                    child = self.crossover(parents[0], parents[1])
                    child = self.mutation(child)
                    if self.constraint(child):
                        update_population.append(child)
                        break
                    else:
                        try_num += 1
                if try_num > self.try_num_max:
                    parents = random.choices(current_population, weights=fitness_list, k=2)
                    update_population.append(parents[0])

            fitness_list = [self.fitness(x) for x in current_population]
            current_population = update_population
            iter += 1
            print(f"迭代轮数：{iter}, 本轮迭代最优的目标函数值为：{max(fitness_list)}")
            print(f"本轮迭代最优的解为：{current_population[fitness_list.index(max(fitness_list))]}")
            print('===================================')

            if max(fitness_list) > self.best_objective_function:
                self.best_objective_function = max(fitness_list)
                self.best_solution = current_population[fitness_list.index(max(fitness_list))]
                print('-----------------------------------------------------------------------')
                print(f"最优解更新：当前最优的目标函数值为：{self.best_objective_function}，最优解为{self.best_solution}")

        print(f"迭代结束，最优的目标函数值为：{self.best_objective_function}")
        print(f"最优的解为：{self.best_solution}")
        return


if __name__ == '__main__':

    knapsack_objective = ObjectiveFunctions.knapsack_objective
    constraint = Constraints.knapsack_constraint
    values = [2, 6, 5, 7, 8, 9, 4, 6, 8, 11, 7, 10, 5, 4, 5, 8]
    weights = [3, 4, 2, 5, 6, 7, 4, 5, 6, 9, 7, 8, 5, 4, 5, 6]
    max_num = [5, 3, 7, 8, 7, 7, 5, 4, 7, 3, 7, 6, 7, 8, 7, 8]
    max_weight = 400
    sma = SMAforknapsack(objective_function=knapsack_objective, values=values, constraint=constraint, weights=weights, max_num=max_num, max_weight=max_weight)
    sma.solve()