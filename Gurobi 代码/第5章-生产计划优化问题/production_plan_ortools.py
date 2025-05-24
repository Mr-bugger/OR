"""
booktitle: 《数学建模与数学规划：方法、案例及实战 Python+COPT/Gurobi实现》
name: 生产计划优化问题--ortools实现
author: 刘兴禄
date: 2019-10-15
"""

from ortools.linear_solver import pywraplp


class Instance:
    """定义算例数据"""

    def __init__(self):
        self.period_num = 7                      # 周期数
        self.raw_material_cost = 90              # 原材料成本
        self.unit_product_time = 5               # 单位产品需要的工时
        self.price = 300                         # 产品售价
        self.init_employee_num = 1000            # 1月初剩余的员工个数
        self.init_inventory = 15000              # 1月初的员工数量
        self.normal_unit_salary = 30             # 正常单位工时工资
        self.overtime_unit_salary = 40           # 加班单位工时工资
        self.work_day_num = 20                   # 每月工作的天数
        self.work_time_each_day = 8              # 员工每天的正常工作时间
        self.overtime_upper_limit = 20           # 每个工人每月的加班工时上限
        self.outsource_unit_cost = 200           # 外包单位成本
        self.unit_inventory_cost = 15            # 单位库存成本
        self.unit_shortage_cost = 35             # 单位缺货成本
        self.hire_cost = 5000                    # 单个工人的雇佣成本
        self.fire_cost = 8000                    # 单个工人的解雇成本
        self.inventory_LB_of_last_month = 10000   # 6月底的最低库存要求

        # 预测需求量（12月的用0补充）
        self.demand = [0, 20000, 40000, 42000, 35000, 19000, 18500]


def build_production_plan_model_and_solve(instance=None):

    """
    创建模型实例
    """
    solver = pywraplp.Solver.CreateSolver('SCIP')

    '''
    创建决策变量
    '''

    x, y, I, e, H, F, L, P, S, O, z = {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}

    for i in range(instance.period_num):
        x[i] = solver.IntVar(0, solver.infinity(), f'x_{i}')
        y[i] = solver.IntVar(0, solver.infinity(), f'y_{i}')
        I[i] = solver.IntVar(0, solver.infinity(), f'I_{i}')
        e[i] = solver.IntVar(-solver.infinity(), solver.infinity(), f'e_{i}')
        H[i] = solver.IntVar(0, solver.infinity(), f'H_{i}')
        F[i] = solver.IntVar(0, solver.infinity(), f'F_{i}')
        L[i] = solver.IntVar(0, solver.infinity(), f'L_{i}')
        P[i] = solver.IntVar(0, solver.infinity(), f'P_{i}')
        S[i] = solver.IntVar(0, solver.infinity(), f'S_{i}')
        O[i] = solver.NumVar(0, solver.infinity(), f'O_{i}')
        z[i] = solver.BoolVar(f'z_{i}')

    '''
    创建目标函数
    '''
    obj = solver.Objective()
    for i in range(1, instance.period_num):
        obj.SetCoefficient(S[i], instance.price)
        obj.SetCoefficient(x[i], -instance.raw_material_cost)
        obj.SetCoefficient(y[i], -instance.outsource_unit_cost)
        obj.SetCoefficient(O[i], -instance.overtime_unit_salary)
        obj.SetCoefficient(P[i], -instance.normal_unit_salary * instance.work_day_num * instance.work_time_each_day)
        obj.SetCoefficient(I[i], -instance.unit_inventory_cost)
        obj.SetCoefficient(L[i], -instance.unit_shortage_cost)
        obj.SetCoefficient(H[i], -instance.hire_cost)
        obj.SetCoefficient(F[i], -instance.fire_cost)
    obj.SetMaximization()

    '''
    添加约束
    '''
    # 约束1-5
    big_M = 1000000
    solver.Add(I[0] == instance.init_inventory)
    solver.Add(P[0] == instance.init_employee_num)
    solver.Add(L[0] == 0)
    solver.Add(S[0] == 0)
    solver.Add(I[instance.period_num - 1] >= instance.inventory_LB_of_last_month)
    solver.Add(F[0] == 0)
    solver.Add(e[0] == 0)
    solver.Add(x[0] == 0)
    solver.Add(y[0] == 0)
    solver.Add(z[0] == 0)
    solver.Add(O[0] == 0)

    # 约束6-14
    for i in range(1, instance.period_num):
        # 约束6
        solver.Add(I[i - 1] + x[i] + y[i] + e[i] == instance.demand[i], name='instance.demand_' + str(i))

        # 约束7
        solver.Add(I[i - 1] + x[i] + y[i] - S[i] == I[i], name='inventory_' + str(i))

        # 约束8
        solver.Add(e[i] - big_M * z[i] <= 0, name='shortage1_' + str(i))

        # 约束9
        solver.Add(1 - e[i] - big_M * (1 - z[i]) <= 0, name='shortage2_' + str(i))

        # 约束10
        solver.Add(L[i] - e[i] - big_M * (1 - z[i]) <= 0, name='shortage3_' + str(i))

        # 约束11
        solver.Add(e[i] - L[i] - big_M * (1 - z[i]) <= 0, name='shortage3_' + str(i))

        # 约束12
        solver.Add(S[i] == instance.demand[i] - L[i], name='sale_' + str(i))

        # 约束13
        solver.Add(P[i - 1] + H[i] - F[i] == P[i], name='employee_' + str(i))

        # 约束14
        solver.Add(
            instance.unit_product_time * x[i] <= P[i] * instance.work_time_each_day * instance.work_day_num + O[i],
            name='time_' + str(i))

        # 约束15
        solver.Add(O[i] <= instance.overtime_upper_limit * P[i], name='instance.overtime_upper_limit_' + str(i))

    '''
    求解模型并输出结果
    '''
    # 求解问题
    status = solver.Solve()

    # 输出结果
    if status == pywraplp.Solver.OPTIMAL:
        print('最优解已找到！')
        optimal_value = obj.Value() -instance.init_inventory * instance.unit_inventory_cost
        print(f'目标函数的最优值（包含常数）: {optimal_value}')

    print('详细计划为\n====================')
    print('|%2s|%4s|  %2s | %2s |  %2s  |%2s| %4s | %4s | %4s |%2s|%3s|%4s|%4s|'
          % ('月份', '期初库存', '生产', '外包', '差值', '缺货', '需求', '销售', '库存'
             , '雇佣', '解雇', '可用员工', '加班时长'))
    for i in range(0, instance.period_num):
        print('%2d' % (i), end='')
        if (i == 0):
            print(' %9d ' % (I[0].solution_value()), end='')
        else:
            print(' %9d ' % (I[i - 1].solution_value()), end='')
        print('%8d' % (x[i].solution_value()), end='')
        print('%7d' % (y[i].solution_value()), end='')
        print('%9d' % (e[i].solution_value()), end='')
        print('%5d' % (L[i].solution_value()), end='')
        print('%9d' % (instance.demand[i]), end='')
        print('%9d' % (S[i].solution_value()), end='')
        print('%9d' % (I[i].solution_value()), end='')
        print('%5d' % (H[i].solution_value()), end='')
        print('%6d' % (F[i].solution_value()), end='')
        print('%8d' % (P[i].solution_value()), end='')
        print('%8d' % (O[i].solution_value()), end='')
        print()


if __name__ == '__main__':
    # 生成算例
    instance = Instance()

    # 调用函数建模求解
    build_production_plan_model_and_solve(instance=instance)
