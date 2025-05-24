"""
booktitle: 《数学建模与数学规划：方法、案例及编程实战 Python+COPT/Gurobi实现》
name: 带容量约束的车辆路径问题（CVRP1-1）- Gurobi - Python接口代码实现
author: 张一白
date: 2022-11-11
"""

import numpy as np
import copy
from gurobipy import *

# 得到决策变量 x 的取值
def getValue(var_dict, nodeNum):
    x_value = np.zeros([nodeNum + 1, nodeNum + 1, K])
    for key in var_dict.keys():
        i = key[0]
        j = key[1]
        k = key[2]
        x_value[i][j][k] = var_dict[key].x
    return x_value

# 根据解得到相应的行驶路径
def getRoute(x_value):
    x = copy.deepcopy(x_value)
    route_list = {}
    for k in range(data.vehicle_num):
        Route = []
        route_list[k] = Route
        flag = True
        for i in range(1, data.node_num):
            R = []
            for j in range(1, data.node_num):
                if i != j and x[i][j][k] >= 0.01:
                    R.append(i)
                    R.append(j)

                    current_node = j
                    Count = 1
                    while (flag != False):
                        for l in range(1, data.node_num + 1):
                            if current_node != l and x[current_node][l][k] >= 0.01:

                                R.append(l)
                                if R[0] == l:
                                    flag = False
                                    Route.append(R)
                                    break
                                if l == data.node_num:
                                    flag = False
                                    break
                                current_node = l
                        Count += 1
        route_list[k] = Route

    return route_list


# 定义删除子环路的 callback
def subtourlim(model, where):
    if (where == GRB.Callback.MIPSOL):
        x_value = np.zeros([data.node_num + 1, data.node_num + 1, data.vehicle_num])
        for m in model.getVars():
            if (m.varName.startswith('x')):
                a = (int)(m.varName.split('_')[1])
                b = (int)(m.varName.split('_')[2])
                c = (int)(m.varName.split('_')[3])
                x_value[a][b][c] = model.cbGetSolution(m)

        # 找到子环路
        tour = getRoute(x_value)
        print('tour = ', tour)

        # 添加消除子环路约束
        for k in range(data.vehicle_num):
            for r in range(len(tour[k])):
                tour[k][r].remove(tour[k][r][0])
                expr = LinExpr()
                for i in range(len(tour[k][r])):
                    for j in range(len(tour[k][r])):
                        if tour[k][r][i] != tour[k][r][j]:
                            expr.addTerms(1.0, model_handler.x[tour[k][r][i], tour[k][r][j], k])
                model.cbLazy(expr <= len(tour[k][r]) - 1)


class Data(object):
    """
    存储算例数据的类
    """

    def __init__(self):
        self.node_num = 0       # 点的数量
        self.demand = []        # 客户点的需求
        self.dis_matrix = []    # 点的距离矩阵
        self.vehicle_num = 3    # 车辆数

    def read_and_print_data(self, path, data):
        """
        读取算例数据中前customer_num个顾客的数据。

        :param path: 文件路径
        :param customer_num: 顾客数量
        :return:
        """

        # 读取算例，只取了前15个点
        f = open(path, 'r')
        sth = f.readlines()

        ori_data = []
        for i in sth:
            item = i.strip("\n").split()
            ori_data.append(item)
        data.node_num = len(ori_data)
        for i in range(data.node_num):
            for j in range(len(ori_data[i])):
                print(ori_data[i][j], end="\t\t")
                ori_data[i][j] = int(ori_data[i][j])
            print()
        print("------------------------------------------")

        # 计算距离矩阵，保留两位小数，并打印矩阵
        data.dis_matrix = [
            [round(math.sqrt(sum((ori_data[i][k] - ori_data[j][k]) ** 2 for k in range(1, 3))), 2) for i in
             range(data.node_num)]
            for j in range(data.node_num)]
        for i in range(len(data.dis_matrix)):
            for j in range(len(data.dis_matrix[i])):
                if i != j:
                    print(data.dis_matrix[i][j], end="\t\t")
            print()

        # 读取算例中的需求列
        data.demand = [ori_data[i][3] for i in range(data.node_num)]


class Model_builder(object):
    """
    构建模型的类。
    """

    def __init__(self):
        self.model = None
        self.big_M = 10000
        self.x = {}

    def build_and_solve_model(self, data=None):
        """
        构建模型的类。

        :param data: 算例数据
        :return:
        """

        # 开始在Gurobi中建模
        try:
            self.model = Model("CVRP1-1")
            # 创建变量
            # 因为复制了配送中心点为{o，d}，所以点的总个数是N+1，默认不存在以{o}为终点和{d}为起点的弧。
            for k in range(data.vehicle_num):
                for i in range(data.node_num):
                    for j in range(1, data.node_num + 1):
                        # 默认不存在点i到点i的弧，也不存在点{o}到点{d}的弧。
                        if i != j:
                            # 以{o}开始的弧
                            if i == 0 and j != data.node_num:
                                self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                                    name="x_%d_%d_%d" % (i, j, k))
                            # 以{d}结尾的弧
                            elif i != 0 and j == data.node_num:
                                self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][0], vtype=GRB.BINARY,
                                                                    name="x_%d_%d_%d" % (i, j, k))
                            # 客户点之间的弧
                            elif i != 0 and i != data.node_num and j != 0 and j != data.node_num:
                                self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                                    name="x_%d_%d_%d" % (i, j, k))

            # 设置目标函数(如果最大化改成-1就可以了)
            self.model.modelsense = 1

            # 每个客户都必须被有且只有一辆车服务，使用可能离开点i的路径刻画
            for i in range(1, data.node_num):
                expr = LinExpr()
                for k in range(data.vehicle_num):
                    for j in range(1, data.node_num + 1):
                        if i != j:
                            expr.addTerms(1, self.x[i, j, k])
                self.model.addConstr(expr == 1, name="Customer_%d_%d" % (i, k))

            # 配送中心或车场发出的车辆要等于回来的车辆
            for k in range(data.vehicle_num):
                self.model.addConstr(sum(self.x[0, j, k] for j in range(1, data.node_num)) - sum(
                    self.x[i, data.node_num, k] for i in range(1, data.node_num)) == 0,
                                     "DepotFlowConstr1_%d" % k)
                self.model.addConstr(sum(self.x[0, j, k] for j in range(1, data.node_num)) <= 1,
                                     "DepotFlowConstr2_%d" % k)

            # 客户点的流平衡约束
            for k in range(data.vehicle_num):
                for j in range(1, data.node_num):
                    lh = LinExpr()
                    rh = LinExpr()
                    for i in range(data.node_num + 1):
                        if i != j:
                            if i != data.node_num:
                                lh.addTerms(1, self.x[i, j, k])
                            if i != 0:
                                rh.addTerms(1, self.x[j, i, k])
                    self.model.addConstr(lh - rh == 0, "PointFlowConstr_%d_%d" % (j, k))

            # 容量约束
            for k in range(data.vehicle_num):
                lh = LinExpr()
                for i in range(1, data.node_num):
                    for j in range(1, data.node_num + 1):
                        if i != j:
                            lh.addTerms(data.demand[i], self.x[i, j, k])
                self.model.addConstr(lh <= 100, "Capacity_%d" % k)

            log_file_name = 'CVRP1-1.log'
            self.model.setParam(GRB.Param.LogFile, log_file_name)       # 设置求解日志的输出路径
            self.model.setParam(GRB.Param.MIPGap, 0)          # 设置 MIPGap 容差为 0
            self.model.Params.lazyConstraints = 1                       # 打开 lazyConstraints

            self.model.optimize(subtourlim)                            # 命令求解器进行求解

            # 如果模型不可行，调用下面的函数进行 debug
            # self.model.computeIIS()
            # self.model.write("model.ilp")

            # 打印最优路径
            print("==========================================")
            print(f"ObjVal: {self.model.ObjVal}")
            print("最优路径：")
            Count = 1
            for k in range(data.vehicle_num):
                for i in range(1, data.node_num):
                    A = True
                    if self.x[0, i, k].x >= 0.9:
                        print("第%d条路径为：" % Count, end="\n")
                        print("场站-客户%d-" % i, end="")
                        current_node = i
                        while (A):
                            for j in range(1, data.node_num + 1):
                                if current_node != j and self.x[current_node, j, k].x >= 0.9:
                                    if j != data.node_num:
                                        print("客户%d-" % j, end="")
                                        current_node = j
                                    else:
                                        print("场站", end="\n")
                                        A = False
                                    break
                        Count += 1

            # 导出CVPR1-1的模型
            self.model.write("CVRP1-1.lp")

        except GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))


if __name__ == "__main__":

    # 调用函数读取数据
    data = Data()
    path = 'r101_CVRP.txt'
    data.read_and_print_data(path, data)

    # 建立模型并求解
    model_handler = Model_builder()
    model_handler.build_and_solve_model(data=data)
