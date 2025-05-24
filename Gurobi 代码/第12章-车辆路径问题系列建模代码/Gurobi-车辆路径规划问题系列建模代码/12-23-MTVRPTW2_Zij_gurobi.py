"""
booktitle: 《数学建模与数学规划：方法、案例及编程实战 Python+COPT/Gurobi实现》
name: 带时间窗的多行程车辆路径问题（MTVRPTW2_zij）- Gurobi - Python接口代码实现
author: 张一白
date: 2022-11-11
"""

from gurobipy import *

class Data(object):
    """
    存储算例数据的类
    """
    def __init__(self):
        self.node_num = 0              # 点的数量
        self.demand = []               # 客户点的需求
        self.dis_matrix = []           # 点的距离矩阵
        self.service_time = []         # 客户的服务时间
        self.early_time_window = []    # 客户的最早硬服务时间
        self.late_time_window = []     # 客户的最晚硬服务时间
        self.vehicle_route_num = 7     # 设置车程数，最差应为N-1个，但会增加模型的对称性，这里设置的车程数小一些
        self.vehicle_num = 7           # 同理设置车辆数，车辆数比车程数略小，这两个参数大了都会使得模型臃肿
        self.vehicle_reload_time = 1   # 设置重新装货时间
        self.vehicle_fixed_cost = 300  # 设置车辆使用成本

    def read_and_print_data(self, path, data):
        """
        读取算例数据中前customer_num个顾客的数据。

        :param path: 文件路径
        :param customer_num: 顾客数量
        :return:
        """

        # 读取算例，只取了前10个客户点
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

        # 读取算例中的需求列和服务时间列
        data.demand = [ori_data[i][3] for i in range(data.node_num)]
        data.service_time = [ori_data[i][6] for i in range(data.node_num)]

        # 存取算例中的时间窗
        data.early_time_window = [ori_data[i][4] for i in range(data.node_num)]
        data.late_time_window = [ori_data[i][5] for i in range(data.node_num)]

class Model_builder(object):
    """
    构建模型的类。
    """
    def __init__(self):
        self.model = None
        self.big_M = 10000
        self.x = {}
        self.y = {}
        self.z = {}
        self.f = {}
        self.q = {}
        self.t = {}

    def build_and_solve_model(self, data=None):
        """
        构建模型的类。

        :param data: 算例数据
        :return:
        """

        # 开始在Gurobi中建模
        try:
            self.model = Model("MTVRPTW2")

            # 创建变量
            self.x = {}
            self.y = {}
            self.z = {}
            self.f = {}
            self.q = {}
            self.t = {}

            # 因为复制了配送中心点为{o，d}，所以点的总个数是N+1，默认不存在以{o}为终点和{d}为起点的弧。
            N = data.node_num + 1

            # 创建路径决策变量x_{ijr}
            for r in range(data.vehicle_route_num):
                for i in range(N - 1):
                    for j in range(1, N):
                        # 默认不存在点i到点i的弧，也不存在点{o}到点{d}的弧。
                        if i != j:
                            # 以{o}开始的弧
                            if i == 0 and j != (N - 1):
                                self.x[i, j, r] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                      name="x_%d_%d_%d" % (i, j, r))
                            # 以{d}结尾的弧
                            elif i != 0 and j == (N - 1):
                                self.x[i, j, r] = self.model.addVar(obj=data.dis_matrix[i][0], vtype=GRB.BINARY,
                                                      name="x_%d_%d_%d" % (i, j, r))
                            # 客户点之间的弧
                            elif i in range(1, N - 1) and j in range(1, N - 1):
                                self.x[i, j, r] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                      name="x_%d_%d_%d" % (i, j, r))

            # 创建分配车程的变量y_{kr}
            for r in range(data.vehicle_route_num):
                for k in range(data.vehicle_num):
                    self.y[k, r] = self.model.addVar(obj=0.0, vtype=GRB.BINARY, name="y_%d_%d" % (k, r))

            # 创建车程排序变量z_{ijk},以及车辆是否被使用的变量f_k
            for k in range(data.vehicle_num):
                self.f[k] = self.model.addVar(obj=data.vehicle_fixed_cost, vtype=GRB.BINARY, name="f_%d" % k)
            for i in range(data.vehicle_route_num):
                for j in range(data.vehicle_route_num):
                    if i != j:
                        self.z[i, j] = self.model.addVar(obj=0.0, vtype=GRB.BINARY, name="z_%d_%d" % (i, j))

            # 设置时间变量和剩余容量变量
            for i in range(N):
                # 创建车辆到达点i剩余容量变量q_i,是连续类型变量，算例设置的车辆容量最大为200，因只使用了其中10个点，将容量限制为75。
                self.q[i] = self.model.addVar(lb=0.0, ub=75, obj=0.0, vtype=GRB.CONTINUOUS, name="q_%d" % i)
                for r in range(data.vehicle_route_num):
                    # 创建车辆到达时间变量，算例中给出了上下界
                    if i in range(1, N - 1):
                        self.t[i, r] = self.model.addVar(lb=data.early_time_window[i], ub=data.late_time_window[i], obj=0.0, vtype=GRB.CONTINUOUS, name="t_%d_%d" % (i, r))
                    else:
                        self.t[i, r] = self.model.addVar(lb=data.early_time_window[0], ub=data.late_time_window[0], obj=0.0, vtype=GRB.CONTINUOUS, name="t_%d_%d" % (i, r))

            # 设置目标函数(如果最大化改成-1就可以了)
            self.model.modelsense = 1

            # 每个客户都必须被有且只有一辆车服务，使用可能离开点i的路径刻画
            for i in range(1, N - 1):
                expr = LinExpr()
                for j in range(1, N):
                    if i != j:
                        for r in range(data.vehicle_route_num):
                            expr.addTerms(1, self.x[i, j, r])
                self.model.addConstr(expr == 1, name="Customer_%d" % (i))

            # 配送中心或车场发出的车辆要等于回来的车辆
            for r in range(data.vehicle_route_num):
                self.model.addConstr(
                    sum(self.x[0, j, r] for j in range(1, N - 1)) - sum(self.x[i, N - 1, r] for i in range(1, N - 1)) == 0,
                    "DepotFlowConstr_%d" % r)

            # 客户点的流平衡约束
            for r in range(data.vehicle_route_num):
                for j in range(1, N - 1):
                    lh = LinExpr()
                    rh = LinExpr()
                    for i in range(N):
                        if i != j:
                            if i != N - 1:
                                lh.addTerms(1, self.x[i, j, r])
                            if i != 0:
                                rh.addTerms(1, self.x[j, i, r])
                    self.model.addConstr(lh - rh == 0, "PointFlowConstr_%d_%d" % (j, r))

            # MTZ约束控制容量变化，并避免环路
            for r in range(data.vehicle_route_num):
                for i in range(1, N - 1):
                    for j in range(1, N):
                        if i != j:
                            self.model.addConstr(self.q[i] + data.demand[i] - self.q[j] <= (1 - self.x[i, j, r]) * self.big_M,
                                        "Capacity_%d_%d_%d" % (i, j, r))

            # 同理构建时间变化约束
            for r in range(data.vehicle_route_num):
                for i in range(0, N - 1):
                    for j in range(1, N):
                        if i != j:
                            if i == 0:
                                if j == N - 1:
                                    continue
                                else:
                                    self.model.addConstr(self.t[i, r] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j, r] <= (1 - self.x[i, j, r]) * self.big_M,
                                                "Time_%d_%d_%d" % (i, j, r))
                            else:
                                if j == N - 1:
                                    self.model.addConstr(self.t[i, r] + data.service_time[i] + data.dis_matrix[i][0] - self.t[j, r] <= (1 - self.x[i, j, r]) * self.big_M,
                                                "Time_%d_%d_%d" % (i, j, r))
                                else:
                                    self.model.addConstr(self.t[i, r] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j, r] <= (1 - self.x[i, j, r]) * self.big_M,
                                                "Time_%d_%d_%d" % (i, j, r))

            # 定义一个车程是一条路径约束
            for r in range(data.vehicle_route_num):
                lh = LinExpr()
                for j in range(1, N - 1):
                    lh.addTerms(1.0, self.x[0, j, r])
                self.model.addConstr(lh <= 1.0, "RouteBound_%d" % r)

            # 车程分配约束
            for r in range(data.vehicle_route_num):
                lh = LinExpr()
                for k in range(data.vehicle_num):
                    lh.addTerms(1.0, self.y[k, r])
                for j in range(1, N - 1):
                    lh.addTerms(-1.0, self.x[0, j, r])
                self.model.addConstr(lh == 0, "AssignmentEqual_%d" % r)

            # 车辆使用约束
            for k in range(data.vehicle_num):
                lh = LinExpr()
                for r in range(data.vehicle_route_num):
                    lh.addTerms(1.0, self.y[k, r])
                self.model.addConstr(lh <= self.f[k] * self.big_M, "UseVehicle_%d" % k)

            # 表示非紧前的M约束
            for k in range(data.vehicle_num):
                for i in range(data.vehicle_route_num):
                    for j in range(data.vehicle_route_num):
                        if i != j:
                            self.model.addConstr(self.t[N - 1, i] + data.vehicle_reload_time - self.t[0, j] <= (3 - self.z[i, j] - self.y[k, i] - self.y[k, j]) * self.big_M,
                                        "DisjunctiveLeft_%d_%d_%d" % (i, j, k))
                            self.model.addConstr(self.t[0, i] - data.vehicle_reload_time - self.t[N - 1, j] >= -(2 + self.z[i, j] - self.y[k, i] - self.y[k, j]) * self.big_M,
                                        "DisjunctiveLeft_%d_%d_%d" % (i, j, k))


            log_file_name = 'MTVRPTW2.log'
            self.model.setParam(GRB.Param.LogFile, log_file_name)       # 设置输出路径
            self.model.setParam(GRB.Param.MIPGap, 0)          # 设置 MIPGap 容差为 0
            self.model.optimize()                                     # 命令求解器进行求解

            # 打印最优路径
            print("==========================================")
            print(f"ObjVal: {self.model.ObjVal}")
            print("最优路径：")
            for k in range(data.vehicle_num):
                if self.f[k].x >= 0.9:
                    print("-------------第%d辆车-------------" % k, end="\n")
                    for r in range(data.vehicle_route_num):
                        if self.y[k, r].x >= 0.9:
                            Count = 1
                            for i in range(1, N - 1):
                                flag = True
                                if self.x[0, i, r].x >= 0.9:
                                    print("第%d条路径为：" % Count, end="\n")
                                    print("场站(%s)-客户%d-" % (round(self.t[0, r].x, 2), i), end="")
                                    current_node = i
                                    while flag:
                                        for j in range(1, N):
                                            if current_node != j and self.x[current_node, j, r].x >= 0.9:
                                                if j != N - 1:
                                                    print("客户%d-" % j, end="")
                                                    current_node = j
                                                else:
                                                    print("场站(%s)" % round(self.t[j, r].x, 2), end="\n")
                                                    flag = False
                                                break
                                    Count += 1
            # 导出MTVRPTW2的模型
            self.model.write("MTVRPTW2.lp")
        except GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))

if __name__ == "__main__":

    # 调用函数读取数据
    data = Data()
    path = 'r101_35.txt'
    data.read_and_print_data(path, data)

    # 建立模型并求解
    model_handler = Model_builder()
    model_handler.build_and_solve_model(data=data)
