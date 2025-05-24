"""
booktitle: 《数学建模与数学规划：方法、案例及编程实战 Python+COPT/Gurobi实现》
name: 多车场车辆路径规划问题（MDVRP2）- Gurobi - Python接口代码实现
author: 张一白
date: 2022-11-11
"""

from gurobipy import *

class Data(object):
    """
    存储算例数据的类
    """

    def __init__(self):
        self.node_num = 0                   # 点的数量
        self.demand = []                    # 客户点的需求
        self.dis_matrix = []                # 点的距离矩阵
        self.depot_num = 2                  # depot数,设置前两个点为车场
        self.vehicle_num = [3, 1]           # 设计车场的最大发车数量，限制第二个车场只能发一辆车
        self.vehicle_set = 3                # 车辆总数为3个

    def read_and_print_data(self, path, data):
        """
        读取算例数据中前customer_num个顾客的数据。

        :param path: 文件路径
        :param customer_num: 顾客数量
        :return:
        """

        # 读取算例，只取了前14个客户点
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
        self.q = {}

    def build_and_solve_model(self, data=None):
        """
        构建模型的类。

        :param data: 算例数据
        :return:
        """
        # 开始在Gurobi中建模
        try:
            self.model = Model("MDVRP2")

            # 将第一个点和第二个点作为都作为车场。节点数复制车场后共有N+D个，客户数等于N-D个
            S = data.node_num - data.depot_num
            N = data.node_num + data.depot_num

            # 创建变量
            for k in range(data.vehicle_set):
                for i in range(N):
                    # 创建车辆到达点i剩余容量变量q_i,是连续类型变量，算例设置的车辆容量最大为200，因只使用了其中14个点，将容量限制为80。
                    self.q[i, k] = self.model.addVar(lb=0.0, ub=80, obj=0.0, vtype=GRB.CONTINUOUS,
                                                     name="q_%d_%d" % (i, k))
                for i in range(N - data.depot_num):
                    for j in range(data.depot_num, N):
                        # 默认不存在点i到点i的弧。
                        if i != j:
                            # 以下两个是排除车场到车场的变量
                            if i in range(data.depot_num):
                                if j in range(data.depot_num, N - data.depot_num):
                                    self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                                        name="x_%d_%d_%d" % (i, j, k))
                            else:
                                if j in range(data.depot_num, N - data.depot_num):
                                    self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                                        name="x_%d_%d_%d" % (i, j, k))
                                else:
                                    self.x[i, j, k] = self.model.addVar(obj=data.dis_matrix[i][j - S - data.depot_num],
                                                                        vtype=GRB.BINARY,
                                                                        name="x_%d_%d_%d" % (i, j, k))

            # 设置目标函数(如果最大化改成-1就可以了)
            self.model.modelsense = 1

            # 车场的发车数量不能超过其可发车的数量
            for i in range(0, data.depot_num):
                expr = LinExpr()
                for j in range(data.depot_num, N - data.depot_num):
                    if i != j:
                        for k in range(data.vehicle_set):
                            expr.addTerms(1, self.x[i, j, k])
                self.model.addConstr(expr <= data.vehicle_num[i], name="DepotCapa_%d" % (i))

            # 每个客户都必须被有且只有一辆车服务，使用可能离开点i的路径刻画
            for i in range(data.depot_num, N - data.depot_num):
                expr = LinExpr()
                for j in range(data.depot_num, N):
                    if i != j:
                        for k in range(data.vehicle_set):
                            expr.addTerms(1, self.x[i, j, k])
                self.model.addConstr(expr == 1, name="Customer_%d" % (i))

            # 配送中心或车场发出的车辆要等于回来的车辆
            for k in range(data.vehicle_set):
                # 每个车只能调用一次
                expr = LinExpr()
                for d in range(data.depot_num):
                    self.model.addConstr(
                        sum(self.x[d, j, k] for j in range(data.depot_num, N - data.depot_num)) - sum(
                            self.x[i, d + S + data.depot_num, k] for i in
                            range(data.depot_num, N - data.depot_num)) == 0,
                        "DepotFlowConstr_%d_%d" % (d, k))
                    for i in range(data.depot_num, N - data.depot_num):
                        expr.addTerms(1, self.x[d, i, k])
                self.model.addConstr(expr <= 1, "UseofVehicle_%d" % k)

            # 客户点的流平衡约束
            for k in range(data.vehicle_set):
                for i in range(data.depot_num, N - data.depot_num):
                    lh = LinExpr()
                    rh = LinExpr()
                    for j in range(data.depot_num, N):
                        if i != j:
                            rh.addTerms(1, self.x[i, j, k])
                    for j in range(N - data.depot_num):
                        if i != j:
                            lh.addTerms(1, self.x[j, i, k])
                    self.model.addConstr(lh - rh == 0, "PointFlowConstr_%d_%d" % (i, k))

            # 两种容量约束控制容量变化，并避免环路
            for k in range(data.vehicle_set):
                for i in range(data.depot_num, N - data.depot_num):
                    for j in range(data.depot_num, N):
                        if i != j:
                            self.model.addConstr(
                                self.q[i, k] + data.demand[i] - self.q[j, k] <= (1 - self.x[i, j, k]) * 200,
                                "Capacity_%d_%d_%d" % (i, j, k))

            log_file_name = 'MDVRP2.log'
            self.model.setParam(GRB.Param.LogFile, log_file_name)         # 设置输出路径
            self.model.setParam(GRB.Param.MIPGap, 0)            # 设置 MIPGap 容差为 0
            self.model.optimize()                                       # 命令求解器进行求解

            for k in range(data.vehicle_set):
                for i in range(N - data.depot_num):
                    for j in range(data.depot_num, N):
                        if i != j:
                            if i in range(data.depot_num):
                                if j in range(data.depot_num + S, N):
                                    continue
                                else:
                                    if self.x[i, j, k].x > 0.9:
                                        print("Name : %s , Value : %s " % (self.x[i, j, k].varName, self.x[i, j, k].x))
                            else:
                                if self.x[i, j, k].x > 0.9:
                                    print("Name : %s , Value : %s " % (self.x[i, j, k].varName, self.x[i, j, k].x))

            # 打印最优路径
            print("==========================================")
            print(f"ObjVal: {self.model.ObjVal}")
            print("最优路径：")
            Count = 1
            for k in range(data.vehicle_set):
                for d in range(data.depot_num):
                    for i in range(data.depot_num, N - data.depot_num):
                        flag = True
                        if self.x[d, i, k].x >= 0.9:
                            print("第%d条路径为：" % Count, end="\n")
                            print("场站%d-客户%d-" % (d, i), end="")
                            current_node = i
                            while flag:
                                I = current_node
                                for j in range(data.depot_num, N):
                                    if current_node != j and self.x[current_node, j, k].x >= 0.9:
                                        if j in range(data.depot_num, data.depot_num + S):
                                            print("客户%d-" % j, end="")
                                            current_node = j
                                        else:
                                            print("场站%d" % d, end="\n")
                                            flag = False
                            Count += 1

            # 导出MDVRP2的模型
            self.model.write("MDVRP2.lp")
        except GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))


if __name__ == "__main__":

    # 调用函数读取数据
    data = Data()
    path = 'r101_MDVRP.txt'
    data.read_and_print_data(path, data)

    # 建立模型并求解
    model_handler = Model_builder()
    model_handler.build_and_solve_model(data=data)
