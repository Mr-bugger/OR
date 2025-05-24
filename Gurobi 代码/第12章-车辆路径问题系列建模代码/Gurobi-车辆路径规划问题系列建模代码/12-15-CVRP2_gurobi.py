"""
booktitle: 《数学建模与数学规划：方法、案例及编程实战 Python+COPT/Gurobi实现》
name: 带容量约束的车辆路径问题（CVRP2）- Gurobi - Python接口代码实现
author: 张一白
date: 2022-11-11
"""

from gurobipy import *

class Data(object):
    """
    存储算例数据的类
    """

    def __init__(self):
        self.node_num = 0       # 点的数量
        self.demand = []        # 客户点的需求
        self.dis_matrix = []    # 点的距离矩阵

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
        self.q = {}

    def build_and_solve_model(self, data=None):
        """
        构建模型的类。

        :param data: 算例数据
        :return:
        """

        # 开始在Gurobi中建模
        try:
            self.model = Model("CVRP2")

            # 创建变量
            # 因为复制了配送中心点为{o，d}，所以点的总个数是N+1，默认不存在以{o}为终点和{d}为起点的弧。
            for i in range(data.node_num + 1):
                # 创建车辆到达点i剩余容量变量q_i,是连续类型变量，算例设置的车辆容量最大为200，因只使用了其中15个点，将容量限制为100。
                self.q[i] = self.model.addVar(lb=0.0, ub=100, obj=0.0, vtype=GRB.CONTINUOUS, name="q_%d" % i)
            for i in range(data.node_num):
                for j in range(1, data.node_num + 1):
                    # 默认不存在点i到点i的弧，也不存在点{o}到点{d}的弧。
                    if i != j:
                        # 以{o}开始的弧
                        if i == 0 and j != (data.node_num):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))
                        # 以{d}结尾的弧
                        elif i != 0 and j == (data.node_num):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][0], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))
                        # 客户点之间的弧
                        elif i != 0 and i != (data.node_num) and j != 0 and j != (data.node_num):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))
            # 设置目标函数(如果最大化改成-1就可以了)
            self.model.modelsense = 1

            # 每个客户都必须被有且只有一辆车服务，使用可能离开点i的路径刻画
            for i in range(1, data.node_num):
                expr = LinExpr()
                for j in range(1, data.node_num + 1):
                    if i != j:
                        expr.addTerms(1, self.x[i, j])
                self.model.addConstr(expr == 1, name="Customer_%d" % (i))

            # 配送中心或车场发出的车辆要等于回来的车辆
            self.model.addConstr(sum(self.x[0, j] for j in range(1, data.node_num)) - sum(
                self.x[i, data.node_num] for i in range(1, data.node_num)) == 0,
                                 "DepotFlowConstr")

            # 客户点的流平衡约束
            for j in range(1, data.node_num):
                lh = LinExpr()
                rh = LinExpr()
                for i in range(data.node_num + 1):
                    if i != j:
                        if i != data.node_num:
                            lh.addTerms(1, self.x[i, j])
                        if i != 0:
                            rh.addTerms(1, self.x[j, i])
                self.model.addConstr(lh - rh == 0, "PointFlowConstr_%d" % j)

            # MTZ约束控制容量变化，并避免环路
            for i in range(1, data.node_num):
                for j in range(1, data.node_num + 1):
                    if i != j:
                        self.model.addConstr(self.q[i] + data.demand[i] - self.q[j] <= (1 - self.x[i, j]) * 200,
                                             "Capacity_%d_%d" % (i, j))

            log_file_name = 'CVRP2.log'
            self.model.setParam(GRB.Param.LogFile, log_file_name)       # 设置输出路径
            self.model.setParam(GRB.Param.MIPGap, 0)          # 设置 MIPGap 容差为 0
            self.model.optimize()                                     # 命令求解器进行求解

            # 打印最优路径
            print("==========================================")
            print(f"ObjVal: {self.model.ObjVal}")
            print("最优路径：")
            Count = 1
            for i in range(1, data.node_num):
                flag = True
                if self.x[0, i].x >= 0.9:
                    print("第%d条路径为：" % Count, end="\n")
                    print("场站-客户%d-" % i, end="")
                    current_node = i
                    while flag:
                        for j in range(1, data.node_num + 1):
                            if current_node != j and self.x[current_node, j].x >= 0.9:
                                if j != data.node_num:
                                    print("客户%d-" % j, end="")
                                    current_node = j
                                else:
                                    print("场站", end="\n")
                                    flag = False
                                break
                    Count += 1

            # 导出CVPR2的模型
            self.model.write("CVRP2.lp")
            self.model.write("CVRP2.mps")
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
