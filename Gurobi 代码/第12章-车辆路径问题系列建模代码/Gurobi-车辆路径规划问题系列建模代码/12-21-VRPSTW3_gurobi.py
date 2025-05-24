"""
booktitle: 《数学建模与数学规划：方法、案例及编程实战 Python+COPT/Gurobi实现》
name: 带软时间窗的车辆路径规划问题（VRPSTW3）- Gurobi - Python接口代码实现
author: 张一白
date: 2022-11-11
"""

from gurobipy import *

class Data(object):
    """
    存储算例数据的类
    """

    def __init__(self):
        self.node_num = 0               # 点的数量
        self.demand = []                # 客户点的需求
        self.dis_matrix = []            # 点的距离矩阵
        self.service_time = []          # 客户的服务时间
        self.early_time_window = []     # 客户的最早硬服务时间
        self.late_time_window = []      # 客户的最晚硬服务时间

    def read_and_print_data(self, path, data):
        """
        读取算例数据中前customer_num个顾客的数据。

        :param path: 文件路径
        :param customer_num: 顾客数量
        :return:
        """

        # 读取算例，只取了前20个客户点
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
        self.q = {}
        self.t = {}
        self.beta = {}
        self.alpha = {}

    def build_and_solve_model(self, data=None):
        """
        构建模型的类。

        :param data: 算例数据
        :return:
        """

        # 开始在Gurobi中建模
        try:
            self.model = Model("VRPSTW3")

            # 因为复制了配送中心点为{o，d}，所以点的总个数是N+1，默认不存在以{o}为终点和{d}为起点的弧。
            N = data.node_num + 1

            # 创建变量
            for i in range(N):
                # 创建车辆到达点i剩余容量变量q_i,是连续类型变量，算例设置的车辆容量最大为200，因只使用了其中20个点，将容量限制为100。
                self.q[i] = self.model.addVar(lb=0.0, ub=100, obj=0.0, vtype=GRB.CONTINUOUS, name="q_%d" % i)
                # 创建迟到时间变量，该变量仅与客户点相关，设定最大迟到时间为10，惩罚成本为0.5每分钟
                self.beta[i] = self.model.addVar(lb=0, ub=10, obj=0.5, vtype=GRB.CONTINUOUS, name="Beta_%d" % i)
                # 同理创建早到时间变量，早到一般没有迟到成本高，而且早到比较常见，因此设置成本为0.3，时间为15
                self.alpha[i] = self.model.addVar(lb=0, ub=15, obj=0.3, vtype=GRB.CONTINUOUS, name="Alpha_%d" % i)
                # 创建车辆到达时间变量，算例中给出了上下界，其中晚到的时间可以最大为10
                if i in range(1, N - 1):
                    self.t[i] = self.model.addVar(lb=data.early_time_window[i] - 15, ub=data.late_time_window[i] + 10,
                                                  obj=0.0, vtype=GRB.CONTINUOUS, name="t_%d" % i)
                else:
                    self.t[i] = self.model.addVar(lb=data.early_time_window[0], ub=data.late_time_window[0], obj=0.0,
                                                  vtype=GRB.CONTINUOUS, name="t_%d" % i)

                for j in range(1, N):
                    # 默认不存在点i到点i的弧，也不存在点{o}到点{d}的弧。
                    if i != j:
                        # 以{o}开始的弧
                        if i == 0 and j != (N - 1):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))
                        # 以{d}结尾的弧
                        elif i != 0 and j == (N - 1):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][0], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))
                        # 客户点之间的弧
                        elif i in range(1, N - 1) and j in range(1, N - 1):
                            self.x[i, j] = self.model.addVar(obj=data.dis_matrix[i][j], vtype=GRB.BINARY,
                                                             name="x_%d_%d" % (i, j))

            # 设置目标函数(如果最大化改成-1就可以了)
            self.model.modelsense = 1

            # 每个客户都必须被有且只有一辆车服务，使用可能离开点i的路径刻画
            for i in range(1, N - 1):
                expr = LinExpr()
                for j in range(1, N):
                    if i != j:
                        expr.addTerms(1, self.x[i, j])
                self.model.addConstr(expr == 1, name="Customer_%d" % (i))

            # 配送中心或车场发出的车辆要等于回来的车辆
            self.model.addConstr(
                sum(self.x[0, j] for j in range(1, N - 1)) - sum(self.x[i, N - 1] for i in range(1, N - 1)) == 0,
                "DepotFlowConstr")

            # 客户点的流平衡约束
            for j in range(1, N - 1):
                lh = LinExpr()
                rh = LinExpr()
                for i in range(N):
                    if i != j:
                        if i != N - 1:
                            lh.addTerms(1, self.x[i, j])
                        if i != 0:
                            rh.addTerms(1, self.x[j, i])
                self.model.addConstr(lh - rh == 0, "PointFlowConstr_%d" % j)

            # MTZ约束控制容量变化，并避免环路
            for i in range(1, N - 1):
                for j in range(1, N):
                    if i != j:
                        self.model.addConstr(self.q[i] + data.demand[i] - self.q[j] <= (1 - self.x[i, j]) * self.big_M,
                                             "Capacity_%d_%d" % (i, j))

            # 同理构建时间变化约束
            for i in range(0, N - 1):
                for j in range(1, N):
                    if i != j:
                        if i == 0:
                            if j == N - 1:
                                continue
                            else:
                                self.model.addConstr(
                                    self.t[i] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j] <= (
                                                1 - self.x[i, j]) * self.big_M,
                                    "Time1_%d_%d" % (i, j))
                                # self.model.addConstr(self.t[i] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j] >= (self.x[i, j]-1) * self.big_M, "Time2_%d_%d" % (i, j))
                        else:
                            if j == N - 1:
                                self.model.addConstr(
                                    self.t[i] + data.service_time[i] + data.dis_matrix[i][0] - self.t[j] <= (
                                                1 - self.x[i, j]) * self.big_M,
                                    "Time1_%d_%d" % (i, j))
                                # self.model.addConstr(self.t[i] + data.service_time[i] + data.dis_matrix[i][0] - self.t[j] >= (self.x[i, j]-1) * self.big_M, "Time2_%d_%d" % (i, j))
                            else:
                                self.model.addConstr(
                                    self.t[i] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j] <= (
                                                1 - self.x[i, j]) * self.big_M,
                                    "Time1_%d_%d" % (i, j))
                                self.model.addConstr(
                                    self.t[i] + data.service_time[i] + data.dis_matrix[i][j] - self.t[j] >= (
                                                self.x[i, j] - 1) * self.big_M,
                                    "Time2_%d_%d" % (i, j))

            # 软时间窗上界确定（晚到）
            for i in range(1, N - 1):
                self.model.addConstr(self.t[i] - self.beta[i] <= data.late_time_window[i], "TimeUB_%d" % i)
            # 软时间窗下界确定（早到）
            for i in range(1, N - 1):
                self.model.addConstr(self.t[i] + self.alpha[i] >= data.early_time_window[i], "TimeLB_%d" % i)

            log_file_name = 'VRPSTW3.log'
            self.model.setParam(GRB.Param.LogFile, log_file_name)       # 设置输出路径
            self.model.setParam(GRB.Param.MIPGap, 0)          # 设置 MIPGap 容差为 0
            self.model.optimize()                                     # 命令求解器进行求解

            # 打印最优路径
            print("==========================================")
            print(f"ObjVal: {self.model.ObjVal}")
            print("最优路径：")
            Count = 1
            for i in range(1, N - 1):
                flag = True
                if self.x[0, i].x >= 0.9:
                    print("第%d条路径为：" % Count, end="\n")
                    print("场站-客户%d" % i, end="")
                    if self.beta[i].x > 0:
                        print("(迟%d)-" % self.beta[i].x, end="")
                    elif self.alpha[i].x > 0:
                        print("(早%d)-" % self.alpha[i].x, end="")
                    else:
                        print("-", end="")
                    current_node = i
                    while flag:
                        for j in range(1, N):
                            if current_node != j and self.x[current_node, j].x >= 0.9:
                                if j != N - 1:
                                    print("客户%d" % j, end="")
                                    if self.beta[j].x > 0:
                                        print("(迟%d)-" % self.beta[j].x, end="")
                                    elif self.alpha[j].x > 0:
                                        print("(早%d)-" % self.alpha[j].x, end="")
                                    else:
                                        print("-", end="")
                                    current_node = j
                                else:
                                    print("场站", end="\n")
                                    flag = False
                                break
                    Count += 1

            # 导出VRPSTW3的模型
            self.model.write("VRPSTW3.lp")
        except GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))


if __name__ == "__main__":

    # 调用函数读取数据
    data = Data()
    path = 'r101_VRPTW.txt'
    data.read_and_print_data(path, data)

    # 建立模型并求解
    model_handler = Model_builder()
    model_handler.build_and_solve_model(data=data)
