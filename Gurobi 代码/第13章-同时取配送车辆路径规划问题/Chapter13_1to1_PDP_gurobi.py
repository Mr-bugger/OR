"""
booktitle: 《数学建模与数学规划：方法、案例及实战 Python+COPT/Gurobi实现》
name: 1to1 PDP优化问题- Gurobi Python接口代码实现
author: 王基光
date: 2022-10-11
"""

from gurobipy import *
import copy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Data(object):
    def __init__(self):
        self.customer_num = 0
        self.node_num = 0
        self.vehicle_num = 0
        self.coor_X = {}
        self.coor_Y = {}
        self.dis_matrix = {}
        self.capacity = 0
        self.arcs = {}
        self.time_matrix = {}
        self.demand = {}

    def read_data(self, file_name, customer_num, vehicle_num):
        data = Data()
        np.random.seed(0)
        data.customer_num = customer_num
        data.node_num = 2 * customer_num + 2
        f = open(file_name, "r")
        lines = f.readlines()
        cnt = 0
        for line in lines:
            cnt += 1
            if (cnt == 5):
                line = line[:-1].strip()
                str_arr = re.split(r" +", line)
                data.vehicle_num = vehicle_num
                data.capacity = (int)(str_arr[1])
            if (cnt == 10):
                line = line[:-1].strip()
                str_arr = re.split(r" +", line)
                node_ID = (int)(str_arr[0])
                data.coor_X[node_ID] = (int)(str_arr[1])
                data.coor_Y[node_ID] = (int)(str_arr[2])
                data.demand[node_ID] = 0
            elif (cnt > 10 and cnt <= 10 + data.customer_num):
                line = line[:-1].strip()
                str_arr = re.split(r" +", line)
                node_ID = (int)(str_arr[0])
                data.coor_X[node_ID] = (int)(str_arr[1])
                data.coor_Y[node_ID] = (int)(str_arr[2])
                data.demand[node_ID] = (int)(str_arr[3])
            elif (cnt > 10 + data.customer_num and cnt <= 10 + 2 * data.customer_num):
                line = line[:-1].strip()
                str_arr = re.split(r" +", line)
                node_ID = (int)(str_arr[0])
                data.coor_X[node_ID] = (int)(str_arr[1])
                data.coor_Y[node_ID] = (int)(str_arr[2])
                data.demand[node_ID] = -data.demand[node_ID - data.customer_num]
        node_ID = data.node_num - 1
        data.coor_X[node_ID] = data.coor_X[0]
        data.coor_Y[node_ID] = data.coor_Y[0]
        data.demand[node_ID] = 0

        for i in range(data.node_num):
            for j in range(data.node_num):
                temp = (data.coor_X[i] - data.coor_X[j]) ** 2 + (data.coor_Y[i] - data.coor_Y[j]) ** 2
                data.dis_matrix[i, j] = round(math.sqrt(temp), 1)
                data.time_matrix[i, j] = round(math.sqrt(temp), 1)
                if (i != j):
                    data.arcs[i, j] = 1
                else:
                    data.arcs[i, j] = 0
        return data

    def printData(self, data):
        print("------数据集 信息--------------\n")
        print("车辆数 = %4d" % data.vehicle_num)
        print("顾客数 = %4d" % data.customer_num)
        print("节点数 = %4d" % data.node_num)
        for i in data.demand.keys():
            # print("%10.0f" %(data.demand[i]),"%10.0f"%data.pickdemand[i], "%10.0f"%data.ready_time[i], "%10.0f"%data.due_time[i],"%10.0f"%data.service_time[i])
            print("%10.0f" % (data.demand[i]))
        print("-------距离矩阵-------\n")
        for i in range(data.node_num):
            for j in range(data.node_num):
                print("%6.2f" % (data.dis_matrix[i, j]), end=" ")


def build_and_solve_1_1_PDP_model(data):
    """
    构建 1_1_PDP_model 并求解

    :param data:
    :return:
    """

    m = Model("1_1PDP")

    # 设置决策变量x
    x = {}
    for i in range(data.node_num - 1):
        for j in range(1, data.node_num):
            for k in range(vehicle_num):
                if i != j:
                    x[i, j, k] = m.addVar(
                        obj=data.dis_matrix[i, j], vtype=GRB.BINARY, name='x_' + str(i) + '_' + str(j) + '_' + str(k)
                    )  # 设置目标函数，默认为最小化

    # 设置决策变量 Q和辅助变量Q_F
    Q = {}
    Q_f = {}
    for i in range(data.node_num):
        for k in range(vehicle_num):
            Q[i, k] = m.addVar(lb=0, ub=data.capacity, vtype=GRB.CONTINUOUS, name='Q_' + str(i) + '_' + str(k))
            Q_f[i, k] = m.addVar(vtype=GRB.CONTINUOUS, name='Qf_' + str(i) + '_' + str(k))

    t = {}  # 设置决策变量t
    for i in range(data.node_num):
        for k in range(vehicle_num):
            t[i, k] = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='t_' + str(i) + '_' + str(k))

    # 添加约束保证起点不能直接连接到终点
    m.addConstr(quicksum(x[0, data.node_num - 1, k] for k in range(vehicle_num)) == 0)

    # 添加约束1：任务必须访问约束
    for i in range(1, 1 + data.customer_num):
        expr1 = LinExpr(0)
        for j in range(1, data.node_num):
            for k in range(vehicle_num):
                if i != j:
                    expr1.addTerms(1, x[i, j, k])
        m.addConstr(expr1 == 1, name='cons1_' + str(i))

    # 添加约束2：保证了取货任务和送货任务由同一辆车完成
    for i in range(1, 1 + data.customer_num):
        for k in range(vehicle_num):
            lhs = LinExpr(0)
            for j in range(1, data.node_num):
                if i != j:
                    lhs.addTerms(1, x[i, j, k])
            rhs = LinExpr(0)
            for j in range(1, data.node_num):
                if customer_num + i != j:
                    rhs.addTerms(1, x[customer_num + i, j, k])
            m.addConstr(lhs == rhs, name="con2_" + str(i) + "_" + str(k))

    # 添加约束3：必须发车约束
    for k in range(vehicle_num):
        lhs = LinExpr(0)
        for j in range(1, data.node_num - 1):
            lhs.addTerms(1, x[0, j, k])
        m.addConstr(lhs == 1, name="con3_" + str(k))

    # 添加约束4：必须返回约束
    for k in range(vehicle_num):
        lhs = LinExpr(0)
        for i in range(1, data.node_num - 1):
            lhs.addTerms(1, x[i, data.node_num - 1, k])
        m.addConstr(lhs == 1, name="con4_" + str(k))

    # 添加约束5：流平衡约束
    for i in range(1, data.node_num - 1):
        for k in range(vehicle_num):
            lhs = LinExpr(0)
            for j in range(1, data.node_num):
                if i != j:
                    lhs.addTerms(1, x[i, j, k])
            rhs = LinExpr(0)
            for j in range(data.node_num - 1):
                if i != j:
                    rhs.addTerms(1, x[j, i, k])
            m.addConstr(lhs == rhs, name='cons5_' + str(i) + "_" + str(k))

    #  添加约束6：时间连续约束; 添加约束8：容量连续约束
    for i in range(data.node_num - 1):
        for k in range(vehicle_num):
            for j in range(1, data.node_num):
                if i != j:
                    m.addConstr(t[j, k] >= (data.time_matrix[i, j] + t[i, k]) * x[i, j, k],
                                name="cons6_" + str(i) + "_" + str(j) + "_" + str(k))
                    m.addConstr(Q[j, k] >= (Q[i, k] + data.demand[j]) * x[i, j, k],
                                name="cons8_" + str(i) + "_" + str(j) + "_" + str(k))  # 容量连续约束

    #  添加约束7：访问每个配送需求的取货点和送货点的顺序，即先取货，后送货
    for i in range(1, 1 + data.customer_num):
        for k in range(vehicle_num):
            m.addConstr(t[i + customer_num, k] - t[i, k] >= data.time_matrix[i, i + customer_num],name="cons7_"+str(i)+"_"+str(k))

    #  添加约束9：载荷和容量约束,并引入辅助变量线性化
    for i in range(data.node_num):
        for k in range(vehicle_num):
            m.addConstr(Q[i, k] >= data.demand[i],name="cons9_1_"+str(i)+"_"+str(k))
            m.addConstr(Q[i, k] >= 0,name="cons9_2_"+str(i)+"_"+str(k))
            m.addConstr(Q_f[i, k] == min_(data.capacity, data.capacity + data.demand[i]),name="cons9_3_"+str(i)+"_"+str(k))
            m.addConstr(Q[i, k] <= Q_f[i, k],name="cons9_4_"+str(i)+"_"+str(k))

    # 设置求解算法参数
    m.Params.MIPFocus = 3
    m.Params.NonConVex = 2
    m.params.MIPGap = 0
    m.Params.lazyConstraints = 1
    m.setParam(GRB.Param.LogFile, "gurobi_1_1pdp.log")

    # 求解模型
    m.optimize()

    # 打印求解结果,记录和打印关键决策变量的解结果
    Slist = []
    S = []
    S.append(0)
    for i in range(data.node_num - 1):
        for j in range(1, data.node_num):
            for k in range(vehicle_num):
                if i != j and x[i, j, k].x > 0.1:
                    print(x[i, j, k].VarName, "=", x[i, j, k].X)

    for k in range(vehicle_num):
        for i in range(data.node_num):
            print(Q[i, k].VarName, "=", Q[i, k].X)

    for k in range(vehicle_num):
        for i in range(data.node_num):
            print(t[i, k].VarName, "=", t[i, k].X)

    # 拼接并打印最优路径
    print("==========================================")
    print(f"ObjVal: {m.ObjVal}")
    print("最优路径：")
    for i in range(1, data.node_num):
        for k in range(vehicle_num):
            if i not in S and x[0, i, k].x > 1 - 1e-3:
                print("[0-" + str(i), end='')
                currNode = i
                S.append(currNode)
                flag = True
                while flag:
                    flag = False
                    for j in range(1, data.node_num):
                        if j not in S and currNode != data.node_num - 1 and x[currNode, j, k].x > 1 - 1e-3:
                            print("-" + str(j), end='')
                            currNode = j
                            S.append(currNode)
                            flag = True
                            break
                print("]")
                Slist.append(S)
                S = [0]

    for i in range(len(Slist)):
        Slist[i].remove(data.node_num - 1)

    # 最优路径可视化，构建网络图并添加节点和弧段
    Graph = nx.DiGraph()
    nodes_name = [0]
    cor_xy = [[data.coor_X[0], data.coor_Y[0]]]
    edges = []

    # 拼接每个车的路径
    for route in Slist:
        edge = []
        edges.append([route[0], route[1]])
        for i in route[1:]:
            nodes_name.append(i)
            cor_xy.append([data.coor_X[i], data.coor_Y[i]])
            edge.append(i)
            if len(edge) == 2:
                edges.append(copy.deepcopy(edge))
                edge.pop(0)
        edge.append(0)
        edges.append(edge)

    pos_location = {nodes_name[i]: x for i, x in enumerate(cor_xy)}
    pos_location = sorted(pos_location.items(), key=lambda x: x[0])

    NEWpos_location = {}
    for i in range(len(pos_location)):
        NEWpos_location[i] = pos_location[i][1]
    nodes_name = sorted(nodes_name)
    Graph.add_nodes_from(nodes_name)
    Graph.add_edges_from(edges)
    nodes_color_dict = ["orangered"]

    # 为起点和终点设置不同的颜色
    for i in range(1, data.node_num - 1):
        if i < 1 + data.customer_num:
            nodes_color_dict.append("lightseagreen")
        else:
            nodes_color_dict.append("deepskyblue")

    # 设置备选颜色
    colorpool = ["turquoise", "slateblue", "cyan", "peru", "gold", "green"]
    edge_color_dict0 = []
    for edge in Graph.edges():
        for typei in range(len(Slist)):
            if edge[0] in Slist[typei] and edge[1] in Slist[typei]:
                edge_color_dict0.append(colorpool[typei])

    e_labels = {}
    for edge0 in edges:
        e_labels[(edge0[0], edge0[1])] = data.dis_matrix[edge0[0], edge0[1]]

    # 绘图
    nx.draw_networkx(Graph, NEWpos_location, node_size=200, node_color=nodes_color_dict, edge_color=edge_color_dict0,
                     labels=None,
                     font_size=8)

    # 保存图片并展示
    plt.savefig("fig_pdp1-1.pdf", dpi=800)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    file_path = "c101.txt"      # 设置数据文件
    data = Data()
    customer_num = 8            # 设置订单需求数量
    vehicle_num = 3             # 设置车辆数量
    data = data.read_data(file_name=file_path, customer_num=customer_num, vehicle_num=vehicle_num)  # 读取数据
    data.capacity = 100          # 设置车辆载荷
    data.printData(data)

    # 调用函数建模和求解
    build_and_solve_1_1_PDP_model(data=data)
