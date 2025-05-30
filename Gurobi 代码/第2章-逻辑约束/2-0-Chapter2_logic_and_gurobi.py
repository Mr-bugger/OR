from gurobipy import *

def and_logic(x1_value, x2_value):
    # 创建模型
    model = Model()
    model.setParam('OutputFlag', 0)
    
    # 定义变量
    x1 = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x1')
    x2 = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='x2')
    y = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='y')
    
    model.setObjective(y, GRB.MAXIMIZE)
    
    # 设置线性化约束
    model.addConstr(y <= x1)
    model.addConstr(y <= x2)
    model.addConstr(y >= x1 + x2 - 1)
    
    # 对x1和x2赋值
    model.addConstr(x1 == x1_value)
    model.addConstr(x2 == x2_value)
    
    model.optimize()
    return y.x

# 验证上述建模结果
x_values = [(1, 1), (0, 1), (0, 0)]
for idx, (x1, x2) in enumerate(x_values):
    y = and_logic(x1, x2)
    print(f'Case {idx}: x1 = {x1}, x2 = {x2} | y = {y}')