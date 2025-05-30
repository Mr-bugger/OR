from gurobipy import *

# 分断点为(0, 0) , (20, 40) , (30, 55) , (40, 67)
x_n = [0, 20, 30, 40]
y_n = [0, 40, 55, 67]
N = len(x_n)

model = Model('Piece-wise Linear Function Example')

# 定义变量
x = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='x')
y = model.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='y')

# 定义权重决策变量
beta = {}
for n in range(N):
    beta[n] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='beta_' + str(n))

# 表示数据取值区间位置的变量
z = {}
for n in range(1, N):
    z[n] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name='z_' + str(n))

# 设置目标函数
model.setObjective(y, GRB.MAXIMIZE)

# 约束：x = b_0 * x_0 + b_1 * x_1 + ... + b_n * x_n
x_eq = LinExpr(0)
for n in range(N):
    x_eq.addTerms(x_n[n], beta[n])
model.addConstr(x == x_eq, name='x_eq')

# 约束：y = b_0 * y_0 + b_1 * y_1 + ... + b_n * y_n
y_eq = LinExpr(0)
for n in range(N):
    y_eq.addTerms(y_n[n], beta[n])
model.addConstr(y == y_eq, name='y_eq')

# 约束：beta_0 + beta_1 + ... + beta_n = 1
beta_eq = LinExpr(0)
for n in range(N):
    beta_eq.addTerms(1, beta[n])
model.addConstr(1 == beta_eq, name='beta_eq')

# 约束：z_0 + z_1 + ... + z_n = 1
z_eq = LinExpr(0)
for n in range(1, N):
    z_eq.addTerms(1, z[n])
model.addConstr(1 == z_eq, name='z_eq')

# 添加其他逻辑约束
model.addConstr(beta[0] <= z[1], name='Logic_' + str(0))
model.addConstr(beta[N - 1] <= z[N - 1], name='Logic_' + str(N - 1))
for n in range(1, N - 1):
    model.addConstr(beta[n] <= z[n] + z[n + 1], name='Logic_' + str(n))

# 设置 x 的取值
model.addConstr(x == 35)

# model.addGenConstrPWL(yvar=y, xvar=x, xpts=x_n, ypts=y_n)

model.optimize()
model.write('PWL.lp')

print('Optimal Obj: {}'.format(model.ObjVal))
print('y = {}'.format(y.x))
print('z1 = {}'.format(z[1].x))
print('z2 = {}'.format(z[2].x))
print('z3 = {}'.format(z[3].x))
print('beta0 = {}'.format(beta[0].x))
print('beta1 = {}'.format(beta[1].x))
print('beta2 = {}'.format(beta[2].x))
print('beta3 = {}'.format(beta[3].x))
