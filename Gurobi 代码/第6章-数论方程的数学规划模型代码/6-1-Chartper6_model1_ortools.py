from ortools.linear_solver import pywraplp

# 创建模型对象
solver = pywraplp.Solver.CreateSolver("SCIP")

# 定义一个大的常数 M
M = 10000

# 创建决策变量（a,b,c为正整数）
a = solver.IntVar(1, solver.infinity(), name="a")
b = solver.IntVar(1, solver.infinity(), name="b")
c = solver.IntVar(1, solver.infinity(), name="c")
m1 = solver.NumVar(0, solver.infinity(), name="m1")
m2 = solver.NumVar(0, solver.infinity(), name="m2")
m3 = solver.NumVar(0, solver.infinity(), name="m3")

# 引入辅助变量和二进制变量
# 对于 a == m1 * (b + c)
z1 = solver.NumVar(0, solver.infinity(), name="z1")
z2 = solver.NumVar(0, solver.infinity(), name="z2")
# 对于 b == m2 * (a + c)
z3 = solver.NumVar(0, solver.infinity(), name="z3")
z4 = solver.NumVar(0, solver.infinity(), name="z4")
# 对于 c == m3 * (a + b)
z5 = solver.NumVar(0, solver.infinity(), name="z5")
z6 = solver.NumVar(0, solver.infinity(), name="z6")

# 添加约束
# 约束6.7
solver.Add(m1 + m2 + m3 - 4 == 0)

# 约束 a == m1 * (b + c) 的线性化
solver.Add(a == z1 + z2)
# 大 M 法约束
for i in range(1, M):
    y1 = solver.BoolVar(f"y1_{i}")
    solver.Add(z1 <= i * M * y1)
    solver.Add(z1 >= i * y1)
    solver.Add(m1 <= i + M * (1 - y1))
    solver.Add(m1 >= i - M * (1 - y1))
    solver.Add(b <= i * M * y1)
    solver.Add(b >= i * y1)

for i in range(1, M):
    y2 = solver.BoolVar(f"y2_{i}")
    solver.Add(z2 <= i * M * y2)
    solver.Add(z2 >= i * y2)
    solver.Add(m1 <= i + M * (1 - y2))
    solver.Add(m1 >= i - M * (1 - y2))
    solver.Add(c <= i * M * y2)
    solver.Add(c >= i * y2)

# 约束 b == m2 * (a + c) 的线性化
solver.Add(b == z3 + z4)
for i in range(1, M):
    y3 = solver.BoolVar(f"y3_{i}")
    solver.Add(z3 <= i * M * y3)
    solver.Add(z3 >= i * y3)
    solver.Add(m2 <= i + M * (1 - y3))
    solver.Add(m2 >= i - M * (1 - y3))
    solver.Add(a <= i * M * y3)
    solver.Add(a >= i * y3)

for i in range(1, M):
    y4 = solver.BoolVar(f"y4_{i}")
    solver.Add(z4 <= i * M * y4)
    solver.Add(z4 >= i * y4)
    solver.Add(m2 <= i + M * (1 - y4))
    solver.Add(m2 >= i - M * (1 - y4))
    solver.Add(c <= i * M * y4)
    solver.Add(c >= i * y4)

# 约束 c == m3 * (a + b) 的线性化
solver.Add(c == z5 + z6)
for i in range(1, M):
    y5 = solver.BoolVar(f"y5_{i}")
    solver.Add(z5 <= i * M * y5)
    solver.Add(z5 >= i * y5)
    solver.Add(m3 <= i + M * (1 - y5))
    solver.Add(m3 >= i - M * (1 - y5))
    solver.Add(a <= i * M * y5)
    solver.Add(a >= i * y5)

for i in range(1, M):
    y6 = solver.BoolVar(f"y6_{i}")
    solver.Add(z6 <= i * M * y6)
    solver.Add(z6 >= i * y6)
    solver.Add(m3 <= i + M * (1 - y6))
    solver.Add(m3 >= i - M * (1 - y6))
    solver.Add(b <= i * M * y6)
    solver.Add(b >= i * y6)

# 设置目标函数
objective = solver.Objective()
objective.SetMinimization()

# 求解问题
status = solver.Solve()
print('Solution:')
print('Objective value =', objective.Value())
print('a =', a.solution_value())
print('b =', b.solution_value())
print('c =', c.solution_value())
print('m1 =', m1.solution_value())
print('m2 =', m2.solution_value())
print('m3 =', m3.solution_value())

# 输出求解结果
if status == pywraplp.Solver.OPTIMAL:
    print('Solution:')
    print('Objective value =', objective.Value())
    print('a =', a.solution_value())
    print('b =', b.solution_value())
    print('c =', c.solution_value())
    print('m1 =', m1.solution_value())
    print('m2 =', m2.solution_value())
    print('m3 =', m3.solution_value())
else:
    print('The problem does not have an optimal solution.')
