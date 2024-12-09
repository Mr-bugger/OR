import numpy as np
from typing import Union, List, Callable
from math import floor, ceil


class ObjectiveFunctions:
    """目标函数集合"""

    @staticmethod
    def knapsack_objective(quantities, values):
        """
        背包问题的目标函数，计算选定物品的总价值。

        :param values: 物品的价值列表
        :param quantities: 每个物品的数量列表
        :return: 选定物品的总价值
        """
        total_value = sum(value * quantity for value, quantity in zip(values, quantities))
        return total_value

    @staticmethod
    def sphere_function(x: Union[float, List[float]]) -> float:
        """球面函数 - 连续可导"""
        if isinstance(x, (int, float)):
            return x ** 2
        return sum([xi ** 2 for xi in x])

    @staticmethod
    def rosenbrock_function(x: List[float]) -> float:
        """Rosenbrock函数 - 连续可导"""
        return sum(100 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))

    @staticmethod
    def absolute_function(x: Union[float, List[float]]) -> float:
        """绝对值函数 - 连续但不可导"""
        if isinstance(x, (int, float)):
            return abs(x)
        return sum(abs(xi) for xi in x)

    @staticmethod
    def step_function(x: Union[float, List[float]]) -> float:
        """阶梯函数 - 非连续"""
        if isinstance(x, (int, float)):
            return floor(x)
        return sum(floor(xi) for xi in x)

    @staticmethod
    def mixed_function(x: Union[float, List[float]]) -> float:
        """混合函数 - 包含连续和非连续部分"""
        if isinstance(x, (int, float)):
            return abs(x) + floor(x / 2) + (x ** 2) / 2
        return sum(abs(xi) + floor(xi / 2) + (xi ** 2) / 2 for xi in x)

    @staticmethod
    def piecewise_function(x: Union[float, List[float]]) -> float:
        """分段函数"""

        def piece(val):
            if val < -1:
                return val ** 2
            elif val < 1:
                return abs(val)
            else:
                return val ** 2 + 1

        if isinstance(x, (int, float)):
            return piece(x)
        return sum(piece(xi) for xi in x)

    @staticmethod
    def discontinuous_periodic(x: Union[float, List[float]]) -> float:
        """不连续周期函数"""

        def single_val(val):
            return ceil(np.sin(val)) + abs(val)

        if isinstance(x, (int, float)):
            return single_val(x)
        return sum(single_val(xi) for xi in x)


class Constraints:
    """约束条件集合"""

    @staticmethod
    def knapsack_constraint(quantities: List[int], weights: List[float], max_num: List[int], max_weight:float):
        """
        背包问题的约束条件，检查选定物品的总重量是否超过最大容量。

        :param weights: 物品的重量列表
        :param selected_items: 选定物品的索引列表
        :param max_weight: 背包的最大容量
        :return: 如果总重量不超过最大容量，返回 True；否则返回 False
        """

        # 判断自变量的元素个数是否符合条件
        if len(quantities) != len(weights) or len(quantities) != len(max_num):
            raise ValueError(f"背包问题中，自变量列表元素个数为{len(quantities)}，重量列表的元素个数为{len(weights)}，不一致")

        # 判断每个物品的个数是否在最大值之内
        for i in range(len(quantities)):
            if quantities[i] > max_num[i]:
                return False

        # 判断总重量是否达标
        total_weight = sum(weight * quantity for weight, quantity in zip(weights, quantities))
        return total_weight <= max_weight

    @staticmethod
    def box_constraint(x: Union[float, List[float]], lower=0, upper=9999) -> bool:
        """框约束"""
        if isinstance(x, (int, float)):
            return lower <= x <= upper
        return all(lower <= xi <= upper for xi in x)

    @staticmethod
    def integer_constraint(x: Union[float, List[float]]) -> bool:
        """整数约束 - 非连续"""
        if isinstance(x, (int, float)):
            return abs(x - round(x)) < 1e-6
        return all(abs(xi - round(xi)) < 1e-6 for xi in x)

    @staticmethod
    def binary_constraint(x: Union[float, List[float]]) -> bool:
        """二值约束 - 离散"""
        if isinstance(x, (int, float)):
            return x in [0, 1]
        return all(xi in [0, 1] for xi in x)

    @staticmethod
    def modulo_constraint(x: Union[float, List[float]], mod_val: int = 3) -> bool:
        """模约束 - 离散"""
        if isinstance(x, (int, float)):
            return x % mod_val == 0
        return all(xi % mod_val == 0 for xi in x)

    @staticmethod
    def piecewise_constraint(x: Union[float, List[float]]) -> bool:
        """分段约束"""

        def check_single(val):
            if val < 0:
                return val >= -5
            else:
                return val <= 3

        if isinstance(x, (int, float)):
            return check_single(x)
        return all(check_single(xi) for xi in x)

    @staticmethod
    def periodic_constraint(x: Union[float, List[float]]) -> bool:
        """周期性约束"""

        def check_single(val):
            return np.sin(val) >= 0

        if isinstance(x, (int, float)):
            return check_single(x)
        return all(check_single(xi) for xi in x)

    @staticmethod
    def sum_constraint(x: List[float], threshold: float = 10) -> bool:
        """和约束"""
        return sum(abs(xi) for xi in x) <= threshold

    @staticmethod
    def alternating_constraint(x: List[float]) -> bool:
        """交替约束 - 要求相邻元素正负交替"""
        if len(x) < 2:
            return True
        return all((x[i] * x[i + 1]) <= 0 for i in range(len(x) - 1))


def test_functions():
    """测试函数示例"""
    obj = ObjectiveFunctions()
    cons = Constraints()

    # 测试非连续目标函数
    x1 = [1.5, -2.3, 0.7]
    print(f"阶梯函数值: {obj.step_function(x1)}")
    print(f"混合函数值: {obj.mixed_function(x1)}")
    print(f"分段函数值: {obj.piecewise_function(x1)}")
    print(f"不连续周期函数值: {obj.discontinuous_periodic(x1)}")

    # 测试约束条件
    x2 = [1, -2, 3]
    print(f"整数约束检查: {cons.integer_constraint(x2)}")
    print(f"模约束检查: {cons.modulo_constraint(x2)}")
    print(f"分段约束检查: {cons.piecewise_constraint(x2)}")
    print(f"周期性约束检查: {cons.periodic_constraint(x2)}")
    print(f"交替约束检查: {cons.alternating_constraint(x2)}")


if __name__ == "__main__":
    test_functions()