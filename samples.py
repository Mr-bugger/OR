import numpy as np
from typing import Union, List, Callable
from math import floor, ceil


class ObjectiveFunctions:
    """目标函数集合"""

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
    def box_constraint(x: Union[float, List[float]], lower: float, upper: float) -> bool:
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