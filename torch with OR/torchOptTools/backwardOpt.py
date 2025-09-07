#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2025/9/2 17:15
# @Author  : xueyudian
# @File    : backwardOpt.py
# @Project : OR

import torch
import numpy as np
from typing import Dict, Callable, Optional, Union


class GradientModifier:
    """
    梯度修改工具类，支持多种梯度调整策略以优化训练效果
    支持：梯度裁剪、阈值过滤、缩放、稀疏化、平滑、不可导函数梯度替换
    已修复：非叶子张量梯度保留警告、梯度调用参数不匹配问题
    """

    def __init__(self):
        # 不可导函数梯度替换规则：{函数标识: 梯度替换函数}
        self.undef_func_grad_rules: Dict[str, Callable] = {}
        # 需要特殊处理的张量（自动为非叶子张量启用retain_grad）
        self.special_tensors = set()

    def register_undef_func_rule(self, func_name: str, grad_func: Callable):
        """注册不可导函数的梯度替换规则"""
        self.undef_func_grad_rules[func_name] = grad_func
        print(f"✅ 已注册不可导函数规则：func_name={func_name}")

    def register_special_tensor(self, tensor: torch.Tensor, tensor_name: str = "unknown"):
        """注册特殊张量（自动为非叶子张量启用retain_grad）"""
        if not tensor.is_leaf:
            tensor.retain_grad()
            print(f"ℹ️ 张量[{tensor_name}]为非叶子张量，已自动启用retain_grad")
        self.special_tensors.add(tensor)
        print(f"✅ 已注册特殊张量：{tensor_name}，形状={tensor.shape}，是否叶子张量={tensor.is_leaf}")

    def clip_by_norm(self, parameters, max_norm: float = 1.0, norm_type: float = 2.0):
        """按范数裁剪梯度"""
        before_norm = torch.nn.utils.clip_grad_norm_(parameters, max_norm, norm_type, error_if_nonfinite=False)
        print(f"📏 梯度范数裁剪：裁剪前总范数={before_norm:.4f}，最大允许范数={max_norm}")

    def clip_by_value(self, parameters, min_val: float = -1.0, max_val: float = 1.0):
        """按值裁剪梯度"""
        grad_stats = []
        for p in parameters:
            if p.grad is not None:
                before_min = p.grad.data.min().item()
                before_max = p.grad.data.max().item()
                p.grad.data.clamp_(min_val, max_val)
                after_min = p.grad.data.min().item()
                after_max = p.grad.data.max().item()
                grad_stats.append({
                    "param": p.__class__.__name__,
                    "before_min": before_min,
                    "before_max": before_max,
                    "after_min": after_min,
                    "after_max": after_max
                })
        print(f"📊 梯度值裁剪：范围=[{min_val}, {max_val}]")
        for stat in grad_stats:
            print(
                f"  - {stat['param']}：裁剪前[{stat['before_min']:.4f}, {stat['before_max']:.4f}] → 裁剪后[{stat['after_min']:.4f}, {stat['after_max']:.4f}]")

    def filter_small_grads(self, parameters, threshold: float = 1e-4):
        """过滤过小梯度（置0）"""
        zero_ratio_stats = []
        for p in parameters:
            if p.grad is not None:
                total = p.grad.data.numel()
                zero_before = (p.grad.data == 0).sum().item()
                mask = torch.abs(p.grad.data) < threshold
                p.grad.data.masked_fill_(mask, 0.0)
                zero_after = (p.grad.data == 0).sum().item()
                zero_ratio = (zero_after / total) * 100
                zero_ratio_stats.append({
                    "param": p.__class__.__name__,
                    "zero_ratio(%)": zero_ratio,
                    "zero_count": zero_after - zero_before
                })
        print(f"🔍 小梯度过滤：阈值={threshold}")
        for stat in zero_ratio_stats:
            print(f"  - {stat['param']}：新增零梯度数={stat['zero_count']}，总零梯度占比={stat['zero_ratio(%)']:.2f}%")

    def scale_grads(self, parameters, scale_factor: float = 1.0):
        """缩放梯度"""
        grad_mean_stats = []
        for p in parameters:
            if p.grad is not None:
                before_mean = p.grad.data.mean().item()
                p.grad.data.mul_(scale_factor)
                after_mean = p.grad.data.mean().item()
                grad_mean_stats.append({
                    "param": p.__class__.__name__,
                    "before_mean": before_mean,
                    "after_mean": after_mean
                })
        print(f"📈 梯度缩放：缩放因子={scale_factor}")
        for stat in grad_mean_stats:
            print(f"  - {stat['param']}：梯度均值 {stat['before_mean']:.6f} → {stat['after_mean']:.6f}")

    def replace_undef_func_grads(self):
        """替换不可导函数的输入张量梯度"""
        print(f"🔄 开始替换不可导函数梯度，共{len(self.special_tensors)}个特殊张量待处理")
        for idx, tensor in enumerate(self.special_tensors, 1):
            if tensor.grad is None:
                print(f"  ❌ 张量{idx}：梯度为None，跳过替换")
                continue

            # 匹配不可导函数
            matched = False
            for func_name, grad_func in self.undef_func_grad_rules.items():
                grad_fn_name = tensor._grad_fn.__class__.__name__.lower()
                if func_name in grad_fn_name:
                    before_grad_mean = tensor.grad.data.mean().item()
                    tensor.grad.data = grad_func(tensor.grad.data)
                    after_grad_mean = tensor.grad.data.mean().item()
                    print(f"  ✅ 张量{idx}：匹配函数[{func_name}]（梯度函数：{tensor._grad_fn.__class__.__name__}）")
                    print(f"     梯度均值 {before_grad_mean:.6f} → {after_grad_mean:.6f}")
                    matched = True
                    break
            if not matched:
                print(f"  ⚠️  张量{idx}：未匹配到任何注册的不可导函数规则")

    def apply_sparsity(self, parameters, sparsity_ratio: float = 0.1):
        """梯度稀疏化（随机置0）"""
        sparsity_stats = []
        for p in parameters:
            if p.grad is not None:
                total = p.grad.data.numel()
                mask = torch.rand_like(p.grad.data) < sparsity_ratio
                zero_count = mask.sum().item()
                p.grad.data.masked_fill_(mask, 0.0)
                sparsity = (zero_count / total) * 100
                sparsity_stats.append({
                    "param": p.__class__.__name__,
                    "sparsity(%)": sparsity,
                    "zero_count": zero_count
                })
        print(f"🌫️  梯度稀疏化：目标稀疏比例={sparsity_ratio * 100}%")
        for stat in sparsity_stats:
            print(f"  - {stat['param']}：实际稀疏占比={stat['sparsity(%)']:.2f}%，置零数量={stat['zero_count']}")

    def smooth_grads(self, parameters, alpha: float = 0.9):
        """梯度平滑（指数移动平均）"""
        if not hasattr(self, 'prev_grads'):
            self.prev_grads = {}
            print(f"📝 初始化梯度历史缓存，平滑系数alpha={alpha}")

        smooth_stats = []
        for p in parameters:
            if p.grad is None:
                continue

            param_id = id(p)
            before_mean = p.grad.data.mean().item()
            # 初始化历史梯度（首次处理）
            if param_id not in self.prev_grads:
                self.prev_grads[param_id] = torch.zeros_like(p.grad.data)
                print(f"  🆕 首次处理{p.__class__.__name__}，初始化历史梯度")

            # 计算平滑后梯度
            smoothed_grad = alpha * self.prev_grads[param_id] + (1 - alpha) * p.grad.data
            after_mean = smoothed_grad.mean().item()
            # 更新当前梯度和历史缓存
            p.grad.data = smoothed_grad
            self.prev_grads[param_id] = smoothed_grad.clone()

            smooth_stats.append({
                "param": p.__class__.__name__,
                "before_mean": before_mean,
                "after_mean": after_mean
            })

        print(f"🧽 梯度平滑：alpha={alpha}")
        for stat in smooth_stats:
            print(f"  - {stat['param']}：梯度均值 {stat['before_mean']:.6f} → {stat['after_mean']:.6f}")

    def apply(self, parameters, strategies: Optional[Dict[str, Union[Dict, bool]]] = None):
        """一键应用多种梯度修改策略"""
        # 默认策略配置
        default_strategies = {
            "clip_by_norm": {"max_norm": 1.0, "norm_type": 2.0},
            "filter_small_grads": {"threshold": 1e-4},
            "replace_undef_func_grads": True
        }
        # 合并策略（用户策略覆盖默认）
        final_strategies = {**default_strategies, **(strategies or {})}

        print("=" * 60)
        print("🚀 开始执行梯度修改策略，共启用{}种策略".format(sum(1 for v in final_strategies.values() if v)))
        print("=" * 60)

        for strategy, config in final_strategies.items():
            if not config:
                print(f"\n❌ 跳过策略：{strategy}（已禁用）")
                continue

            # 处理配置格式
            if isinstance(config, bool):
                config = {}
            print(f"\n📌 正在执行策略：{strategy}，配置={config}")

            # 调用对应策略方法
            if hasattr(self, strategy):
                try:
                    if strategy == "replace_undef_func_grads":
                        getattr(self, strategy)()  # 无parameters参数
                    else:
                        getattr(self, strategy)(parameters, **config)
                    print(f"✅ 策略{strategy}执行完成")
                except Exception as e:
                    print(f"❌ 策略{strategy}执行失败：{str(e)}")
            else:
                print(f"❌ 未知策略：{strategy}，跳过执行")

        print("\n" + "=" * 60)
        print("🎉 所有梯度修改策略执行完毕")
        print("=" * 60)


# ------------------------------
# 详细测试案例（打印完整过程结果）
# ------------------------------
if __name__ == "__main__":
    # 1. 初始化基础组件
    torch.manual_seed(42)  # 固定随机种子，确保结果可复现
    print("=" * 80)
    print("📋 初始化测试环境：")
    print(f"  - PyTorch版本：{torch.__version__}")
    print(f"  - 随机种子：42")
    print("=" * 80)

    # 2. 创建模型、梯度修改器、优化器
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 20),  # 第一层：输入10维→输出20维
        torch.nn.ReLU(),
        torch.nn.Linear(20, 2)  # 第二层：输入20维→输出2维（分类任务）
    )
    grad_modifier = GradientModifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n📦 模型结构：")
    for idx, layer in enumerate(model, 1):
        print(
            f"  {idx}. {layer.__class__.__name__} → 输入维度：{layer.in_features if hasattr(layer, 'in_features') else 'N/A'}，输出维度：{layer.out_features if hasattr(layer, 'out_features') else 'N/A'}")
    print(f"🔧 优化器：Adam，学习率=1e-3")

    # 3. 注册不可导函数规则（以torch.round为例）
    print(f"\n📝 注册不可导函数规则：")
    grad_modifier.register_undef_func_rule(
        func_name="round",  # 匹配RoundBackward梯度函数
        grad_func=lambda grad: torch.ones_like(grad) * 0.5  # 自定义梯度为0.5
    )

    # 4. 生成测试数据
    batch_size = 8
    input_data = torch.randn(batch_size, 10)  # 输入：(8,10)
    target = torch.randint(0, 2, (batch_size,))  # 标签：0/1分类
    print(f"\n📊 测试数据：")
    print(f"  - 输入形状：{input_data.shape}，均值={input_data.mean().item():.4f}，标准差={input_data.std().item():.4f}")
    print(f"  - 标签形状：{target.shape}，标签分布：{torch.bincount(target).tolist()}（0的数量，1的数量）")

    # 5. 前向传播（包含不可导操作torch.round）
    print(f"\n🔄 前向传播：")
    hidden = model[0](input_data)  # 第一层输出：(8,20)
    relu_out = model[1](hidden)  # ReLU输出：(8,20)
    logits = model[2](relu_out)  # 最终输出（未激活）：(8,2)
    rounded_logits = torch.round(logits)  # 不可导操作：对输出取整
    loss = torch.nn.functional.cross_entropy(rounded_logits, target)  # 计算损失

    print(f"  - 第一层输出（Linear）：形状={hidden.shape}，均值={hidden.mean().item():.4f}")
    print(f"  - ReLU输出：形状={relu_out.shape}，非零元素占比={((relu_out > 0).sum() / relu_out.numel() * 100):.2f}%")
    print(
        f"  - 最终logits：形状={logits.shape}，均值={logits.mean().item():.4f}，范围=[{logits.min().item():.4f}, {logits.max().item():.4f}]")
    print(f"  - 取整后logits：形状={rounded_logits.shape}，唯一值={torch.unique(rounded_logits).tolist()}")
    print(f"  - 损失值：{loss.item():.6f}")

    # 6. 注册不可导操作的输入张量（rounded_logits的输入是logits）
    print(f"\n📌 注册特殊张量：")
    grad_modifier.register_special_tensor(logits, tensor_name="logits（不可导操作输入）")

    # 7. 反向传播（计算梯度）
    print(f"\n🔙 反向传播：")
    optimizer.zero_grad()  # 清空历史梯度
    print(f"  - 清空历史梯度完成")
    loss.backward()  # 计算梯度
    print(f"  - 梯度计算完成，开始打印关键梯度信息：")

    # 打印反向传播后的原始梯度（未修改前）
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"    - {name}：梯度均值={param.grad.mean().item():.6f}，范数={param.grad.norm().item():.4f}")
        else:
            print(f"    - {name}：梯度为None")
    # 打印不可导操作输入张量的原始梯度
    if logits.grad is not None:
        print(f"    - logits（不可导输入）：梯度均值={logits.grad.mean().item():.6f}，范数={logits.grad.norm().item():.4f}")
    else:
        print(f"    - logits（不可导输入）：梯度为None")

    # 8. 应用梯度修改策略（多策略组合）
    print(f"\n🎯 应用梯度修改策略：")
    grad_modifier.apply(
        parameters=model.parameters(),
        strategies={
            "clip_by_norm": {"max_norm": 1.5, "norm_type": 2.0},  # 范数裁剪
            "clip_by_value": {"min_val": -0.8, "max_val": 0.8},  # 值裁剪
            "filter_small_grads": {"threshold": 1e-3},  # 小梯度过滤
            "scale_grads": {"scale_factor": 1.2},  # 梯度缩放
            "apply_sparsity": {"sparsity_ratio": 0.15},  # 梯度稀疏化
            "smooth_grads": {"alpha": 0.85},  # 梯度平滑
            "replace_undef_func_grads": True  # 不可导梯度替换
        }
    )

    # 9. 打印修改后的最终梯度信息
    print(f"\n📊 修改后的最终梯度信息：")
    print("-" * 50)
    total_params = 0
    total_zero_grads = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_count = param.numel()
            zero_count = (param.grad.data == 0).sum().item()
            zero_ratio = (zero_count / param_count) * 100
            total_params += param_count
            total_zero_grads += zero_count

            print(f"参数：{name}")
            print(f"  - 形状：{param.shape}，总元素数：{param_count}")
            print(f"  - 梯度均值：{param.grad.mean().item():.6f}，梯度范数：{param.grad.norm().item():.4f}")
            print(f"  - 零梯度数量：{zero_count}，零梯度占比：{zero_ratio:.2f}%")
            print(f"  - 梯度范围：[{param.grad.min().item():.6f}, {param.grad.max().item():.6f}]")
            print()
        else:
            print(f"参数：{name} → 梯度为None\n")

    overall_zero_ratio = (total_zero_grads / total_params) * 100
    print(f"📈 全局梯度统计：")
    print(f"  - 总参数数量：{total_params}")
    print(f"  - 总零梯度数量：{total_zero_grads}")
    print(f"  - 全局零梯度占比：{overall_zero_ratio:.2f}%")

    # 10. 参数更新与训练后模型状态
    print(f"\n⚡ 执行参数更新：")
    optimizer.step()
    print(f"  - 参数更新完成")

    # 打印更新后的参数变化（对比更新前后）
    print(f"\n🔍 参数更新前后变化（以第一层权重为例）：")
    first_layer = model[0]  # 取第一层Linear
    # 重新计算前向传播，查看更新后的输出变化
    updated_hidden = first_layer(input_data)
    hidden_diff = torch.abs(updated_hidden - hidden).mean().item()  # 输出差异均值

    print(f"  - 第一层权重更新前均值：{hidden.mean().item():.6f}")
    print(f"  - 第一层权重更新后均值：{updated_hidden.mean().item():.6f}")
    print(f"  - 第一层输出差异均值：{hidden_diff:.6f}")

    # 计算更新后的损失（验证梯度修改效果）
    updated_relu = model[1](updated_hidden)
    updated_logits = model[2](updated_relu)
    updated_rounded = torch.round(updated_logits)
    updated_loss = torch.nn.functional.cross_entropy(updated_rounded, target)

    print(f"\n📉 损失变化：")
    print(f"  - 更新前损失：{loss.item():.6f}")
    print(f"  - 更新后损失：{updated_loss.item():.6f}")
    print(f"  - 损失变化量：{updated_loss.item() - loss.item():.6f}")

    # 11. 不可导梯度替换验证
    print(f"\n✅ 不可导梯度替换验证：")
    if logits.grad is not None:
        # 检查logits梯度是否被替换为0.5（自定义规则）
        grad_unique = torch.unique(logits.grad)
        is_replaced = torch.allclose(logits.grad, torch.ones_like(logits.grad) * 0.5, atol=1e-6)
        print(f"  - logits梯度唯一值：{[round(v.item(), 6) for v in grad_unique]}")
        print(f"  - 是否符合自定义梯度规则（全部为0.5）：{'是' if is_replaced else '否'}")
        print(f"  - logits梯度形状：{logits.grad.shape}，梯度总和：{logits.grad.sum().item():.4f}")
    else:
        print(f"  - logits梯度为None，替换失败")

    print(f"\n" + "=" * 80)
    print("📋 测试案例执行完毕，所有过程结果已打印")
    print("=" * 80)
