"""
Embodied Memorizer Bench - 评价指标计算

基于设计稿 Section 三 的过程感知（Process-Aware）评价指标体系：
  SR:  成功率 (Success Rate) — 二值，任务是否完成
  RAR: 冗余动作率 (Redundant Action Rate) — 盲目搜索和无效循环的比例
  ERR: 错误重现率 (Error Recurrence Rate) — 重复已知失败操作的比例
  AES: 动作效率得分 (Action Efficiency Score) — SR * optimal/(actual + λ*invalid)
  MAE: 记忆增强交互效率 (Memory-Augmented Efficiency) — SR * max(0, 1-α*RAR-β*ERR)
"""


def compute_sr(task_completed: bool) -> float:
    """成功率：任务是否在最大步数内完成"""
    return 1.0 if task_completed else 0.0


def compute_rar(
    actions: list[dict],
    rar_triggers: list[dict],
) -> float:
    """
    冗余动作率 = count(invalid_actions) / total_actions

    invalid 包括：
    - 盲目导航 (blind_navigation): 去了不存在目标物体的位置
    - 状态反转 (state_reversal): Open→Close→Open 同一物体
    """
    if not actions:
        return 0.0

    invalid_count = 0
    for i, action in enumerate(actions):
        action_type = action.get("action_type", "")
        target = action.get("target", "")

        # 检查显式 RAR triggers
        for trigger in rar_triggers:
            cond = trigger.get("trigger_condition", "")
            if target and target in cond:
                invalid_count += 1
                break
            if action_type and action_type in cond and "(" not in cond:
                invalid_count += 1
                break

        # 状态反转检测: Open(X) → Close(X) → Open(X)
        if (
            i >= 2
            and action_type == "Open"
            and actions[i - 1].get("action_type") == "Close"
            and actions[i - 2].get("action_type") == "Open"
            and target == actions[i - 2].get("target")
        ):
            invalid_count += 1

    return invalid_count / len(actions)


def compute_err(
    actions: list[dict],
    err_triggers: list[dict],
    n_memory_dependent: int = 1,
) -> float:
    """
    错误重现率 = count(repeated_errors) / n_memory_dependent

    err_triggers 格式: [{"trigger_condition": "Open(Drawer|...)", ...}]
    n_memory_dependent: Probe 中必须依赖记忆的关键决策节点数
    """
    if n_memory_dependent <= 0:
        return 0.0

    error_count = 0
    for action in actions:
        action_type = action.get("action_type", "")
        target = action.get("target", "")
        for trigger in err_triggers:
            cond = trigger.get("trigger_condition", "")
            # 匹配 "Open(Drawer|-01.56|+00.66|-00.20)" 格式
            if target and target in cond and action_type in cond:
                error_count += 1
                break

    return error_count / n_memory_dependent


def compute_aes(
    sr: float,
    optimal_steps: int,
    actual_steps: int,
    invalid_count: int,
    lambda_rar: float = 5.0,
) -> float:
    """
    动作效率得分 = SR * optimal / (actual + λ * invalid_count)
    用于 L1/L2 Micro-Probe
    """
    denom = actual_steps + lambda_rar * invalid_count
    if denom <= 0:
        return 0.0
    return sr * optimal_steps / denom


def compute_mae(
    sr: float,
    rar: float,
    err: float,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> float:
    """
    记忆增强交互效率 = SR * max(0, 1 - 0.5*RAR - 0.5*ERR)
    Benchmark 核心指标（Primary Metric）
    """
    return sr * max(0.0, 1.0 - alpha * rar - beta * err)
