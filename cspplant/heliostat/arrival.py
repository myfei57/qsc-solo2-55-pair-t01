"""一轮追日的到位计划（纯判定，不动驱动）。

窗口起点把每台参与自动追日的镜子算一遍并**冻结**成一份
:class:`RoundPlan`：本轮目标角、当前回读角、按两轴限速算出的所需转动时间、
判定结论、处置动作与指令次序都在这份计划里。随后的指令下发、到位验证、
审计落盘与控制台展示只引用这一份计划，不允许各算各的——这是"指令、判定与
审计一致"的唯一保证。

定死的两条口径：

1. **判定逐台，处置按镜区**。每台镜子独立得出结论，互不连坐；来不及时是
   "低一档精度继续追日"（``coarse``）还是"直接退出聚焦"（``exit``），只
   由该台所属镜区的策略表决定，策略表未登记的镜区一律按最保守的 ``exit``。
2. **驱动回读缺失属于部分失败，按没跟上处理**。拿不到当前角就无法证明来得
   及，结论直接记 ``lag``（原因 ``readback_missing``），再走镜区处置；它不
   是联锁拒绝，不阻断本轮其余镜子。

时间预算：本轮窗口长 ``period_s``；窗口之外最多允许提前 ``max_lead_s``
起转（跨轮提前量，防止上一轮就把镜子全速转出去）。所需时间不超过窗口长度
为 ``in_time``；需要提前、但提前量不超过上限为 ``early_start``；连提前量
吃满也来不及为 ``lag``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .aim import AimAngles, angle_error

# 判定结论。
IN_TIME = "in_time"
EARLY_START = "early_start"
LAG = "lag"
SKIP = "skip"

# 落后处置（按镜区）。
POLICY_COARSE = "coarse"
POLICY_EXIT = "exit"
POLICIES = (POLICY_COARSE, POLICY_EXIT)
DEFAULT_POLICY = POLICY_EXIT

# 指令档位与动作。
GRADE_FINE = "fine"
GRADE_COARSE = "coarse"
ACTION_AIM = "aim"
ACTION_DEFOCUS = "defocus"
ACTION_SKIP = "skip"

# 一轮内的相位：起转指令必须全部早于退出指令，顺序不能乱。
PHASE_START = "start"
PHASE_EXIT = "exit"

# start 相位内部的批次：提前起转最先，低一档继续追日其次，正常起转最后。
_BATCH = {EARLY_START: 0, LAG: 1, IN_TIME: 2}


@dataclass(frozen=True)
class ArrivalInput:
    """一台镜子的计划输入：静态归属、当前回读（缺失为 None）与本轮目标。"""

    mirror: str
    group: str
    current: AimAngles | None
    target: AimAngles


@dataclass(frozen=True)
class ArrivalCommand:
    """一条按顺序排好的驱动指令。"""

    seq: int
    phase: str
    mirror: str
    group: str
    action: str
    grade: str
    lead_s: float
    verdict: str
    reason: str

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "phase": self.phase,
            "mirror": self.mirror,
            "group": self.group,
            "action": self.action,
            "grade": self.grade,
            "lead_s": round(self.lead_s, 3),
            "verdict": self.verdict,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArrivalItem:
    """一台镜子本轮的到位判定与处置。"""

    mirror: str
    group: str
    verdict: str
    action: str
    phase: str
    grade: str
    policy: str
    target: AimAngles
    azimuth_error_deg: float | None
    elevation_error_deg: float | None
    need_s: float | None
    lead_s: float
    reason: str

    def as_dict(self) -> dict:
        return {
            "mirror": self.mirror,
            "group": self.group,
            "verdict": self.verdict,
            "action": self.action,
            "phase": self.phase,
            "grade": self.grade,
            "policy": self.policy,
            "target": self.target.as_dict(),
            "azimuth_error_deg": None
            if self.azimuth_error_deg is None
            else round(self.azimuth_error_deg, 4),
            "elevation_error_deg": None
            if self.elevation_error_deg is None
            else round(self.elevation_error_deg, 4),
            "need_s": None if self.need_s is None else round(self.need_s, 3),
            "lead_s": round(self.lead_s, 3),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RoundPlan:
    """一轮冻结的到位计划：逐台条目 + 由条目导出的有序指令。"""

    plan_id: str
    round: int
    ts: float
    period_s: float
    max_lead_s: float
    coarse_tolerance_deg: float
    policies: dict = field(default_factory=dict)
    entries: tuple[ArrivalItem, ...] = ()

    def _ordered_entries(self) -> list[ArrivalItem]:
        actionable = [item for item in self.entries if item.action != ACTION_SKIP]
        starts = [item for item in actionable if item.phase == PHASE_START]
        exits = [item for item in actionable if item.phase == PHASE_EXIT]
        starts.sort(key=lambda item: (_BATCH[item.verdict], -item.lead_s, item.group, item.mirror))
        exits.sort(key=lambda item: (item.group, item.mirror))
        return starts + exits

    def commands(self) -> list[ArrivalCommand]:
        """按本轮时序导出指令：起转（提前 → 低档 → 正常）→ 退出。"""

        ordered = self._ordered_entries()
        commands: list[ArrivalCommand] = []
        for seq, item in enumerate(ordered):
            commands.append(
                ArrivalCommand(
                    seq=seq,
                    phase=item.phase,
                    mirror=item.mirror,
                    group=item.group,
                    action=item.action,
                    grade=item.grade,
                    lead_s=item.lead_s,
                    verdict=item.verdict,
                    reason=item.reason,
                )
            )
        return commands

    def named(self, mirror: str) -> ArrivalItem:
        """按镜名取计划条目。"""

        for item in self.entries:
            if item.mirror == mirror:
                return item
        raise KeyError(mirror)

    def count(self, verdict: str) -> int:
        return sum(1 for item in self.entries if item.verdict == verdict)

    def lead_budget_s(self) -> float:
        """本批提前起转镜实际占用的最大提前量。"""

        return max((item.lead_s for item in self.entries), default=0.0)

    def as_dict(self) -> dict:
        return {
            "plan_id": self.plan_id,
            "round": self.round,
            "ts": round(self.ts, 3),
            "period_s": round(self.period_s, 3),
            "max_lead_s": round(self.max_lead_s, 3),
            "coarse_tolerance_deg": round(self.coarse_tolerance_deg, 3),
            "policies": dict(self.policies),
            "commands": [command.as_dict() for command in self.commands()],
            "entries": [item.as_dict() for item in self.entries],
            "counts": {
                "planned": len(self.entries),
                "in_time": self.count(IN_TIME),
                "early_start": self.count(EARLY_START),
                "lag": self.count(LAG),
                "skip": self.count(SKIP),
            },
        }


def required_time_s(current: AimAngles, target: AimAngles, azimuth_rate_deg_s: float,
                    elevation_rate_deg_s: float) -> tuple[float, float, float]:
    """按两轴限速算转到目标所需时间，返回 (秒, 方位角差, 仰角差)。

    两个轴同时转，整镜到位时间取较慢的那个轴；角度差取方位最短弧。
    """

    error = angle_error(current, target)
    need = max(
        abs(error.azimuth_deg) / azimuth_rate_deg_s,
        abs(error.elevation_deg) / elevation_rate_deg_s,
    )
    return need, error.azimuth_deg, error.elevation_deg


def build_round_plan(
    plan_id: str,
    round_no: int,
    ts: float,
    inputs: list[ArrivalInput],
    *,
    period_s: float,
    max_lead_s: float,
    azimuth_rate_deg_s: float,
    elevation_rate_deg_s: float,
    coarse_tolerance_deg: float,
    policies: dict[str, str] | None = None,
    skipped: tuple[tuple[str, str, str], ...] = (),
) -> RoundPlan:
    """逐台判定并冻结一轮到位计划。

    ``skipped`` 携带本轮不参与自动追日的 ``(镜名, 镜区, 原因)``（手动接管／
    避让），它们只记账，不产生指令。
    """

    if period_s <= 0.0:
        raise ValueError("本轮窗口长度必须为正")
    if max_lead_s < 0.0:
        raise ValueError("提前起转上限不得为负")
    policy_table = dict(policies or {})

    entries: list[ArrivalItem] = []
    for data in inputs:
        policy = policy_table.get(data.group, DEFAULT_POLICY)
        if policy not in POLICIES:
            policy = DEFAULT_POLICY
        if data.current is None:
            # 回读缺失是部分失败：无法证明来得及，直接按没跟上处置。
            entries.append(
                ArrivalItem(
                    mirror=data.mirror,
                    group=data.group,
                    verdict=LAG,
                    action=ACTION_DEFOCUS if policy == POLICY_EXIT else ACTION_AIM,
                    phase=PHASE_EXIT if policy == POLICY_EXIT else PHASE_START,
                    grade=GRADE_FINE if policy == POLICY_EXIT else GRADE_COARSE,
                    policy=policy,
                    target=data.target,
                    azimuth_error_deg=None,
                    elevation_error_deg=None,
                    need_s=None,
                    lead_s=0.0,
                    reason="readback_missing",
                )
            )
            continue

        need, az_delta, el_delta = required_time_s(
            data.current, data.target, azimuth_rate_deg_s, elevation_rate_deg_s
        )
        if need <= period_s:
            entries.append(
                ArrivalItem(
                    mirror=data.mirror,
                    group=data.group,
                    verdict=IN_TIME,
                    action=ACTION_AIM,
                    phase=PHASE_START,
                    grade=GRADE_FINE,
                    policy=policy,
                    target=data.target,
                    azimuth_error_deg=az_delta,
                    elevation_error_deg=el_delta,
                    need_s=need,
                    lead_s=0.0,
                    reason="within_round",
                )
            )
        elif need <= period_s + max_lead_s:
            entries.append(
                ArrivalItem(
                    mirror=data.mirror,
                    group=data.group,
                    verdict=EARLY_START,
                    action=ACTION_AIM,
                    phase=PHASE_START,
                    grade=GRADE_FINE,
                    policy=policy,
                    target=data.target,
                    azimuth_error_deg=az_delta,
                    elevation_error_deg=el_delta,
                    need_s=need,
                    lead_s=need - period_s,
                    reason="lead_start",
                )
            )
        else:
            entries.append(
                ArrivalItem(
                    mirror=data.mirror,
                    group=data.group,
                    verdict=LAG,
                    action=ACTION_DEFOCUS if policy == POLICY_EXIT else ACTION_AIM,
                    phase=PHASE_EXIT if policy == POLICY_EXIT else PHASE_START,
                    grade=GRADE_FINE if policy == POLICY_EXIT else GRADE_COARSE,
                    policy=policy,
                    target=data.target,
                    azimuth_error_deg=az_delta,
                    elevation_error_deg=el_delta,
                    need_s=need,
                    lead_s=0.0,
                    reason="cannot_finish",
                )
            )

    for mirror, group, why in skipped:
        entries.append(
            ArrivalItem(
                mirror=mirror,
                group=group,
                verdict=SKIP,
                action=ACTION_SKIP,
                phase=PHASE_START,
                grade=GRADE_FINE,
                policy=policy_table.get(group, DEFAULT_POLICY),
                target=AimAngles(0.0, 0.0),
                azimuth_error_deg=None,
                elevation_error_deg=None,
                need_s=None,
                lead_s=0.0,
                reason=why,
            )
        )

    return RoundPlan(
        plan_id=str(plan_id),
        round=int(round_no),
        ts=float(ts),
        period_s=float(period_s),
        max_lead_s=float(max_lead_s),
        coarse_tolerance_deg=float(coarse_tolerance_deg),
        policies=policy_table,
        entries=tuple(entries),
    )


def lead_ticks(plan: RoundPlan) -> int:
    """提前起转组在窗口开始前先转多少个整秒（向上取整，只多不少）。"""

    return max(1, int(math.ceil(plan.lead_budget_s()))) if plan.lead_budget_s() > 0.0 else 0


def period_ticks(plan: RoundPlan) -> int:
    """本轮窗口按整秒步进的步数。"""

    return max(1, int(math.ceil(plan.period_s)))
