"""一轮追日的到位计划、起转时序与落后处置。

太阳走得快时，单轮窗口可能不够某些镜子从当前角度转到目标角。这里在**下发
任何指令之前**逐台回答四个问题：

1. 这一轮来不来得及转到位（按两轴各自所需转动时间的较大值估算）；
2. 来得及但要抢时间的，最晚什么时候必须起转（提前起转）；
3. 来不及的，按所属镜区的落后策略处置——降一档精度继续追日，或直接退出聚焦；
4. 落后处置是**按镜区批量**还是逐台：判定逐台做，处置命令按镜区打成同一
   批次（同一个 ``batch_id``、排序与序号连续），不允许一台一个口径。

计划器只产出不可变的数据结构（计划、判定、指令清单），不直接碰驱动；真正
下发由 :mod:`cspplant.field.arrival` 的运行器按封存清单顺序执行。计划一旦
封存就不再改写，控制台展示、指令下发与审计落的是同一份计划。

驱动回读缺失或过期属于**部分失败**：计划阶段拿不到回读按"来不及"处理，
判定阶段拿不到回读按"没跟上"处理，绝不因为"没有回读"就默认到位。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import InterlockError
from .aim import AimAngles, angle_error

# 逐台判定结论。
FEASIBLE = "feasible"          # 正常起转即可在截止前到位
EARLY = "early"                # 来得及，但必须早于本轮统一起转时刻起转
INFEASIBLE = "infeasible"      # 即使立即起转也到不了位，按镜区策略提前处置
SKIPPED = "skipped"            # 避让或手动接管，不参与本轮追日

# 判定时刻的最终状态。
ARRIVED = "arrived"
LAGGED = "lagged"
EXITED = "exited"
DEGRADED = "degraded"
NOT_TRACKED = "not_tracked"

# 落后镜区策略。
LAG_DEGRADE = "degrade"        # 降一档精度继续追日
LAG_EXIT = "exit"              # 直接退出聚焦

# 指令种类。
SLEW = "slew"
TRACK_DEGRADED = "track_degraded"
EXIT_FOCUS = "exit_focus"

PLAN_BATCH = "plan"
LAG_BATCH_PREFIX = "lag"

# 封存阶段顺序：提前起转 → 计划内退出 → 计划内降档 → 正常起转；
# 截止后的落后处置追加在后：落后退出 → 落后降档。
PHASE_EARLY_SLEW = 0
PHASE_PLAN_EXIT = 1
PHASE_PLAN_DEGRADE = 2
PHASE_NORMAL_SLEW = 3
PHASE_LAG_EXIT = 4
PHASE_LAG_DEGRADE = 5


@dataclass(frozen=True)
class ArrivalPolicy:
    """到位计划的固定口径。

    ``max_tier`` 是允许的最低精度档：1 为全精度（对焦公差），档数每加一，
    公差按 ``tolerance_multiplier`` 放宽一档。已经处在最低档还落后的镜子，
    即使镜区策略是降档也必须退出聚焦。
    """

    max_tier: int = 3
    tolerance_multiplier: float = 2.0
    slew_margin_s: float = 0.0
    readback_stale_after_s: float = 2.0
    early_lead_s: float = 60.0

    def validate(self) -> None:
        if self.max_tier < 1:
            raise InterlockError("最低精度档不得小于 1", component="arrival")
        if self.tolerance_multiplier < 1.0:
            raise InterlockError("降档公差倍数不得小于 1", component="arrival")
        if self.slew_margin_s < 0.0:
            raise InterlockError("起转余量不得为负", component="arrival")
        if self.readback_stale_after_s <= 0.0:
            raise InterlockError("回读有效期必须为正", component="arrival")
        if self.early_lead_s <= 0.0:
            raise InterlockError("提前起转窗口必须为正", component="arrival")

    def tolerance_deg(self, base_tolerance_deg: float, tier: int) -> float:
        """返回某精度档对应的对焦公差。"""

        return float(base_tolerance_deg) * (self.tolerance_multiplier ** (int(tier) - 1))

    def as_dict(self) -> dict:
        return {
            "max_tier": self.max_tier,
            "tolerance_multiplier": self.tolerance_multiplier,
            "slew_margin_s": round(self.slew_margin_s, 3),
            "readback_stale_after_s": round(self.readback_stale_after_s, 3),
            "early_lead_s": round(self.early_lead_s, 3),
        }


@dataclass(frozen=True)
class RoundWindow:
    """一轮追日的时序窗口。

    ``release_at`` 为计划封存时刻（允许提前起转的最早时刻），``launch_at``
    为正常统一起转时刻，``deadline_at`` 为对焦截止时刻。三者必须单调不回退。
    """

    round_id: str
    release_at: float
    launch_at: float
    deadline_at: float

    def __post_init__(self) -> None:
        if not self.round_id:
            raise InterlockError("轮次必须有标识", component="arrival")
        if not self.release_at <= self.launch_at <= self.deadline_at:
            raise InterlockError("轮次时序必须满足 封存 ≤ 起转 ≤ 截止", component="arrival")

    def as_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "release_at": round(self.release_at, 3),
            "launch_at": round(self.launch_at, 3),
            "deadline_at": round(self.deadline_at, 3),
        }


@dataclass(frozen=True)
class Readback:
    """一次驱动角度回读；缺失或过期都是部分失败。"""

    ts: float
    angles: AimAngles | None

    def valid_at(self, now: float, stale_after_s: float) -> bool:
        """回读是否存在且未过期（时间戳不得来自未来）。"""

        return (
            self.angles is not None
            and self.ts <= now + 1e-9
            and now - self.ts <= float(stale_after_s) + 1e-9
        )


@dataclass(frozen=True)
class ZoneLagPolicy:
    """一个镜区的落后处置口径。"""

    zone: str
    action: str

    def __post_init__(self) -> None:
        if self.action not in (LAG_DEGRADE, LAG_EXIT):
            raise InterlockError(f"未知落后处置策略: {self.action}", component="arrival")

    def as_dict(self) -> dict:
        return {"zone": self.zone, "action": self.action}


@dataclass(frozen=True)
class MirrorInput:
    """计划一台镜子所需的输入。"""

    name: str
    zone: str
    target: AimAngles
    readback: Readback
    tier: int = 1
    participating: bool = True


@dataclass(frozen=True)
class MirrorArrival:
    """一台镜子在本轮的判定结论。"""

    mirror: str
    zone: str
    verdict: str
    reason: str
    start_kind: str
    tier: int
    tolerance_deg: float
    target: AimAngles
    actual: AimAngles | None
    readback_ts: float | None
    required_slew_s: float
    latest_safe_start: float | None
    planned_start: float | None
    lag_action: str

    @property
    def tracks(self) -> bool:
        """本轮是否仍以追日方式参与（正常或降档）。"""

        return self.verdict in (FEASIBLE, EARLY) or (
            self.verdict == INFEASIBLE and self.lag_action == LAG_DEGRADE
        )

    def as_dict(self) -> dict:
        return {
            "mirror": self.mirror,
            "zone": self.zone,
            "verdict": self.verdict,
            "reason": self.reason,
            "start_kind": self.start_kind,
            "tier": self.tier,
            "tolerance_deg": round(self.tolerance_deg, 3),
            "target": self.target.as_dict(),
            "actual": None if self.actual is None else self.actual.as_dict(),
            "readback_ts": None if self.readback_ts is None else round(self.readback_ts, 3),
            "required_slew_s": round(self.required_slew_s, 3),
            "latest_safe_start": (
                None if self.latest_safe_start is None else round(self.latest_safe_start, 3)
            ),
            "planned_start": None if self.planned_start is None else round(self.planned_start, 3),
            "lag_action": self.lag_action,
        }


@dataclass(frozen=True)
class PlannedCommand:
    """封存清单里的一条指令（起转、降档追日或退出聚焦）。"""

    seq: int
    command_id: str
    round_id: str
    mirror: str
    zone: str
    kind: str
    scheduled_at: float
    batch_id: str
    tier: int
    tolerance_deg: float

    def as_dict(self) -> dict:
        return {
            "seq": self.seq,
            "command_id": self.command_id,
            "round_id": self.round_id,
            "mirror": self.mirror,
            "zone": self.zone,
            "kind": self.kind,
            "scheduled_at": round(self.scheduled_at, 3),
            "batch_id": self.batch_id,
            "tier": self.tier,
            "tolerance_deg": round(self.tolerance_deg, 3),
        }


@dataclass(frozen=True)
class JudgeEntry:
    """一台镜子在截止时刻的到位判定。"""

    mirror: str
    zone: str
    status: str
    reason: str
    action: str
    from_tier: int
    to_tier: int
    tolerance_deg: float
    batch_id: str = ""

    @property
    def tracks(self) -> bool:
        return self.status in (ARRIVED, DEGRADED)

    def as_dict(self) -> dict:
        return {
            "mirror": self.mirror,
            "zone": self.zone,
            "status": self.status,
            "reason": self.reason,
            "action": self.action,
            "from_tier": self.from_tier,
            "to_tier": self.to_tier,
            "tolerance_deg": round(self.tolerance_deg, 3),
            "batch_id": self.batch_id,
        }


@dataclass(frozen=True)
class RoundPlan:
    """一轮封存后的到位计划：计划是指令、判定与审计的唯一口径。"""

    window: RoundWindow
    policy: ArrivalPolicy
    zone_policies: tuple[ZoneLagPolicy, ...]
    arrivals: tuple[MirrorArrival, ...]
    commands: tuple[PlannedCommand, ...]
    plan_command_count: int

    @property
    def round_id(self) -> str:
        return self.window.round_id

    def arrival(self, mirror: str) -> MirrorArrival:
        for item in self.arrivals:
            if item.mirror == mirror:
                return item
        raise InterlockError(f"{mirror} 不在本轮到位计划里", component="arrival")

    def get(self, command_id: str) -> PlannedCommand:
        for command in self.commands:
            if command.command_id == command_id:
                return command
        raise InterlockError(f"指令不在封存清单里: {command_id}", component="arrival")

    def summary(self) -> dict:
        counts: dict[str, int] = {}
        for item in self.arrivals:
            counts[item.verdict] = counts.get(item.verdict, 0) + 1
        return {
            "round_id": self.round_id,
            "window": self.window.as_dict(),
            "policy": self.policy.as_dict(),
            "zone_policies": [policy.as_dict() for policy in self.zone_policies],
            "verdicts": counts,
            "commands": len(self.commands),
            "plan_commands": self.plan_command_count,
        }

    def as_dict(self) -> dict:
        return {
            **self.summary(),
            "mirrors": [item.as_dict() for item in self.arrivals],
            "command_manifest": [command.as_dict() for command in self.commands],
        }


@dataclass(frozen=True)
class JudgeReport:
    """截止时刻的整轮判定结果，含落后处置批次与封存指令。"""

    round_id: str
    judged_at: float
    entries: tuple[JudgeEntry, ...]
    commands: tuple[PlannedCommand, ...]
    batches: tuple[str, ...]

    def status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.entries:
            counts[entry.status] = counts.get(entry.status, 0) + 1
        return counts

    def entry(self, mirror: str) -> JudgeEntry:
        for item in self.entries:
            if item.mirror == mirror:
                return item
        raise InterlockError(f"{mirror} 不在本轮判定结果里", component="arrival")

    def as_dict(self) -> dict:
        return {
            "round_id": self.round_id,
            "judged_at": round(self.judged_at, 3),
            "statuses": self.status_counts(),
            "batches": list(self.batches),
            "entries": [item.as_dict() for item in self.entries],
            "command_manifest": [command.as_dict() for command in self.commands],
        }


def slew_time_s(
    target: AimAngles,
    actual: AimAngles,
    azimuth_rate_deg_s: float,
    elevation_rate_deg_s: float,
) -> float:
    """按两轴限速估算从当前角度转到目标角度所需秒数（取较慢的轴）。"""

    error = angle_error(actual, target)
    az_time = abs(error.azimuth_deg) / max(azimuth_rate_deg_s, 1e-9)
    el_time = abs(error.elevation_deg) / max(elevation_rate_deg_s, 1e-9)
    return max(az_time, el_time)


def worst_case_slew_s(
    target: AimAngles,
    azimuth_rate_deg_s: float,
    elevation_rate_deg_s: float,
) -> float:
    """回读缺失时的保守转动时间：按从零位起转估，宁可误判来不及。"""

    az_time = abs(target.azimuth_deg) / max(azimuth_rate_deg_s, 1e-9)
    el_time = abs(target.elevation_deg) / max(elevation_rate_deg_s, 1e-9)
    return max(az_time, el_time)


class _PendingCommand:
    """封存前的暂存指令。"""

    __slots__ = ("mirror", "zone", "kind", "scheduled_at", "batch_id", "tier", "tolerance_deg")

    def __init__(
        self,
        mirror: str,
        zone: str,
        kind: str,
        scheduled_at: float,
        batch_id: str,
        tier: int,
        tolerance_deg: float,
    ) -> None:
        self.mirror = mirror
        self.zone = zone
        self.kind = kind
        self.scheduled_at = scheduled_at
        self.batch_id = batch_id
        self.tier = tier
        self.tolerance_deg = tolerance_deg


class ArrivalPlanner:
    """纯计算的到位计划器：产出封存计划与判定，不持有驱动状态。"""

    def __init__(
        self,
        zone_policies: dict[str, str],
        azimuth_rate_deg_s: float,
        elevation_rate_deg_s: float,
        base_tolerance_deg: float,
        policy: ArrivalPolicy | None = None,
    ) -> None:
        if float(azimuth_rate_deg_s) <= 0.0 or float(elevation_rate_deg_s) <= 0.0:
            raise InterlockError("两轴转速必须为正", component="arrival")
        self._rates = (float(azimuth_rate_deg_s), float(elevation_rate_deg_s))
        self._base_tolerance = float(base_tolerance_deg)
        self._policy = policy or ArrivalPolicy()
        self._policy.validate()
        # 镜区顺序按策略登记顺序固定，时序与批次都照这个顺序走。
        self._zone_policies = tuple(
            ZoneLagPolicy(zone=zone, action=action) for zone, action in zone_policies.items()
        )
        self._actions = {item.zone: item.action for item in self._zone_policies}
        self._zone_order = {item.zone: index for index, item in enumerate(self._zone_policies)}

    @property
    def policy(self) -> ArrivalPolicy:
        return self._policy

    def zone_policies(self) -> tuple[ZoneLagPolicy, ...]:
        return self._zone_policies

    def _lag_action(self, zone: str) -> str:
        try:
            return self._actions[zone]
        except KeyError as error:
            raise InterlockError(f"镜区 {zone} 没有登记落后处置策略", component="arrival") from error

    def _tolerance(self, tier: int) -> float:
        return self._policy.tolerance_deg(self._base_tolerance, tier)

    def plan_round(self, window: RoundWindow, inputs: list[MirrorInput]) -> RoundPlan:
        """封存一轮到位计划。"""

        arrivals: list[MirrorArrival] = []
        pending: list[_PendingCommand] = []

        for item in inputs:
            if not item.participating:
                arrivals.append(self._skipped(item))
                continue

            readback_ok = item.readback.valid_at(window.release_at, self._policy.readback_stale_after_s)
            if not readback_ok:
                # 部分失败：拿不到有效回读时按最保守口径处理——直接判来不及。
                required = worst_case_slew_s(item.target, *self._rates)
                verdict = INFEASIBLE
                reason = "readback_missing" if item.readback.angles is None else "readback_stale"
                latest_safe = None
            else:
                actual = item.readback.angles
                assert actual is not None
                required = slew_time_s(item.target, actual, *self._rates)
                latest_safe = window.deadline_at - required - self._policy.slew_margin_s
                if latest_safe + 1e-9 >= window.launch_at:
                    verdict, reason = FEASIBLE, "within_window"
                elif latest_safe + 1e-9 >= max(window.release_at, window.launch_at - self._policy.early_lead_s):
                    verdict, reason = EARLY, "early_start_required"
                else:
                    verdict, reason = INFEASIBLE, "insufficient_slew_time"

            lag_action = self._lag_action(item.zone)
            tier = item.tier

            if verdict == FEASIBLE:
                planned_start = window.launch_at
                pending.append(
                    _PendingCommand(item.name, item.zone, SLEW, planned_start, PLAN_BATCH,
                                    tier, self._tolerance(tier))
                )
                arrivals.append(
                    self._arrival(item, verdict, reason, "normal", tier, required,
                                  latest_safe, planned_start, lag_action, readback_ok)
                )
            elif verdict == EARLY:
                # 最晚必须起转时刻即计划起转时刻；不得早于封存时刻。
                planned_start = max(window.release_at, latest_safe)
                pending.append(
                    _PendingCommand(item.name, item.zone, SLEW, planned_start, PLAN_BATCH,
                                    tier, self._tolerance(tier))
                )
                arrivals.append(
                    self._arrival(item, verdict, reason, "early", tier, required,
                                  latest_safe, planned_start, lag_action, readback_ok)
                )
            else:
                # 来不及：提前标出并按镜区口径在起转前完成处置。
                planned_tier, effective_action, kind = self._plan_lag(
                    item, lag_action, tier, window.launch_at, pending
                )
                arrivals.append(
                    self._arrival(item, verdict, reason, "none", planned_tier, required,
                                  None, None, effective_action, readback_ok)
                )

        commands = self._seal(pending, window.round_id, window.launch_at, start_seq=0)
        return RoundPlan(
            window=window,
            policy=self._policy,
            zone_policies=self._zone_policies,
            arrivals=tuple(arrivals),
            commands=commands,
            plan_command_count=len(commands),
        )

    def judge(
        self,
        plan: RoundPlan,
        readbacks: dict[str, Readback],
        judged_at: float,
    ) -> JudgeReport:
        """在截止时刻按回读逐台判定，并封存落后处置指令（按镜区批量）。"""

        if judged_at + 1e-9 < plan.window.deadline_at:
            raise InterlockError("未到截止时刻不得判定到位", component="arrival")

        entries: list[JudgeEntry] = []
        pending: list[_PendingCommand] = []

        for arrival in plan.arrivals:
            if arrival.verdict == SKIPPED:
                entries.append(
                    JudgeEntry(
                        arrival.mirror, arrival.zone, NOT_TRACKED, "excluded_or_manual",
                        "", arrival.tier, arrival.tier, arrival.tolerance_deg,
                    )
                )
                continue
            if arrival.verdict == INFEASIBLE and arrival.lag_action == LAG_EXIT:
                # 计划阶段已判退出：判定时刻只确认退出结论，不再要求回读。
                entries.append(
                    JudgeEntry(
                        arrival.mirror, arrival.zone, EXITED, "planned_exit", LAG_EXIT,
                        arrival.tier, arrival.tier, arrival.tolerance_deg,
                    )
                )
                continue

            readback = readbacks.get(arrival.mirror)
            ok = readback is not None and readback.valid_at(
                judged_at, plan.policy.readback_stale_after_s
            )
            if ok:
                actual = readback.angles
                assert actual is not None
                error = angle_error(actual, arrival.target)
                arrived = (
                    abs(error.azimuth_deg) <= arrival.tolerance_deg + 1e-9
                    and abs(error.elevation_deg) <= arrival.tolerance_deg + 1e-9
                )
                reason = "within_tolerance" if arrived else "slew_incomplete"
            else:
                # 回读缺失/过期是部分失败：一律按没跟上处理。
                arrived = False
                reason = "readback_missing" if readback is None or readback.angles is None else "readback_stale"

            if arrived:
                entries.append(
                    JudgeEntry(
                        arrival.mirror, arrival.zone, ARRIVED, reason, "",
                        arrival.tier, arrival.tier, arrival.tolerance_deg,
                    )
                )
            else:
                zone_action = self._lag_action(arrival.zone)
                batch_id = f"{LAG_BATCH_PREFIX}-{arrival.zone}"
                if zone_action == LAG_DEGRADE and arrival.tier < plan.policy.max_tier:
                    to_tier = arrival.tier + 1
                    pending.append(
                        _PendingCommand(arrival.mirror, arrival.zone, TRACK_DEGRADED, judged_at,
                                        batch_id, to_tier, self._tolerance(to_tier))
                    )
                    entries.append(
                        JudgeEntry(
                            arrival.mirror, arrival.zone, DEGRADED, reason, LAG_DEGRADE,
                            arrival.tier, to_tier, self._tolerance(to_tier), batch_id,
                        )
                    )
                else:
                    # 镜区要求退出，或已在最低档无处可降：退出聚焦。
                    pending.append(
                        _PendingCommand(arrival.mirror, arrival.zone, EXIT_FOCUS, judged_at,
                                        batch_id, arrival.tier, arrival.tolerance_deg)
                    )
                    entries.append(
                        JudgeEntry(
                            arrival.mirror, arrival.zone, LAGGED, reason, LAG_EXIT,
                            arrival.tier, arrival.tier, arrival.tolerance_deg, batch_id,
                        )
                    )

        # 同镜区的落后镜子必须落在同一批次：按镜区顺序排序、序号连续。
        commands = self._seal(pending, plan.round_id, launch_at=float("inf"),
                              start_seq=plan.plan_command_count)
        batches: list[str] = []
        for command in commands:
            if command.batch_id not in batches:
                batches.append(command.batch_id)
        return JudgeReport(plan.round_id, judged_at, tuple(entries), commands, tuple(batches))

    # ---- 内部装配与封存 -----------------------------------------------

    def _plan_lag(
        self,
        item: MirrorInput,
        lag_action: str,
        tier: int,
        launch_at: float,
        pending: list[_PendingCommand],
    ) -> tuple[int, str, str]:
        """为来不及的镜子登记计划内处置指令，返回 (处置后档数, 实际动作, 指令种类)。"""

        if lag_action == LAG_DEGRADE and tier < self._policy.max_tier:
            to_tier = tier + 1
            pending.append(
                _PendingCommand(item.name, item.zone, TRACK_DEGRADED, launch_at,
                                PLAN_BATCH, to_tier, self._tolerance(to_tier))
            )
            return to_tier, LAG_DEGRADE, TRACK_DEGRADED
        if lag_action == LAG_DEGRADE:
            # 已在最低精度档：无处可降，退出聚焦。
            pending.append(
                _PendingCommand(item.name, item.zone, EXIT_FOCUS, launch_at,
                                PLAN_BATCH, tier, self._tolerance(tier))
            )
            return tier, LAG_EXIT, EXIT_FOCUS
        pending.append(
            _PendingCommand(item.name, item.zone, EXIT_FOCUS, launch_at,
                            PLAN_BATCH, tier, self._tolerance(tier))
        )
        return tier, LAG_EXIT, EXIT_FOCUS

    def _skipped(self, item: MirrorInput) -> MirrorArrival:
        return MirrorArrival(
            mirror=item.name, zone=item.zone, verdict=SKIPPED, reason="excluded_or_manual",
            start_kind="none", tier=item.tier, tolerance_deg=self._tolerance(item.tier),
            target=item.target, actual=item.readback.angles, readback_ts=item.readback.ts,
            required_slew_s=0.0, latest_safe_start=None, planned_start=None, lag_action="",
        )

    def _arrival(
        self,
        item: MirrorInput,
        verdict: str,
        reason: str,
        start_kind: str,
        tier: int,
        required: float,
        latest_safe: float | None,
        planned_start: float | None,
        lag_action: str,
        readback_ok: bool,
    ) -> MirrorArrival:
        return MirrorArrival(
            mirror=item.name, zone=item.zone, verdict=verdict, reason=reason,
            start_kind=start_kind, tier=tier, tolerance_deg=self._tolerance(tier),
            target=item.target,
            actual=item.readback.angles if readback_ok else None,
            readback_ts=item.readback.ts if readback_ok else None,
            required_slew_s=required, latest_safe_start=latest_safe,
            planned_start=planned_start, lag_action=lag_action,
        )

    def _seal(
        self,
        pending: list[_PendingCommand],
        round_id: str,
        launch_at: float,
        start_seq: int,
    ) -> tuple[PlannedCommand, ...]:
        """按固定顺序封存：阶段 → 计划时刻 → 镜区 → 镜名，并连续编号。"""

        def sort_key(cmd: _PendingCommand) -> tuple:
            if cmd.batch_id == PLAN_BATCH:
                if cmd.kind == SLEW and cmd.scheduled_at < launch_at:
                    phase = PHASE_EARLY_SLEW
                elif cmd.kind == EXIT_FOCUS:
                    phase = PHASE_PLAN_EXIT
                elif cmd.kind == TRACK_DEGRADED:
                    phase = PHASE_PLAN_DEGRADE
                else:
                    phase = PHASE_NORMAL_SLEW
            else:
                phase = PHASE_LAG_EXIT if cmd.kind == EXIT_FOCUS else PHASE_LAG_DEGRADE
            return (
                phase,
                cmd.scheduled_at,
                self._zone_order.get(cmd.zone, len(self._zone_order)),
                cmd.mirror,
            )

        ordered = sorted(pending, key=sort_key)
        sealed: list[PlannedCommand] = []
        for offset, cmd in enumerate(ordered):
            seq = start_seq + offset + 1
            sealed.append(
                PlannedCommand(
                    seq=seq,
                    command_id=f"cmd-{round_id}-{seq:04d}",
                    round_id=round_id,
                    mirror=cmd.mirror,
                    zone=cmd.zone,
                    kind=cmd.kind,
                    scheduled_at=cmd.scheduled_at,
                    batch_id=cmd.batch_id,
                    tier=cmd.tier,
                    tolerance_deg=cmd.tolerance_deg,
                )
            )
        return tuple(sealed)
