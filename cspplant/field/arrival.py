"""到位计划的执行、回读判定与审计。

:class:`ArrivalRunner` 把纯计算的 :class:`~cspplant.heliostat.arrival.ArrivalPlanner`
接到真实镜场上，守住四条现场约束：

1. **只按封存清单下发**：运行器不能临时编指令，每条下发的指令都必须来自
   计划封存的 ``command_manifest``，且只能按封存顺序（``seq``）下发；
2. **时序不能乱**：``issue_due`` 只下发计划时刻已到的下一条，提前起转指令
   早于正常起转，计划内退出/降档夹在两者之间，截止后的落后处置追加在后；
3. **回读缺失按没跟上处理**：判定用的回读缺失或过期即部分失败，绝不因为
   "没有回读"就默认到位；
4. **指令、判定、审计同源**：下发、判定与控制台展示用的是同一个封存计划，
   每个动作都经事件总线落同一条审计流水。

落后处置口径定死为**按镜区批量**：判定逐台做，但同镜区的落后镜子共用一个
``lag-<zone>`` 批次，封存排序与序号连续，不允许逐台穿插处置。
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from ..errors import InterlockError
from ..eventbus import Event
from ..heliostat import AimAngles, HeliostatRegistry
from ..heliostat.aim import aim_angles_for
from ..heliostat.arrival import (
    ARRIVED,
    DEGRADED,
    EARLY,
    EXITED,
    EXIT_FOCUS,
    FEASIBLE,
    INFEASIBLE,
    LAG_DEGRADE,
    LAG_EXIT,
    LAGGED,
    NOT_TRACKED,
    SKIPPED,
    SLEW,
    TRACK_DEGRADED,
    ArrivalPlanner,
    ArrivalPolicy,
    JudgeReport,
    MirrorInput,
    PlannedCommand,
    Readback,
    RoundPlan,
    RoundWindow,
)
from ..heliostat.drive import TRACKING
from ..heliostat.sun import SunPosition
from ..ids import IdFactory
from ..store.jsonfile import JsonStore
from .field import FieldController

ROUND_DOCUMENT = "arrival_round"
PLAN_DOCUMENT_PREFIX = "arrival_plan"
JUDGE_DOCUMENT_PREFIX = "arrival_judge"
ROUND_STATE_DOCUMENT = "arrival_state"

# 默认镜区落后口径：a 区直接退出聚焦，b/c 区降一档精度继续追日。
DEFAULT_ZONE_POLICIES: dict[str, str] = {"a": LAG_EXIT, "b": LAG_DEGRADE, "c": LAG_DEGRADE}


@dataclass(frozen=True)
class ReadbackCapture:
    """一次全场回读采集的结果。"""

    ts: float
    readbacks: dict[str, Readback]
    missing: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "ts": round(self.ts, 3),
            "mirrors": sorted(self.readbacks),
            "missing": list(self.missing),
        }


class ArrivalRunner:
    """按封存计划驱动镜场、判定到位并落审计。"""

    def __init__(
        self,
        field: FieldController,
        registry: HeliostatRegistry,
        *,
        bus,
        clock,
        store: JsonStore | None = None,
        zone_policies: dict[str, str] | None = None,
        policy: ArrivalPolicy | None = None,
        id_factory: IdFactory | None = None,
    ) -> None:
        self._field = field
        self._registry = registry
        self._bus = bus
        self._clock = clock
        self._store = store
        self._planner = ArrivalPlanner(
            zone_policies=zone_policies or DEFAULT_ZONE_POLICIES,
            azimuth_rate_deg_s=field.heliostat_limits.azimuth_rate_deg_s,
            elevation_rate_deg_s=field.heliostat_limits.elevation_rate_deg_s,
            base_tolerance_deg=field.heliostat_limits.aim_tolerance_deg,
            policy=policy,
        )
        self._ids = id_factory or IdFactory("round")
        self._tiers: dict[str, int] = {}
        self._latest_plan: RoundPlan | None = None
        self._latest_judge: JudgeReport | None = None
        self._next_plan_seq = 0
        self._lock = threading.RLock()
        self._restore()

    # ---- 属性与恢复 ---------------------------------------------------

    def tier_of(self, mirror: str) -> int:
        return self._tiers.get(mirror, 1)

    def _restore(self) -> None:
        if self._store is None or not self._store.exists(ROUND_STATE_DOCUMENT):
            return
        state = self._store.read(ROUND_STATE_DOCUMENT)
        if isinstance(state, dict):
            self._tiers = {str(name): int(tier) for name, tier in state.get("tiers", {}).items()}

    def _persist_state(self) -> None:
        if self._store is not None:
            self._store.write(ROUND_STATE_DOCUMENT, {"tiers": dict(self._tiers)})

    # ---- 回读采集 -----------------------------------------------------

    def capture_readbacks(self, now: float | None = None) -> ReadbackCapture:
        """从全部参与追日的驱动采集当前角度；参与镜取不到角度即部分失败。"""

        moment = self._clock.now() if now is None else float(now)
        readbacks: dict[str, Readback] = {}
        missing: list[str] = []
        for mirror in self._registry.all():
            drive = self._field.drive(mirror.name)
            if drive.excluded or drive.manual:
                continue
            angles = drive.current_angles()
            if angles is None:
                missing.append(mirror.name)
            else:
                readbacks[mirror.name] = Readback(ts=moment, angles=angles)
        return ReadbackCapture(ts=moment, readbacks=readbacks, missing=tuple(missing))

    # ---- 计划 ---------------------------------------------------------

    def targets_for(self, sun: SunPosition) -> dict[str, AimAngles]:
        """按当前太阳位置算出全场每台镜子的追日目标角。"""

        return {
            mirror.name: aim_angles_for(
                mirror.x_m, mirror.y_m, mirror.z_m, sun, self._field.aim_point
            )
            for mirror in self._registry.all()
        }

    def build_inputs(
        self, targets: dict[str, AimAngles], capture: ReadbackCapture
    ) -> list[MirrorInput]:
        """把目标角与回读采集装配成计划输入。"""

        inputs: list[MirrorInput] = []
        for mirror in self._registry.all():
            drive = self._field.drive(mirror.name)
            inputs.append(
                MirrorInput(
                    name=mirror.name,
                    zone=mirror.group,
                    target=targets[mirror.name],
                    readback=capture.readbacks.get(mirror.name, Readback(capture.ts, None)),
                    tier=self.tier_of(mirror.name),
                    participating=not drive.excluded and not drive.manual,
                )
            )
        return inputs

    def next_round_id(self) -> str:
        return self._ids.next()

    def plan_round(
        self,
        targets: dict[str, AimAngles],
        *,
        round_id: str | None = None,
        release_at: float | None = None,
        launch_at: float | None = None,
        deadline_at: float | None = None,
        window_s: float = 60.0,
        capture: ReadbackCapture | None = None,
    ) -> RoundPlan:
        """采集回读、封存一轮到位计划，并落审计与盘存（供控制台展示）。"""

        with self._lock:
            moment = self._clock.now()
            release = moment if release_at is None else float(release_at)
            launch = release + window_s if launch_at is None else float(launch_at)
            deadline = launch + window_s if deadline_at is None else float(deadline_at)
            if capture is None:
                capture = self.capture_readbacks(now=release)
            window = RoundWindow(
                round_id=round_id or self.next_round_id(),
                release_at=release,
                launch_at=launch,
                deadline_at=deadline,
            )
            inputs = self.build_inputs(targets, capture)
            plan = self._planner.plan_round(window, inputs)
            self._latest_plan = plan
            self._next_plan_seq = 0
            self._audit_plan(plan, capture)
            if self._store is not None:
                self._store.write(f"{PLAN_DOCUMENT_PREFIX}-{plan.round_id}", plan.as_dict())
                self._store.write(ROUND_DOCUMENT, plan.summary())
            return plan

    def _audit_plan(self, plan: RoundPlan, capture: ReadbackCapture) -> None:
        summary = plan.summary()
        self._bus.publish(
            Event.of(
                "arrival", "arrival", "arrival_round_planned",
                severity="notice",
                message=f"轮次 {plan.round_id} 到位计划已封存",
                round_id=plan.round_id,
                release_at=round(plan.window.release_at, 3),
                launch_at=round(plan.window.launch_at, 3),
                deadline_at=round(plan.window.deadline_at, 3),
                feasible=summary["verdicts"].get(FEASIBLE, 0),
                early=summary["verdicts"].get(EARLY, 0),
                infeasible=summary["verdicts"].get(INFEASIBLE, 0),
                skipped=summary["verdicts"].get(SKIPPED, 0),
                readback_missing=len(capture.missing),
                commands=len(plan.commands),
            )
        )
        for arrival in plan.arrivals:
            self._bus.publish(
                Event.of(
                    "arrival", "arrival", "arrival_mirror_verdict",
                    asset=arrival.mirror,
                    severity="warning" if arrival.verdict == INFEASIBLE else "notice",
                    round_id=plan.round_id,
                    mirror=arrival.mirror,
                    zone=arrival.zone,
                    verdict=arrival.verdict,
                    reason=arrival.reason,
                    start_kind=arrival.start_kind,
                    tier=arrival.tier,
                    lag_action=arrival.lag_action,
                    required_slew_s=round(arrival.required_slew_s, 3),
                    planned_start=(
                        None if arrival.planned_start is None
                        else round(arrival.planned_start, 3)
                    ),
                )
            )

    # ---- 下发（严格按封存顺序） ---------------------------------------

    def issue_due(
        self, plan: RoundPlan | None = None, now: float | None = None
    ) -> list[dict]:
        """下发计划时刻已到且轮到封存序号的指令，保持封存顺序不乱。"""

        with self._lock:
            current = self._require_plan(plan)
            moment = self._clock.now() if now is None else float(now)
            commands = list(current.commands)
            issued: list[dict] = []
            while self._next_plan_seq < len(commands):
                command = commands[self._next_plan_seq]
                if command.scheduled_at > moment + 1e-9:
                    break
                self._issue(command)
                self._next_plan_seq += 1
                issued.append(command.as_dict())
            return issued

    def issue_command(self, command_id: str, plan: RoundPlan | None = None) -> dict:
        """下发指定封存指令；不是下一条即拒绝（时序不得插队）。"""

        with self._lock:
            current = self._require_plan(plan)
            commands = list(current.commands)
            if self._next_plan_seq >= len(commands):
                raise InterlockError("封存清单已全部下发", component="arrival")
            command = commands[self._next_plan_seq]
            if command.command_id != command_id:
                raise InterlockError(
                    f"时序插队被拒绝：下一条应是 {command.command_id}，而不是 {command_id}",
                    component="arrival",
                )
            self._issue(command)
            self._next_plan_seq += 1
            return command.as_dict()

    def issued_plan_count(self) -> int:
        return self._next_plan_seq

    def step_tracking(self, dt: float = 1.0) -> int:
        """把仍在追日状态的驱动推进一个步长，返回推进的镜台数。"""

        moved = 0
        for mirror in self._registry.all():
            drive = self._field.drive(mirror.name)
            if drive.state == TRACKING:
                drive.step(dt)
                moved += 1
        return moved

    def _require_plan(self, plan: RoundPlan | None) -> RoundPlan:
        current = plan or self._latest_plan
        if current is None:
            raise InterlockError("还没有封存的到位计划", component="arrival")
        return current

    def _issue(self, command: PlannedCommand) -> None:
        if command.kind == EXIT_FOCUS:
            self._field.drive(command.mirror).defocus()
        else:
            self._field.drive(command.mirror).aim(self._latest_plan.arrival(command.mirror).target)
        self._bus.publish(
            Event.of(
                "arrival", "arrival", "arrival_command_issued",
                asset=command.mirror,
                severity="notice",
                message=self._command_message(command),
                round_id=command.round_id,
                command_id=command.command_id,
                seq=command.seq,
                mirror=command.mirror,
                zone=command.zone,
                kind=command.kind,
                scheduled_at=round(command.scheduled_at, 3),
                batch_id=command.batch_id,
                tier=command.tier,
            )
        )

    @staticmethod
    def _command_message(command: PlannedCommand) -> str:
        if command.kind == SLEW:
            return f"{command.mirror} 按封存计划起转"
        if command.kind == TRACK_DEGRADED:
            return f"{command.mirror} 降档至第 {command.tier} 档精度继续追日"
        return f"{command.mirror} 退出聚焦"

    # ---- 截止判定与落后批次处置 ---------------------------------------

    def judge_round(
        self,
        readbacks: dict[str, Readback] | None = None,
        *,
        judged_at: float | None = None,
        plan: RoundPlan | None = None,
    ) -> JudgeReport:
        """截止时刻逐台判定到位，封存并下发落后处置批次，落审计与盘存。"""

        with self._lock:
            current = self._require_plan(plan)
            moment = current.window.deadline_at if judged_at is None else float(judged_at)
            if readbacks is None:
                capture = self.capture_readbacks(now=moment)
                readbacks = capture.readbacks
            report = self._planner.judge(current, dict(readbacks), moment)
            for command in report.commands:
                self._issue(command)
            for entry in report.entries:
                self._tiers[entry.mirror] = entry.to_tier
            self._latest_judge = report
            self._persist_state()
            self._audit_judge(report)
            if self._store is not None:
                self._store.write(
                    f"{JUDGE_DOCUMENT_PREFIX}-{report.round_id}", report.as_dict()
                )
            return report

    def _audit_judge(self, report: JudgeReport) -> None:
        counts = report.status_counts()
        lagged = counts.get(LAGGED, 0)
        self._bus.publish(
            Event.of(
                "arrival", "arrival", "arrival_round_judged",
                severity="warning" if lagged else "notice",
                message=f"轮次 {report.round_id} 到位判定完成",
                round_id=report.round_id,
                judged_at=round(report.judged_at, 3),
                arrived=counts.get(ARRIVED, 0),
                degraded=counts.get(DEGRADED, 0),
                lagged=lagged,
                exited=counts.get(EXITED, 0),
                not_tracked=counts.get(NOT_TRACKED, 0),
                batches=",".join(report.batches),
                commands=len(report.commands),
            )
        )
        for batch in report.batches:
            members = [entry.mirror for entry in report.entries if entry.batch_id == batch]
            action = LAG_EXIT if any(
                entry.status == LAGGED and entry.batch_id == batch for entry in report.entries
            ) else LAG_DEGRADE
            self._bus.publish(
                Event.of(
                    "arrival", "arrival", "arrival_lag_batch",
                    severity="warning",
                    message=f"镜区批次 {batch} 按 {action} 口径处置 {len(members)} 台",
                    round_id=report.round_id,
                    batch_id=batch,
                    lag_action=action,
                    mirrors=",".join(members),
                    count=len(members),
                )
            )

    # ---- 控制台视图 ---------------------------------------------------

    def summary(self) -> dict:
        """最近一轮计划与判定的计数汇总，供状态快照与审计摘要共用。"""

        plan = self._latest_plan
        if plan is None:
            return {
                "round_id": None,
                "verdicts": {},
                "planned_tracking": 0,
                "statuses": {},
                "tracking_after_judge": 0,
                "pending_plan_commands": 0,
                "batches": [],
            }
        planned_tracking = sum(1 for arrival in plan.arrivals if arrival.tracks)
        judge = self._latest_judge
        return {
            "round_id": plan.round_id,
            "verdicts": plan.summary()["verdicts"],
            "planned_tracking": planned_tracking,
            "statuses": {} if judge is None else judge.status_counts(),
            "tracking_after_judge": (
                planned_tracking if judge is None else sum(1 for entry in judge.entries if entry.tracks)
            ),
            "pending_plan_commands": len(plan.commands) - self._next_plan_seq,
            "batches": [] if judge is None else list(judge.batches),
        }

    def view(self) -> dict:
        """供控制台展示的最近一轮到位计划与判定视图。"""

        plan = self._latest_plan
        return {
            "summary": self.summary(),
            "plan": None if plan is None else plan.as_dict(),
            "judge": None if self._latest_judge is None else self._latest_judge.as_dict(),
            "pending_plan_commands": (
                0 if plan is None else len(plan.commands) - self._next_plan_seq
            ),
            "tiers": dict(sorted(self._tiers.items())),
        }
