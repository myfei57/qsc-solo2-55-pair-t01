"""一轮到位计划：可行性估算、提前起转、镜区落后策略、回读部分失败与批次时序。"""

from __future__ import annotations

import pytest

from cspplant.app import build_plant
from cspplant.clock import ManualClock
from cspplant.config import default_config
from cspplant.errors import InterlockError
from cspplant.heliostat import AimAngles, Heliostat, HeliostatRegistry
from cspplant.heliostat.arrival import (
    ARRIVED,
    DEGRADED,
    EXITED,
    FEASIBLE,
    INFEASIBLE,
    LAG_DEGRADE,
    LAG_EXIT,
    LAGGED,
    SKIPPED,
    SLEW,
    EXIT_FOCUS,
    TRACK_DEGRADED,
    ArrivalPlanner,
    ArrivalPolicy,
    MirrorInput,
    Readback,
    RoundWindow,
)
from cspplant.field.arrival import DEFAULT_ZONE_POLICIES

# 两轴限速取自默认配置：方位 1.2 度/秒，仰角 0.9 度/秒，对焦公差 0.6 度。
AZ_RATE = 1.2
EL_RATE = 0.9
TOL = 0.6


def _planner(zones: dict[str, str] | None = None, policy: ArrivalPolicy | None = None) -> ArrivalPlanner:
    return ArrivalPlanner(
        zone_policies=zones or {"a": LAG_EXIT, "b": LAG_DEGRADE},
        azimuth_rate_deg_s=AZ_RATE,
        elevation_rate_deg_s=EL_RATE,
        base_tolerance_deg=TOL,
        policy=policy,
    )


def _input(
    name: str,
    zone: str,
    target: AimAngles,
    angles: AimAngles | None,
    *,
    ts: float = 0.0,
    tier: int = 1,
    participating: bool = True,
) -> MirrorInput:
    return MirrorInput(
        name=name,
        zone=zone,
        target=target,
        readback=Readback(ts=ts, angles=angles),
        tier=tier,
        participating=participating,
    )


def _window(release: float = 0.0, launch: float = 60.0, deadline: float = 120.0) -> RoundWindow:
    return RoundWindow("r-1", release, launch, deadline)


# ---- 可行性与提前起转 -----------------------------------------------


def test_feasible_when_slew_fits_window():
    planner = _planner()
    # 需转 60 度 / 1.2 = 50 秒，60 秒窗口内够。
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0))],
    )
    arrival = plan.arrival("m1")
    assert arrival.verdict == FEASIBLE
    assert arrival.start_kind == "normal"
    assert arrival.planned_start == pytest.approx(60.0)
    assert plan.commands[0].kind == SLEW
    assert plan.commands[0].scheduled_at == pytest.approx(60.0)


def test_early_start_when_latest_safe_start_precedes_launch():
    planner = _planner(policy=ArrivalPolicy(early_lead_s=60.0))
    # 需转 100 度 / 1.2 ≈ 83.3 秒：最晚起转 ≈ 36.7 秒，落在提前起转窗口内。
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(100.0, 0.0), AimAngles(0.0, 0.0))],
    )
    arrival = plan.arrival("m1")
    assert arrival.verdict == "early"
    assert arrival.start_kind == "early"
    assert arrival.latest_safe_start == pytest.approx(120.0 - 100.0 / AZ_RATE)
    assert arrival.planned_start == pytest.approx(120.0 - 100.0 / AZ_RATE)
    assert plan.commands[0].scheduled_at < 60.0


def test_infeasible_when_even_immediate_start_is_too_late():
    planner = _planner()
    # 需转 180 度 / 1.2 = 150 秒，超过截止距封存的 120 秒。
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0))],
    )
    assert plan.arrival("m1").verdict == INFEASIBLE
    assert plan.arrival("m1").reason == "insufficient_slew_time"


def test_skipped_mirrors_get_no_commands():
    planner = _planner()
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0), participating=False)],
    )
    assert plan.arrival("m1").verdict == SKIPPED
    assert plan.commands == ()


# ---- 回读缺失/过期：部分失败按来不及 --------------------------------


def test_missing_readback_at_plan_time_is_infeasible():
    planner = _planner()
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(60.0, 0.0), None)],
    )
    arrival = plan.arrival("m1")
    assert arrival.verdict == INFEASIBLE
    assert arrival.reason == "readback_missing"
    assert arrival.actual is None


def test_stale_readback_at_plan_time_is_infeasible():
    planner = _planner(policy=ArrivalPolicy(readback_stale_after_s=2.0))
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0), ts=-10.0)],
    )
    assert plan.arrival("m1").verdict == INFEASIBLE
    assert plan.arrival("m1").reason == "readback_stale"


# ---- 来不及按镜区策略处置 -------------------------------------------


def test_exit_zone_marks_infeasible_mirror_for_exit():
    planner = _planner({"a": LAG_EXIT})
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0))],
    )
    arrival = plan.arrival("m1")
    assert arrival.verdict == INFEASIBLE
    assert arrival.lag_action == LAG_EXIT
    assert plan.commands[0].kind == EXIT_FOCUS
    assert arrival.tier == 1


def test_degrade_zone_widens_tolerance_one_tier():
    planner = _planner({"b": LAG_DEGRADE})
    plan = planner.plan_round(
        _window(),
        [_input("m1", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0))],
    )
    arrival = plan.arrival("m1")
    assert arrival.lag_action == LAG_DEGRADE
    assert arrival.tier == 2
    assert arrival.tolerance_deg == pytest.approx(TOL * 2.0)
    assert plan.commands[0].kind == TRACK_DEGRADED


def test_degrade_at_lowest_tier_falls_back_to_exit():
    planner = _planner({"b": LAG_DEGRADE}, policy=ArrivalPolicy(max_tier=2))
    plan = planner.plan_round(
        _window(),
        [_input("m1", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0), tier=2)],
    )
    arrival = plan.arrival("m1")
    assert arrival.lag_action == LAG_EXIT
    assert plan.commands[0].kind == EXIT_FOCUS


def test_unknown_zone_policy_is_rejected():
    with pytest.raises(InterlockError):
        ArrivalPlanner({"z": "hold"}, AZ_RATE, EL_RATE, TOL)
    planner = _planner({"a": LAG_EXIT})
    with pytest.raises(InterlockError):
        planner.plan_round(
            _window(),
            [_input("m1", "zzz", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0))],
        )


# ---- 封存时序：提前 → 退出 → 降档 → 正常起转 -------------------------


def test_manifest_orders_early_exit_degrade_normal_and_numbers_continuously():
    planner = _planner({"a": LAG_EXIT, "b": LAG_DEGRADE})
    inputs = [
        _input("early", "b", AimAngles(100.0, 0.0), AimAngles(0.0, 0.0)),
        _input("ok", "b", AimAngles(10.0, 0.0), AimAngles(0.0, 0.0)),
        _input("out", "a", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0)),
        _input("down", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0)),
    ]
    plan = planner.plan_round(_window(), inputs)
    kinds = [(c.mirror, c.kind) for c in plan.commands]
    assert [kind for _, kind in kinds] == [SLEW, EXIT_FOCUS, TRACK_DEGRADED, SLEW]
    assert [mirror for mirror, _ in kinds] == ["early", "out", "down", "ok"]
    assert [c.seq for c in plan.commands] == [1, 2, 3, 4]
    assert [c.command_id for c in plan.commands] == [
        "cmd-r-1-0001",
        "cmd-r-1-0002",
        "cmd-r-1-0003",
        "cmd-r-1-0004",
    ]


# ---- 截止判定与按镜区批量处置 ----------------------------------------


def _infeasible_round(planner: ArrivalPlanner) -> object:
    return planner.plan_round(
        _window(),
        [
            _input("a1", "a", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0)),
            _input("b1", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0)),
            _input("b2", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0)),
        ],
    )


def test_judge_confirms_planned_exit_without_demanding_readback():
    planner = _planner({"a": LAG_EXIT, "b": LAG_DEGRADE})
    plan = _infeasible_round(planner)
    # a 区计划退出，判定时刻即使不给回读也是 exited，而不是 lagged。
    report = planner.judge(plan, {}, judged_at=120.0)
    assert report.entry("a1").status == EXITED
    assert report.entry("a1").reason == "planned_exit"


def test_judge_missing_readback_counts_as_lagged_and_batches_by_zone():
    planner = _planner({"a": LAG_EXIT, "b": LAG_DEGRADE})
    plan = _infeasible_round(planner)
    # 不给任何回读：计划内降档的 b 区两台按没跟上处理（再降一档）。
    report = planner.judge(plan, {}, judged_at=120.0)
    assert report.entry("b1").status == DEGRADED
    assert report.entry("b2").status == DEGRADED
    assert report.entry("b1").reason == "readback_missing"
    assert report.entry("b1").batch_id == "lag-b"
    assert report.entry("b2").batch_id == "lag-b"
    assert report.batches == ("lag-b",)
    # 同一镜区落在同一批次，序号连续且紧接计划指令之后。
    assert [c.mirror for c in report.commands] == ["b1", "b2"]
    assert [c.seq for c in report.commands] == [len(plan.commands) + 1, len(plan.commands) + 2]
    assert all(c.batch_id == "lag-b" for c in report.commands)


def test_judge_exit_zone_lag_commands_share_one_zone_batch():
    planner = _planner({"a": LAG_EXIT, "b": LAG_EXIT})
    plan = planner.plan_round(
        _window(),
        [
            _input("a1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0)),
            _input("b1", "b", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0)),
        ],
    )
    # 两台都计划正常起转，但截止时仍停在零位（回读显示没转）。
    readbacks = {
        "a1": Readback(120.0, AimAngles(0.0, 0.0)),
        "b1": Readback(120.0, AimAngles(0.0, 0.0)),
    }
    report = planner.judge(plan, readbacks, judged_at=120.0)
    assert report.entry("a1").status == LAGGED
    assert report.entry("b1").status == LAGGED
    assert report.entry("a1").batch_id == "lag-a"
    assert report.entry("b1").batch_id == "lag-b"
    assert list(report.batches) == ["lag-a", "lag-b"]
    assert [c.batch_id for c in report.commands] == ["lag-a", "lag-b"]


def test_judge_arrived_uses_tier_tolerance():
    planner = _planner({"b": LAG_DEGRADE})
    # 来不及 → 计划降到 2 档（公差 1.2 度）；截止时还差 1.0 度：全精度算没到位、
    # 降档公差内算到位。
    plan = planner.plan_round(
        _window(),
        [_input("b1", "b", AimAngles(180.0, 0.0), AimAngles(0.0, 0.0))],
    )
    report = planner.judge(
        plan, {"b1": Readback(120.0, AimAngles(179.0, 0.0))}, judged_at=120.0
    )
    assert report.entry("b1").status == ARRIVED


def test_judge_before_deadline_is_rejected():
    planner = _planner()
    plan = planner.plan_round(
        _window(),
        [_input("m1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0))],
    )
    with pytest.raises(InterlockError):
        planner.judge(plan, {}, judged_at=119.0)


def test_stale_readback_at_judge_is_lagged():
    planner = _planner({"a": LAG_EXIT})
    plan = planner.plan_round(
        _window(),
        [_input("a1", "a", AimAngles(60.0, 0.0), AimAngles(0.0, 0.0))],
    )
    report = planner.judge(
        plan, {"a1": Readback(110.0, AimAngles(60.0, 0.0))}, judged_at=120.0
    )
    assert report.entry("a1").status == LAGGED
    assert report.entry("a1").reason == "readback_stale"


# ---- 平台端到端：封存展示、时序执行、审计同源 -------------------------


def test_platform_round_plans_dispatches_judges_and_audits(platform):
    platform.start()
    platform.track(820.0, 720.0)
    report = platform.track_round(window_s=60.0, minute_of_day=720.0)
    assert report["ok"] is True
    assert report["judge"]["statuses"] == {"arrived": 123}

    # 控制台展示与判定同源：封存计划已供视图读取。
    view = platform.arrivals.view()
    assert view["plan"]["round_id"] == report["round_id"]
    assert len(view["plan"]["command_manifest"]) == 123
    assert view["judge"]["round_id"] == report["round_id"]

    # 审计：封存、逐台结论、下发、判定各有流水，且轮次一致。
    actions = [entry["action"] for entry in platform.audit.tail(2000)]
    assert "arrival_round_planned" in actions
    assert "arrival_round_judged" in actions
    assert "arrival_command_issued" in actions
    round_records = [
        entry for entry in platform.audit.tail(2000)
        if entry["payload"].get("round_id") == report["round_id"]
    ]
    assert round_records


def test_platform_short_round_applies_zone_policy_and_lag_batches(platform):
    platform.start()
    platform.track(820.0, 400.0)
    report = platform.track_round(window_s=2.0, minute_of_day=430.0)
    statuses = report["judge"]["statuses"]
    # a 区 41 台计划退出；b/c 区 82 台计划降档且在放宽公差内追日。
    assert statuses.get("exited") == 41
    assert statuses.get("degraded") == 82
    assert report["plan"]["verdicts"][INFEASIBLE] == 123

    # 落盘：计划文档已写进存储。
    doc = platform.store.read(f"arrival_plan-{report['round_id']}")
    assert doc["round_id"] == report["round_id"]
    assert doc["commands"] == 123


def test_platform_missing_readback_is_handled_as_not_keeping_up(platform):
    platform.start()
    platform.track(820.0, 400.0)
    # 正常窗口让 b 区镜子处于追日状态，随后摘掉一台 b 区镜子的截止回读。
    report = platform.track_round(
        window_s=60.0, minute_of_day=720.0, missing_readbacks=["m060"]
    )
    entry = next(item for item in report["judge"]["entries"] if item["mirror"] == "m060")
    # 回读缺失：按没跟上处理；b 区降一档，落在 lag-b 批次。
    assert entry["status"] == DEGRADED
    assert entry["reason"] == "readback_missing"
    assert entry["batch_id"] == "lag-b"
    assert entry["to_tier"] == entry["from_tier"] + 1

    # 审计里必须能看到该轮回读缺失与镜区批次处置。
    recent = platform.audit.tail(2000)
    judged = [e for e in recent if e["action"] == "arrival_round_judged"][-1]
    assert "lag-b" in judged["payload"].get("batches", "")


def test_plan_manifest_cannot_be_reordered_when_issuing(platform, clock):
    platform.start()
    platform.track(820.0, 720.0)
    plan = platform.arrivals.plan_round(
        platform.arrivals.targets_for(platform.sun_at(720.0)), window_s=60.0
    )
    second = plan.commands[1].command_id
    with pytest.raises(InterlockError):
        platform.arrivals.issue_command(second)
    # 按序第一条可以发。
    issued = platform.arrivals.issue_command(plan.commands[0].command_id)
    assert issued["seq"] == 1


def test_default_zone_policies_cover_all_three_zones(platform):
    assert set(DEFAULT_ZONE_POLICIES) == {"a", "b", "c"}
    assert DEFAULT_ZONE_POLICIES["a"] == LAG_EXIT
    assert DEFAULT_ZONE_POLICIES["b"] == LAG_DEGRADE
    assert DEFAULT_ZONE_POLICIES["c"] == LAG_DEGRADE
    view = platform.status()["arrival"]
    assert view["round_id"] is None
