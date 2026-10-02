"""一轮到位计划的纯判定：时间预算、提前起转、镜区处置与指令次序。"""

from __future__ import annotations

import pytest

from cspplant.config import HeliostatLimits
from cspplant.heliostat import arrival
from cspplant.heliostat.aim import AimAngles
from cspplant.heliostat.arrival import (
    ACTION_AIM,
    ACTION_DEFOCUS,
    ACTION_SKIP,
    EARLY_START,
    GRADE_COARSE,
    IN_TIME,
    LAG,
    SKIP,
    ArrivalInput,
    build_round_plan,
    lead_ticks,
    required_time_s,
)

HL = HeliostatLimits(0.6, 1.2, 0.9, 12.0, 450.0, 0.94)
TARGET = AimAngles(180.0, 30.0)


def _input(name: str, group: str, az_off: float = 0.0, el: float = 30.0) -> ArrivalInput:
    return ArrivalInput(name, group, AimAngles(180.0 + az_off, el), TARGET)


def test_required_time_takes_the_slower_axis():
    # 方位差 12 度（10 秒），仰角差 18 度（20 秒）→ 整镜时间取仰角。
    need, az, el = required_time_s(AimAngles(168.0, 12.0), TARGET, 1.2, 0.9)
    assert az == pytest.approx(12.0)
    assert el == pytest.approx(18.0)
    assert need == pytest.approx(20.0)
    # 方位跨 0/360 走最短弧：350 -> 10 只差 20 度。
    need_short, _, _ = required_time_s(AimAngles(350.0, 30.0), AimAngles(10.0, 30.0), 1.0, 1.0)
    assert need_short == pytest.approx(20.0)


def test_verdicts_split_by_round_budget_and_lead_cap():
    # 目标取当前角本身（无位移），只留方位差：窗口 10 秒、提前上限 10 秒。
    at = AimAngles(200.0, 30.0)
    pin = ArrivalInput("m-in", "a", AimAngles(195.0, 30.0), at)    # 5 度 → 来得及
    pearly = ArrivalInput("m-early", "a", AimAngles(182.0, 30.0), at)  # 18 度 → 提前 5 秒
    plag = ArrivalInput("m-lag", "b", AimAngles(260.0, 30.0), at)  # 60 度 → 50 秒，来不及
    plan = build_round_plan(
        "p", 1, 0.0, [pin, pearly, plag],
        period_s=10.0,
        max_lead_s=10.0,
        azimuth_rate_deg_s=1.2,
        elevation_rate_deg_s=0.9,
        coarse_tolerance_deg=2.0,
        policies={"a": "coarse", "b": "exit"},
    )
    by = {item.mirror: item for item in plan.entries}
    assert by["m-in"].verdict == IN_TIME
    assert by["m-in"].lead_s == 0.0
    assert by["m-early"].verdict == EARLY_START
    assert by["m-early"].lead_s == pytest.approx(5.0)
    assert by["m-lag"].verdict == LAG
    assert plan.count(EARLY_START) == 1


def test_lag_disposition_is_per_mirror_but_policy_comes_from_zone():
    # a 区低一档继续追日；b/c 区直接退出聚焦。判定逐台，同区不连坐。
    plan = build_round_plan(
        "p", 1, 0.0,
        [_input("a1", "a", 60.0), _input("a2", "a", 1.0), _input("b1", "b", 60.0),
         _input("c1", "c", 60.0), _input("x1", "x", 60.0)],
        period_s=1.0,
        max_lead_s=0.0,
        azimuth_rate_deg_s=1.2,
        elevation_rate_deg_s=0.9,
        coarse_tolerance_deg=2.0,
        policies={"a": "coarse", "b": "exit", "c": "exit"},
    )
    by = {item.mirror: item for item in plan.entries}
    assert by["a1"].action == ACTION_AIM and by["a1"].grade == GRADE_COARSE
    assert by["a1"].phase == "start"
    assert by["b1"].action == ACTION_DEFOCUS and by["b1"].phase == "exit"
    assert by["c1"].action == ACTION_DEFOCUS and by["c1"].phase == "exit"
    # 未登记镜区按最保守的退出聚焦。
    assert by["x1"].action == ACTION_DEFOCUS and by["x1"].policy == "exit"
    # 同区另一台来得及就正常对焦，不被连坐。
    assert by["a2"].verdict == IN_TIME and by["a2"].grade == "fine"


def test_readback_missing_is_lag_regardless_of_zone():
    plan = build_round_plan(
        "p", 1, 0.0,
        [ArrivalInput("a9", "a", None, TARGET), ArrivalInput("b9", "b", None, TARGET)],
        period_s=60.0,
        max_lead_s=30.0,
        azimuth_rate_deg_s=1.2,
        elevation_rate_deg_s=0.9,
        coarse_tolerance_deg=2.0,
        policies={"a": "coarse", "b": "exit"},
    )
    by = {item.mirror: item for item in plan.entries}
    for item in by.values():
        assert item.verdict == LAG
        assert item.reason == "readback_missing"
        assert item.need_s is None
    assert by["a9"].grade == GRADE_COARSE and by["a9"].action == ACTION_AIM
    assert by["b9"].action == ACTION_DEFOCUS and by["b9"].phase == "exit"


def test_commands_order_starts_before_exits_with_early_batch_first():
    plan = build_round_plan(
        "p", 1, 0.0,
        [_input("a-lag", "a", 60.0), _input("b-lag", "b", 60.0),
         _input("a-early", "a", 18.0), _input("a-in", "a", 1.0)],
        period_s=10.0,
        max_lead_s=10.0,
        azimuth_rate_deg_s=1.2,
        elevation_rate_deg_s=0.9,
        coarse_tolerance_deg=2.0,
        policies={"a": "coarse", "b": "exit"},
    )
    commands = plan.commands()
    phases = [cmd.phase for cmd in commands]
    assert phases == sorted(phases, key=lambda p: 0 if p == "start" else 1)
    starts = [cmd for cmd in commands if cmd.phase == "start"]
    verdicts = [cmd.verdict for cmd in starts]
    assert verdicts.index(EARLY_START) < verdicts.index(LAG) < verdicts.index(IN_TIME)
    # 提前起转组按提前量降序，其余同批按区、镜名定序（时序确定性）。
    early = [cmd for cmd in starts if cmd.verdict == EARLY_START]
    assert [cmd.lead_s for cmd in early] == sorted((cmd.lead_s for cmd in early), reverse=True)
    assert [cmd.seq for cmd in commands] == list(range(len(commands)))
    # 退出相位只包含退出指令。
    assert all(cmd.action == ACTION_DEFOCUS for cmd in commands if cmd.phase == "exit")


def test_skipped_mirrors_are_recorded_but_get_no_command():
    plan = build_round_plan(
        "p", 1, 0.0, [_input("a1", "a", 1.0)],
        period_s=10.0, max_lead_s=0.0,
        azimuth_rate_deg_s=1.2, elevation_rate_deg_s=0.9, coarse_tolerance_deg=2.0,
        policies={"a": "coarse"},
        skipped=(("m021", "a", "manual"), ("m022", "a", "excluded")),
    )
    assert plan.count(SKIP) == 2
    assert all(item.reason in ("manual", "excluded") for item in plan.entries if item.verdict == SKIP)
    assert all(cmd.action != ACTION_SKIP for cmd in plan.commands())
    assert len(plan.commands()) == 1


def test_lead_ticks_rounds_up_so_early_mirrors_get_enough_time():
    plan = build_round_plan(
        "p", 1, 0.0, [_input("a1", "a", 13.8)],
        period_s=10.0, max_lead_s=10.0,
        azimuth_rate_deg_s=1.2, elevation_rate_deg_s=0.9, coarse_tolerance_deg=2.0,
        policies={"a": "coarse"},
    )
    # need = 11.5 秒，提前 1.5 秒 → 整秒预算向上取整为 2 秒。
    assert plan.named("a1").verdict == EARLY_START
    assert lead_ticks(plan) == 2


def test_plan_dictionary_is_a_stable_snapshot():
    plan = build_round_plan(
        "p", 1, 123.0, [_input("a1", "a", 1.0), _input("b1", "b", 60.0)],
        period_s=10.0, max_lead_s=0.0,
        azimuth_rate_deg_s=1.2, elevation_rate_deg_s=0.9, coarse_tolerance_deg=2.0,
        policies={"a": "coarse", "b": "exit"},
    )
    payload = plan.as_dict()
    assert payload["plan_id"] == "p"
    assert payload["counts"]["lag"] == 1
    # 指令与条目是同一判定导出的，镜名集合完全一致。
    assert {c["mirror"] for c in payload["commands"]} == {
        e["mirror"] for e in payload["entries"] if e["action"] != ACTION_SKIP
    }


def test_invalid_budget_rejected():
    with pytest.raises(ValueError):
        build_round_plan(
            "p", 1, 0.0, [], period_s=0.0, max_lead_s=0.0,
            azimuth_rate_deg_s=1.0, elevation_rate_deg_s=1.0, coarse_tolerance_deg=2.0,
        )
