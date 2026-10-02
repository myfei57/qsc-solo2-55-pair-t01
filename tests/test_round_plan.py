"""镜场一轮到位计划的编排、逐台处置、回读部分失败与审计一致性。"""

from __future__ import annotations

import dataclasses

import pytest

from cspplant.app import build_plant
from cspplant.config import ConfigError, PlatformConfig, default_config


def _short_round_cfg(**overrides):
    cfg = default_config()
    params = {
        "round_period_s": 2.0,
        "early_lead_max_s": 30.0,
        "coarse_aim_tolerance_deg": 2.5,
    }
    params.update(overrides)
    return dataclasses.replace(cfg, field=dataclasses.replace(cfg.field, **params))


def test_normal_round_marks_every_mirror_in_time(platform):
    """默认时间预算下一轮全部来得及：计划、执行、审计一致。"""

    platform.start()
    report = platform.track(820.0, 720.0)
    assert report["ok"] is True
    assert report["aimed"] == len(platform.heliostats.names())
    counts = report["plan"]["counts"]
    assert counts["in_time"] == 123
    assert counts["early_start"] == counts["lag"] == counts["skip"] == 0
    plan = platform.field.arrival_plan()
    assert plan["counts"]["planned"] == 123


def test_plan_is_single_source_of_truth_for_commands_verify_and_audit(platform):
    platform.start()
    report = platform.track(820.0, 720.0)
    plan = report["plan"]
    # 每条执行结果都能在同一份计划里找到同名条目，动作与档位一致。
    for outcome in report["outcomes"]:
        entry = next(item for item in plan["entries"] if item["mirror"] == outcome["mirror"])
        assert entry["action"] == outcome["action"]
        assert entry["grade"] == outcome["grade"]
    # 审计里落的完整计划就是返回的这份对象。
    full = [
        rec for rec in platform.audit.tail(500)
        if rec["action"] == "arrival_plan" and "entries" in rec["payload"]
    ]
    assert full[-1]["payload"]["entries"] == plan["entries"]
    assert full[-1]["payload"]["commands"] == plan["commands"]


def test_lag_disposition_follows_zone_policy_per_mirror(tmp_path, clock):
    """来不及的镜子逐台判、逐台处置：a 区低档追日，b/c 区退出聚焦。"""

    platform = build_plant(_short_round_cfg(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    platform.track(820.0, 720.0)
    report = platform.track(820.0, 745.0)
    assert report["plan"]["counts"]["lag"] == 123
    for outcome in report["outcomes"]:
        assert outcome["verdict"] == "lag"
        if outcome["group"] == "a":
            assert outcome["action"] == "aim" and outcome["grade"] == "coarse"
        else:
            assert outcome["action"] == "defocus"
            assert outcome["defocused"] is True  # 退出镜确实转到离焦姿态
    assert sum(1 for o in report["outcomes"] if o["grade"] == "coarse") == 41
    assert sum(1 for o in report["outcomes"] if o["action"] == "defocus") == 82


def test_command_sequence_starts_all_before_exits(tmp_path, clock):
    """一轮时序定死：提前起转 → 低档追日 → 正常起转，最后才退出。"""

    platform = build_plant(_short_round_cfg(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    platform.track(820.0, 720.0)
    report = platform.track(820.0, 745.0)
    phases = [cmd["phase"] for cmd in report["plan"]["commands"]]
    assert phases == sorted(phases, key=lambda p: 0 if p == "start" else 1)
    starts = [cmd for cmd in report["plan"]["commands"] if cmd["phase"] == "start"]
    batch = {"early_start": 0, "lag": 1, "in_time": 2}
    ordered = [batch[cmd["verdict"]] for cmd in starts]
    assert ordered == sorted(ordered)
    assert all(cmd["action"] == "defocus" for cmd in report["plan"]["commands"] if cmd["phase"] == "exit")


def test_early_start_mirrors_finish_within_round(tmp_path, clock):
    """需要跨轮提前量的镜子先吃 lead 预算，窗口末必须到位。"""

    platform = build_plant(default_config(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    first = platform.track(820.0, 720.0)  # 60 秒窗口，全部正常就位
    assert first["plan"]["counts"]["in_time"] == 123

    # 下一轮窗口收窄到 2 秒，太阳走了 20 分钟（约 5 度转角，落在提前区间）。
    platform.field._limits = dataclasses.replace(
        default_config().field,
        round_period_s=2.0,
        early_lead_max_s=30.0,
        coarse_aim_tolerance_deg=2.5,
    )
    report = platform.track(820.0, 740.0)
    assert report["plan"]["counts"]["early_start"] == 123
    early = [o for o in report["outcomes"] if o["verdict"] == "early_start"]
    assert all(o["within_fine"] for o in early)
    assert report["ok"] is True
    # 提前起转镜排在起转相位最前。
    starts = [cmd for cmd in report["plan"]["commands"] if cmd["phase"] == "start"]
    assert all(cmd["verdict"] == "early_start" for cmd in starts)


def test_exit_commands_are_dispatched_only_after_starts_finish(tmp_path, clock):
    """退出指令在起转相位走完后才下达：起转相位期间退出镜仍持有追日目标。"""

    from cspplant.heliostat.arrival import PHASE_EXIT
    from cspplant.heliostat.drive import DEFOCUS, HeliostatDrive

    platform = build_plant(_short_round_cfg(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    platform.track(820.0, 720.0)

    exit_mirror = "m042"  # b 区，来不及必退出
    drive = platform.field.drive(exit_mirror)
    exit_dispatched = {"value": False}
    defocus_steps_before_dispatch: list[str] = []

    original_step = HeliostatDrive.step

    def watched_step(self, dt):  # noqa: ANN001
        if self.name == exit_mirror and self.state == DEFOCUS and not exit_dispatched["value"]:
            defocus_steps_before_dispatch.append(self.name)
        return original_step(self, dt)

    exit_cmd_seen_at: list[int] = []
    original_defocus = drive.defocus

    def guarded_defocus():
        exit_dispatched["value"] = True
        plan = platform.field.arrival_plan()
        assert plan is not None
        start_seqs = [c["seq"] for c in plan["commands"] if c["phase"] == "start"]
        exit_seqs = [
            c["seq"] for c in plan["commands"] if c["mirror"] == exit_mirror and c["phase"] == PHASE_EXIT
        ]
        exit_cmd_seen_at.append(exit_seqs[0] - max(start_seqs))
        return original_defocus()

    drive.defocus = guarded_defocus  # type: ignore[method-assign]
    HeliostatDrive.step = watched_step  # type: ignore[assignment]
    try:
        report = platform.track(820.0, 745.0)
    finally:
        HeliostatDrive.step = original_step  # type: ignore[assignment]

    plan = report["plan"]
    exit_cmd = next(
        cmd for cmd in plan["commands"] if cmd["mirror"] == exit_mirror and cmd["phase"] == PHASE_EXIT
    )
    start_seqs = [cmd["seq"] for cmd in plan["commands"] if cmd["phase"] == "start"]
    assert exit_cmd["seq"] == max(start_seqs) + 1
    # 退出指令紧跟最后一条起转指令（中间没有夹别的退出指令）。
    assert exit_cmd_seen_at == [1]
    assert defocus_steps_before_dispatch == []  # 起转相位没有镜子在执行离焦转动


def test_readback_missing_before_plan_is_partial_failure_handled_as_lag(platform):
    """计划阶段回读缺失：按没跟上走镜区处置，其余镜子不受影响，记 alarm。"""

    platform.start()
    platform.field.drive("m001").set_readback(False)  # a 区：低档追日
    platform.field.drive("m042").set_readback(False)  # b 区：退出聚焦
    report = platform.track(820.0, 720.0)

    plan_by_name = {item["mirror"]: item for item in report["plan"]["entries"]}
    assert plan_by_name["m001"]["verdict"] == "lag"
    assert plan_by_name["m001"]["reason"] == "readback_missing"
    assert plan_by_name["m001"]["grade"] == "coarse"
    assert plan_by_name["m042"]["verdict"] == "lag"
    assert plan_by_name["m042"]["action"] == "defocus"

    outcomes = {item["mirror"]: item for item in report["outcomes"]}
    assert outcomes["m001"]["lagged"] is True
    assert outcomes["m042"]["lagged"] is True
    assert outcomes["m002"]["within_fine"] is True
    assert report["aimed"] == 121

    # 部分失败不联锁整轮：ok 仍可为假，但原因只挂在没跟上的镜子上。
    assert any("m001" in v for v in report["violations"])
    assert any("m042" in v for v in report["violations"])
    assert not any("m002" in v for v in report["violations"])

    alarms = [
        rec for rec in platform.audit.entries(500)
        if rec["action"] == "mirror_lagging" and rec["severity"] == "alarm"
    ]
    assert {rec["asset"] for rec in alarms} == {"m001", "m042"}


def test_readback_lost_mid_round_is_also_lag(tmp_path, clock):
    """指令按计划发出去后回读掉线：窗口末按没跟上，不允许算对焦成功。"""

    platform = build_plant(default_config(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    platform.track(820.0, 720.0)
    platform.field.drive("m010").set_readback(False)
    clock.advance(60)
    report = platform.track(820.0, 721.0)
    outcome = next(o for o in report["outcomes"] if o["mirror"] == "m010")
    assert outcome["readback_ok"] is False
    assert outcome["lagged"] is True
    assert any("m010" in v for v in report["violations"])


def test_readback_recovery_returns_mirror_to_normal(tmp_path, clock):
    platform = build_plant(default_config(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    platform.track(820.0, 720.0)
    platform.set_mirror_readback("m001", False)
    clock.advance(60)
    report = platform.track(820.0, 721.0)
    assert next(o for o in report["outcomes"] if o["mirror"] == "m001")["lagged"] is True
    platform.set_mirror_readback("m001", True)
    clock.advance(60)
    report = platform.track(820.0, 722.0)
    assert next(o for o in report["outcomes"] if o["mirror"] == "m001")["within_fine"] is True
    assert platform.field.status()["readback_failures"] == 1


def test_arrival_plan_is_served_to_console(tmp_path, clock):
    platform = build_plant(default_config(), clock=clock, storage_root=str(tmp_path / "s"))
    platform.start()
    assert platform.arrival_plan()["available"] is False
    platform.track(820.0, 720.0)
    payload = platform.arrival_plan()
    assert payload["available"] is True
    assert payload["plan"]["counts"]["planned"] == 123
    assert len(payload["plan"]["commands"]) == 123


def test_manual_and_excluded_mirrors_are_skipped_not_judged(platform):
    platform.start()
    platform.manual_aim("m021", 195.0, 32.0, "tester")
    report = platform.track(820.0, 720.0)
    skipped = [item for item in report["plan"]["entries"] if item["verdict"] == "skip"]
    assert [(item["mirror"], item["reason"]) for item in skipped] == [("m021", "manual")]
    assert all(item["mirror"] != "m021" for item in report["outcomes"])


def test_zone_lag_policies_are_reported_in_status(platform):
    platform.start()
    platform.track(820.0, 720.0)
    status = platform.field.status()
    assert status["lag_policies"] == {"a": "coarse", "b": "exit", "c": "exit"}
    assert status["arrival_round"] == 1


def test_coarse_tolerance_must_be_wider_than_fine_tolerance():
    from cspplant.config import config_to_mapping

    mapping = config_to_mapping(default_config())
    mapping["limits"]["field"]["coarse_aim_tolerance_deg"] = 0.1  # 窄于对焦公差 0.6
    with pytest.raises(ConfigError):
        PlatformConfig.from_mapping(mapping)


def test_lead_cap_must_be_smaller_than_round_period():
    from cspplant.config import FieldLimits

    with pytest.raises(ConfigError):
        FieldLimits.from_mapping(
            {
                "focus_threshold_w_m2": 320.0,
                "dispatch_rate_pct_s": 6.0,
                "sweep_batch": 12,
                "safe_zone_margin_m": 6.0,
                "round_period_s": 10.0,
                "early_lead_max_s": 10.0,
                "coarse_aim_tolerance_deg": 2.0,
            }
        )
