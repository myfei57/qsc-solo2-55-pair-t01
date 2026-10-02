"""一轮到位计划的执行结果（逐台）。

结果只回答三件事：计划给这台镜子定的档位/动作是什么、窗口末驱动回读在不在、
按计划公差评出来到没到位。判定与审计都拿这份结果对照同一份
:class:`cspplant.heliostat.arrival.ArrivalItem`，没有第二套口径。
"""

from __future__ import annotations

from dataclasses import dataclass

from ..heliostat.arrival import ACTION_DEFOCUS, GRADE_COARSE, GRADE_FINE, ArrivalItem
from ..heliostat.drive import HeliostatDrive


@dataclass(frozen=True)
class RoundOutcome:
    """一台镜子本轮指令执行后的窗口末状态。"""

    mirror: str
    group: str
    verdict: str
    action: str
    grade: str
    reason: str
    readback_ok: bool
    within_fine: bool
    within_coarse: bool
    defocused: bool

    @classmethod
    def from_drive(
        cls, item: ArrivalItem, drive: HeliostatDrive, *, phase_executed: bool = True
    ) -> "RoundOutcome":
        readback_ok = drive.readback_ok
        if item.action == ACTION_DEFOCUS:
            return cls(
                mirror=item.mirror,
                group=item.group,
                verdict=item.verdict,
                action=item.action,
                grade=GRADE_FINE,
                reason=item.reason,
                readback_ok=readback_ok,
                within_fine=False,
                within_coarse=False,
                defocused=readback_ok and phase_executed and drive.at_target(),
            )
        return cls(
            mirror=item.mirror,
            group=item.group,
            verdict=item.verdict,
            action=item.action,
            grade=item.grade,
            reason=item.reason,
            readback_ok=readback_ok,
            within_fine=readback_ok and drive.within(drive.limits.aim_tolerance_deg),
            within_coarse=readback_ok and drive.within(_coarse_tolerance(drive)),
            defocused=False,
        )

    @property
    def lagged(self) -> bool:
        """窗口末是否按没跟上认定（含回读缺失）。"""

        if self.grade == GRADE_COARSE:
            return not self.readback_ok or not self.within_coarse
        if self.action == ACTION_DEFOCUS:
            return not self.readback_ok or not self.defocused
        return not self.readback_ok or not self.within_fine

    def as_dict(self) -> dict:
        return {
            "mirror": self.mirror,
            "group": self.group,
            "verdict": self.verdict,
            "action": self.action,
            "grade": self.grade,
            "reason": self.reason,
            "readback_ok": self.readback_ok,
            "within_fine": self.within_fine,
            "within_coarse": self.within_coarse,
            "defocused": self.defocused,
            "lagged": self.lagged,
        }


def _coarse_tolerance(drive: HeliostatDrive) -> float:
    """低档追日时驱动上生效的宽公差（由镜场按计划下发）。"""

    return drive.tolerance_deg
