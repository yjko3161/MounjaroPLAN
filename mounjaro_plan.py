"""Mounjaro (tirzepatide) weight loss plan generator.

This module exposes a single entry point `generate_mounjaro_report` that consumes a
configuration dictionary and returns a structured payload containing the simulated
plan, predicted body composition changes (when possible), cost summary, and a full
HTML report string.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional


# Default model parameters for body composition estimation
FAT_RATIO = 0.75  # kg fat lost per kg total loss
MUSCLE_RATIO = 0.05  # kg muscle lost per kg total loss


@dataclass
class PlanStep:
    """Represents a single 4-week dose step."""

    name: str
    dose: float
    loss: float
    price: int


@dataclass
class Config:
    """Normalized user input with defaults applied."""

    start_weight: float
    current_weight: float
    target_weight: float
    loss_2_5: Optional[float] = None
    loss_5: Optional[float] = None
    loss_7_5: Optional[float] = None
    loss_10: Optional[float] = None
    activity_level: str = "baseline"  # baseline | none | moderate | active
    period_weeks: Optional[int] = None
    weekly_dose_plan: List[float] = field(default_factory=list)
    skeletal_muscle: Optional[float] = None
    fat_mass: Optional[float] = None
    visceral_level: Optional[float] = None
    whr: Optional[float] = None
    maintenance_start_dose: float = 5.0
    maintenance_dose: float = 5.0
    maintenance_interval_weeks: int = 4
    maintenance_months: int = 3
    start_date: Optional[str] = None


@dataclass
class CostSummary:
    adaptation_cost: int
    maintenance_cost: int
    completed_adaptation_cost: int
    maintenance_pens: int
    maintenance_bundles: int

    @property
    def total_cost(self) -> int:
        return self.adaptation_cost + self.maintenance_cost

    @property
    def upcoming_adaptation_cost(self) -> int:
        return max(self.adaptation_cost - self.completed_adaptation_cost, 0)


@dataclass
class PlanRow:
    step_name: str
    weeks: int
    dose: float
    phase: str
    expected_loss: float
    expected_weight: float
    start_weight: float
    status: str


@dataclass
class PlanSummary:
    """Aggregated view of the adaptation phase."""

    achieved_loss: float
    remaining_loss: float
    total_weeks: int
    upcoming_weeks: int
    projected_weight: float
    total_weeks_with_maintenance: int


@dataclass
class BodyCompRow:
    label: str
    weight: float
    skeletal_muscle: float
    fat_mass: float
    body_fat_percent: float


class InvalidConfigError(ValueError):
    """Raised when the provided configuration is missing mandatory fields."""


def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        candidate = int(value)
        return candidate if candidate > 0 else default
    except (TypeError, ValueError):
        return default


def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_weekly_dose_plan(value: Any) -> List[float]:
    """Normalize the weekly dose plan input.

    Accepts lists of numbers or comma/space separated strings. Invalid entries
    are ignored.
    """

    if isinstance(value, list):
        result: List[float] = []
        for item in value:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return result

    if isinstance(value, str):
        chunks = [chunk.strip() for chunk in value.replace("\n", ",").split(",")]
        result: List[float] = []
        for chunk in chunks:
            if not chunk:
                continue
            try:
                result.append(float(chunk))
            except ValueError:
                continue
        return result

    return []


def apply_defaults(config: Dict[str, Any]) -> Config:
    """Normalize user input and apply defaults for optional fields."""

    if not isinstance(config, dict):
        raise InvalidConfigError("config must be a dictionary")

    try:
        start_weight = float(config["start_weight"])
        current_weight = float(config["current_weight"])
        target_weight = float(config["target_weight"])
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidConfigError("start_weight, current_weight, target_weight are required and must be numbers") from exc

    return Config(
        start_weight=start_weight,
        current_weight=current_weight,
        target_weight=target_weight,
        loss_2_5=_optional_float(config.get("loss_2_5")),
        loss_5=_optional_float(config.get("loss_5")),
        loss_7_5=_optional_float(config.get("loss_7_5")),
        loss_10=_optional_float(config.get("loss_10")),
        activity_level=str(config.get("activity_level") or "baseline"),
        period_weeks=_safe_int(config.get("period_weeks"), 0) or None,
        weekly_dose_plan=_parse_weekly_dose_plan(config.get("weekly_dose_plan")),
        skeletal_muscle=config.get("skeletal_muscle"),
        fat_mass=config.get("fat_mass"),
        visceral_level=config.get("visceral_level"),
        whr=config.get("whr"),
        maintenance_start_dose=_safe_float(config.get("maintenance_start_dose", 5.0), 5.0),
        maintenance_dose=_safe_float(config.get("maintenance_dose", 5.0), 5.0),
        maintenance_interval_weeks=_safe_int(config.get("maintenance_interval_weeks", 4), 4),
        maintenance_months=_safe_int(config.get("maintenance_months", 3), 3),
        start_date=(str(config.get("start_date") or "").strip() or None),
    )


def build_steps(cfg: Config) -> List[PlanStep]:
    """Create the ordered list of plan steps with defaults."""

    def _activity_adjusted_losses() -> Dict[str, float]:
        base = {"2.5mg": -3.0, "5mg": -2.5, "7.5mg": -3.0, "10mg": -4.0}
        adjustments = {
            "none": {"7.5mg": -2.0, "10mg": -3.0},
            "moderate": {"7.5mg": -3.5, "10mg": -4.5},
            "active": {"7.5mg": -5.0, "10mg": -7.0},
        }
        override = adjustments.get(cfg.activity_level.lower(), {})
        return {**base, **override}

    activity_defaults = _activity_adjusted_losses()

    def _loss_value(key: str, user_value: Optional[float]) -> float:
        return user_value if user_value is not None else activity_defaults[key]

    return [
        PlanStep("2.5mg", 2.5, _loss_value("2.5mg", cfg.loss_2_5), 280000),
        PlanStep("5mg", 5.0, _loss_value("5mg", cfg.loss_5), 380000),
        PlanStep("7.5mg", 7.5, _loss_value("7.5mg", cfg.loss_7_5), 549000),
        PlanStep("10mg", 10.0, _loss_value("10mg", cfg.loss_10), 549000),
    ]


PHASE_DECAY = [1.0, 0.8, 0.6]


# 감량 예측 관련 계수
DOSE_DECAY_FACTOR = 0.65
PLATEAU_THRESHOLDS = (
    (10.0, 0.5),
    (7.0, 0.7),
    (3.0, 0.85),
)


def _phase_label(index: int) -> str:
    mapping = {
        0: "초기(1~4주)",
        1: "중기(5~8주)",
        2: "후기(9~12주+)",
    }
    return mapping.get(index, "후기(9~12주+)")


def _base_loss_rate(current_dose_mg: float) -> float:
    """용량에 따른 기본 주당 감량량(kg).

    구간별 선형 보간을 사용해 2.5~5mg: 약 0.3~0.5, 7.5mg: 약 0.7,
    10mg 이상: 약 1.0kg/week로 매핑한다.
    """

    dose = max(current_dose_mg, 0)
    if dose <= 2.5:
        return 0.3
    if dose <= 5:
        # 2.5~5mg 구간: 0.3 -> 0.5
        return 0.3 + (dose - 2.5) / 2.5 * 0.2
    if dose <= 7.5:
        # 5~7.5mg 구간: 0.5 -> 0.7
        return 0.5 + (dose - 5.0) / 2.5 * 0.2
    # 7.5~10+mg 구간: 0.7 -> 1.0 (10mg 기준)
    capped = min(dose, 12.0)
    return 0.7 + (capped - 7.5) / 2.5 * 0.3


def _base_loss_for_projection(current_dose_mg: float) -> float:
    """Dose → weekly loss mapping tuned for 임의 용량 패턴.

    Uses linear interpolation between anchors so any dose value can be projected
    without forcing stepwise titration.
    """

    anchors = [
        (0.0, 0.0),
        (2.5, 0.4),
        (5.0, 0.6),
        (7.5, 0.8),
        (10.0, 1.05),
        (12.0, 1.15),
    ]

    dose = max(current_dose_mg, 0.0)
    for (low_dose, low_loss), (high_dose, high_loss) in zip(anchors, anchors[1:]):
        if dose <= high_dose:
            span = high_dose - low_dose or 1.0
            ratio = (dose - low_dose) / span
            return low_loss + ratio * (high_loss - low_loss)

    return anchors[-1][1]


def _adaptation_factor(weeks_on_same_dose: int) -> float:
    """Reduce effectiveness when the same dose is sustained."""

    if weeks_on_same_dose <= 1:
        return 1.0
    return max(0.4, 1.0 - 0.05 * (weeks_on_same_dose - 1))


def _plateau_factor(total_lost_kg: float) -> float:
    """누적 감량량에 따른 플래토 계수(감쇠)."""

    for threshold, factor in PLATEAU_THRESHOLDS:
        if total_lost_kg >= threshold:
            return factor
    return 1.0


def _projected_weekly_loss(current_dose_mg: float, prev_dose_mg: float, total_lost_kg: float) -> float:
    """용량, 감량 추세, 플래토를 반영한 주당 감량량."""

    weekly_loss = _base_loss_rate(current_dose_mg)
    if prev_dose_mg > current_dose_mg:
        weekly_loss *= DOSE_DECAY_FACTOR

    weekly_loss *= _plateau_factor(total_lost_kg)
    return max(weekly_loss, 0.0)


def _simulate_weekly_course(cfg: Config, steps: List[PlanStep]) -> List[Dict[str, Any]]:
    """주차 단위 감량을 시뮬레이션하여 시계열로 반환."""

    if cfg.target_weight >= cfg.start_weight:
        return []

    weekly_course: List[Dict[str, Any]] = []
    current_weight = cfg.start_weight
    step_index = 0

    while current_weight > cfg.target_weight:
        step = steps[step_index] if step_index < len(steps) else steps[-1]
        phase_idx = min(step_index, len(PHASE_DECAY) - 1)

        step_total_loss = abs(step.loss) * PHASE_DECAY[phase_idx]
        weekly_target_loss = step_total_loss / 4 if step_total_loss else 0.0

        if weekly_target_loss <= 0:
            break

        for _ in range(4):
            if current_weight <= cfg.target_weight:
                break

            weekly_loss = min(weekly_target_loss, current_weight - cfg.target_weight)

            if weekly_loss <= 0:
                return weekly_course

            current_weight = round(current_weight - weekly_loss, 4)
            weekly_course.append(
                {
                    "step_index": step_index,
                    "dose": step.dose,
                    "loss": weekly_loss,
                    "weight": current_weight,
                }
            )

        step_index += 1

    return weekly_course


def simulate_plan(cfg: Config, steps: List[PlanStep]) -> List[PlanRow]:
    """Simulate 감량을 목표 체중까지 이어가도록 반복합니다."""

    weekly_course = _simulate_weekly_course(cfg, steps)
    if not weekly_course:
        return []

    plan_rows: List[PlanRow] = []
    idx = 0
    start_weight = cfg.start_weight

    while idx < len(weekly_course):
        step_idx = weekly_course[idx]["step_index"]
        step = steps[step_idx] if step_idx < len(steps) else steps[-1]
        phase_idx = min(step_idx, len(PHASE_DECAY) - 1)

        group: List[Dict[str, Any]] = []
        while idx < len(weekly_course) and weekly_course[idx]["step_index"] == step_idx:
            group.append(weekly_course[idx])
            idx += 1

        expected_loss = sum(item["loss"] for item in group)
        expected_weight = group[-1]["weight"] if group else start_weight

        if cfg.current_weight <= expected_weight:
            status = "완료"
        elif cfg.current_weight < start_weight:
            status = "진행 중"
        else:
            status = "예정"

        plan_rows.append(
            PlanRow(
                step_name=step.name,
                weeks=len(group),
                dose=step.dose,
                phase=_phase_label(phase_idx),
                expected_loss=round(expected_loss, 1),
                expected_weight=round(expected_weight, 1),
                start_weight=round(start_weight, 1),
                status=status,
            )
        )

        start_weight = expected_weight

    return plan_rows


def calculate_costs(cfg: Config, plan_rows: List[PlanRow], steps: List[PlanStep]) -> CostSummary:
    """Calculate adaptation and maintenance costs including completed history."""

    price_lookup = {step.name: step.price for step in steps}
    adaptation_cost = sum(price_lookup.get(row.step_name, 0) for row in plan_rows)
    completed_adaptation_cost = sum(
        price_lookup.get(row.step_name, 0) for row in plan_rows if row.status in {"완료", "진행 중"}
    )

    total_maintenance_weeks = cfg.maintenance_months * 4
    shots_needed = ceil(total_maintenance_weeks / cfg.maintenance_interval_weeks) if total_maintenance_weeks else 0
    bundles = ceil(shots_needed / 4) if shots_needed else 0

    if cfg.maintenance_dose <= 2.5:
        price_maint = 280000
    elif cfg.maintenance_dose <= 5:
        price_maint = 380000
    else:
        price_maint = 549000

    maintenance_cost = bundles * price_maint if bundles > 0 else 0

    return CostSummary(
        adaptation_cost=adaptation_cost,
        maintenance_cost=maintenance_cost,
        completed_adaptation_cost=completed_adaptation_cost,
        maintenance_pens=shots_needed,
        maintenance_bundles=bundles,
    )


def summarize_plan(cfg: Config, plan_rows: List[PlanRow], weight_projection: Optional[List[Dict[str, Any]]] = None) -> PlanSummary:
    """Aggregate remaining 감량, 기간, 도달 예상 체중."""

    achieved_loss = max(cfg.start_weight - cfg.current_weight, 0)
    remaining_loss = max(cfg.current_weight - cfg.target_weight, 0)
    projected_weight = cfg.current_weight

    if weight_projection:
        total_weeks = len(weight_projection) - 1
        upcoming_weeks = total_weeks
        projected_weight = weight_projection[-1].get("expected_weight_kg", cfg.current_weight)
    else:
        total_weeks = sum(row.weeks for row in plan_rows)
        upcoming_weeks = sum(row.weeks for row in plan_rows if row.status in {"예정", "진행 중"})
        if plan_rows:
            projected_weight = plan_rows[-1].expected_weight

    return PlanSummary(
        achieved_loss=round(achieved_loss, 1),
        remaining_loss=round(remaining_loss, 1),
        total_weeks=total_weeks,
        upcoming_weeks=upcoming_weeks,
        projected_weight=round(projected_weight, 1),
        total_weeks_with_maintenance=total_weeks + cfg.maintenance_months * 4,
    )


def _normalized_weekly_dose_plan(cfg: Config, plan_rows: List[PlanRow], steps: Optional[List[PlanStep]]) -> List[float]:
    """Choose the best-effort weekly dose plan for projection."""

    if cfg.weekly_dose_plan:
        return list(cfg.weekly_dose_plan)

    if plan_rows:
        expanded: List[float] = []
        for row in plan_rows:
            expanded.extend([row.dose] * max(row.weeks, 0))
        if expanded:
            return expanded

    if steps:
        expanded: List[float] = []
        for step in steps:
            expanded.extend([step.dose] * 4)
        return expanded

    return []


def build_weight_projection(cfg: Config, plan_rows: List[PlanRow], steps: Optional[List[PlanStep]] = None) -> List[Dict[str, Any]]:
    """Create weekly weight projection strictly following the provided dose plan."""

    total_target_loss = max(cfg.start_weight - cfg.target_weight, 0.0)
    if total_target_loss <= 0:
        return [
            {
                "week_index": 0,
                "dose_mg": 0.0,
                "expected_loss_kg": 0.0,
                "expected_weight_kg": round(cfg.start_weight, 1),
                "label": "시작",
            }
        ]

    weekly_dose_plan = _normalized_weekly_dose_plan(cfg, plan_rows, steps)
    total_weeks = cfg.period_weeks or len(weekly_dose_plan)
    if total_weeks <= 0:
        total_weeks = len(weekly_dose_plan) or 12

    if not weekly_dose_plan:
        weekly_dose_plan = [2.5] * total_weeks
    elif len(weekly_dose_plan) < total_weeks:
        weekly_dose_plan.extend([weekly_dose_plan[-1]] * (total_weeks - len(weekly_dose_plan)))

    try:
        start_date = datetime.fromisoformat(cfg.start_date).date() if cfg.start_date else None
    except ValueError:
        start_date = None

    raw_losses: List[float] = []
    total_lost_raw = 0.0
    prev_dose = weekly_dose_plan[0]
    weeks_on_same = 0

    for idx in range(total_weeks):
        current_dose = weekly_dose_plan[idx] if idx < len(weekly_dose_plan) else prev_dose

        if idx == 0:
            weeks_on_same = 1
        elif current_dose == prev_dose:
            weeks_on_same += 1
        else:
            weeks_on_same = 1

        weekly_loss = _base_loss_for_projection(current_dose)
        weekly_loss *= _adaptation_factor(weeks_on_same)
        if idx > 0 and prev_dose > current_dose:
            weekly_loss *= DOSE_DECAY_FACTOR
        weekly_loss *= _plateau_factor(total_lost_raw)

        weekly_loss = max(weekly_loss, 0.0)
        raw_losses.append(weekly_loss)
        total_lost_raw += weekly_loss
        prev_dose = current_dose

    if total_lost_raw <= 0:
        scaled_losses = [total_target_loss / total_weeks] * total_weeks
    else:
        scale = total_target_loss / total_lost_raw
        scaled_losses = [loss * scale for loss in raw_losses]

    projection: List[Dict[str, Any]] = []
    current_weight = cfg.start_weight

    start_label = start_date.isoformat() if start_date else "시작"
    projection.append(
        {
            "week_index": 0,
            "dose_mg": weekly_dose_plan[0] if weekly_dose_plan else 0.0,
            "expected_loss_kg": 0.0,
            "expected_weight_kg": round(current_weight, 1),
            "label": start_label,
        }
    )

    for idx, weekly_loss in enumerate(scaled_losses, start=1):
        remaining_to_target = max(current_weight - cfg.target_weight, 0.0)
        applied_loss = min(weekly_loss, remaining_to_target)
        current_weight = round(current_weight - applied_loss, 3)

        if start_date:
            label_date = start_date + timedelta(weeks=idx)
            label = f"{label_date.isoformat()}"
        else:
            label = f"{idx}주차"

        projection.append(
            {
                "week_index": idx,
                "dose_mg": weekly_dose_plan[idx - 1] if idx - 1 < len(weekly_dose_plan) else weekly_dose_plan[-1],
                "expected_loss_kg": round(applied_loss, 3),
                "expected_weight_kg": round(current_weight, 3),
                "label": label,
            }
        )

    return projection


def predict_body_composition(cfg: Config, plan_rows: List[PlanRow], steps: List[PlanStep]) -> List[BodyCompRow]:
    """Estimate body composition changes if InBody inputs are available."""

    if cfg.skeletal_muscle is None or cfg.fat_mass is None:
        return []

    try:
        curr_weight = float(cfg.current_weight)
        curr_fat = float(cfg.fat_mass)
        curr_muscle = float(cfg.skeletal_muscle)
    except (TypeError, ValueError):
        return []

    start_delta = max(cfg.start_weight - curr_weight, 0)
    start_fat = max(curr_fat + start_delta * FAT_RATIO, 0)
    start_muscle = max(curr_muscle + start_delta * MUSCLE_RATIO, 0)

    rows: List[BodyCompRow] = [
        BodyCompRow(
            label="시작(추정)",
            weight=round(cfg.start_weight, 1),
            skeletal_muscle=round(start_muscle, 1),
            fat_mass=round(start_fat, 1),
            body_fat_percent=round(start_fat / cfg.start_weight * 100, 1) if cfg.start_weight > 0 else 0,
        ),
        BodyCompRow(
            label="현재",
            weight=round(curr_weight, 1),
            skeletal_muscle=round(curr_muscle, 1),
            fat_mass=round(curr_fat, 1),
            body_fat_percent=round(curr_fat / curr_weight * 100, 1) if curr_weight > 0 else 0,
        ),
    ]

    proj_weight = cfg.start_weight
    proj_fat = start_fat
    proj_muscle = start_muscle

    # Use existing plan rows first (status-aware), then extend with remaining steps
    # so that 7.5mg/10mg projections are available even if 목표 체중으로 인해
    # 본 계획이 일찍 종료되더라도 체성분 예측은 시작 체중 기준으로 이어간다.
    extended_steps: List[PlanRow] = list(plan_rows)

    projected_chain_weight = cfg.start_weight
    for row in plan_rows:
        projected_chain_weight -= row.expected_loss

    if len(plan_rows) < len(steps):
        for idx in range(len(plan_rows), len(steps)):
            step = steps[idx]
            phase_idx = min(idx, len(PHASE_DECAY) - 1)
            projected_loss = abs(step.loss) * PHASE_DECAY[phase_idx]

            start_weight = max(round(projected_chain_weight, 1), 0)
            projected_weight = max(round(projected_chain_weight - projected_loss, 1), 0)

            extended_steps.append(
                PlanRow(
                    step_name=step.name,
                    weeks=4,
                    dose=step.dose,
                    phase=_phase_label(phase_idx),
                    expected_loss=round(projected_loss, 1),
                    expected_weight=projected_weight,
                    start_weight=start_weight,
                    status="예정",
                )
            )

            projected_chain_weight -= projected_loss

    for row in extended_steps:
        weight_loss_step = row.expected_loss
        fat_loss_step = weight_loss_step * FAT_RATIO
        muscle_loss_step = weight_loss_step * MUSCLE_RATIO

        proj_weight -= weight_loss_step
        proj_fat -= fat_loss_step
        proj_muscle -= muscle_loss_step

        body_fat_percent = round(proj_fat / proj_weight * 100, 1) if proj_weight > 0 else 0

        rows.append(
            BodyCompRow(
                label=f"{row.step_name} 종료",
                weight=round(proj_weight, 1),
                skeletal_muscle=round(proj_muscle, 1),
                fat_mass=round(proj_fat, 1),
                body_fat_percent=body_fat_percent,
            )
        )

    if cfg.maintenance_months > 0 and rows:
        rows.append(
            BodyCompRow(
                label="유지 종료",
                weight=round(proj_weight, 1),
                skeletal_muscle=round(proj_muscle, 1),
                fat_mass=round(proj_fat, 1),
                body_fat_percent=round(proj_fat / proj_weight * 100, 1) if proj_weight > 0 else 0,
            )
        )

    return rows


def extend_plan_with_maintenance(cfg: Config, plan_rows: List[PlanRow]) -> List[PlanRow]:
    """Return plan rows including a 유지 구간 for display purposes."""

    if cfg.maintenance_months <= 0:
        return plan_rows

    display_rows = list(plan_rows)
    maintenance_weeks = cfg.maintenance_months * 4
    final_weight = plan_rows[-1].expected_weight if plan_rows else cfg.current_weight

    display_rows.append(
        PlanRow(
            step_name=f"유지({cfg.maintenance_interval_weeks}주)",
            weeks=maintenance_weeks,
            dose=cfg.maintenance_dose,
            phase="유지",
            expected_loss=0.0,
            expected_weight=round(final_weight, 1),
            start_weight=round(final_weight, 1),
            status="예정",
        )
    )

    return display_rows


def build_timeline(cfg: Config, plan_rows: List[PlanRow], weight_projection: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """Compute 예상 종료일 및 남은 D-DAY based on start_date."""

    if not cfg.start_date:
        return {}

    try:
        start_date = datetime.fromisoformat(cfg.start_date).date()
    except ValueError:
        return {}

    adaptation_weeks = sum(row.weeks for row in plan_rows)
    if weight_projection:
        adaptation_weeks = max(adaptation_weeks, len(weight_projection) - 1)

    maintenance_weeks = cfg.maintenance_months * 4
    total_weeks = adaptation_weeks + maintenance_weeks

    estimated_end = start_date + timedelta(weeks=total_weeks)
    today = datetime.utcnow().date()
    remaining_days = (estimated_end - today).days

    maintenance_start = start_date + timedelta(weeks=adaptation_weeks)

    if remaining_days > 0:
        d_day = f"D-{remaining_days}"
    elif remaining_days == 0:
        d_day = "D-DAY"
    else:
        d_day = f"D+{abs(remaining_days)}"

    return {
        "start_date": start_date.isoformat(),
        "maintenance_start_date": maintenance_start.isoformat(),
        "estimated_end_date": estimated_end.isoformat(),
        "remaining_days": remaining_days,
        "d_day": d_day,
        "total_weeks": total_weeks,
    }


def format_currency(value: int) -> str:
    return format(int(value), ",")


def _render_projection_table(weight_projection: List[Dict[str, Any]]) -> str:
    if not weight_projection:
        return "<p>감량 계획이 필요하지 않습니다.</p>"

    rows_html = "".join(
        f"<tr><td>{row['week_index']}</td><td>{row['dose_mg']:.1f}</td><td>{row['expected_loss_kg']:.2f}</td>"
        f"<td>{row['expected_weight_kg']:.2f}</td></tr>"
        for row in weight_projection
    )

    return f"""
    <table>
      <thead>
        <tr><th>주차</th><th>투여 용량(mg)</th><th>해당 주 감량(kg)</th><th>예상 체중(kg)</th></tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    <p class='note'>※ 위 감량 수치는 입력한 weekly_dose_plan을 그대로 사용한 단순 모델 추정치입니다.</p>
    """


def _render_cost_table(cost_summary: CostSummary) -> str:
    return f"""
    <table>
      <tbody>
        <tr><th>적응기 누적비용</th><td>{format_currency(cost_summary.completed_adaptation_cost)}원</td></tr>
        <tr><th>적응기 예정비용</th><td>{format_currency(cost_summary.upcoming_adaptation_cost)}원</td></tr>
        <tr><th>적응기 총비용</th><td>{format_currency(cost_summary.adaptation_cost)}원</td></tr>
        <tr><th>유지기 총비용</th><td>{format_currency(cost_summary.maintenance_cost)}원</td></tr>
        <tr><th>유지기 펜 소요량</th><td>{cost_summary.maintenance_pens}펜 (4펜 묶음 {cost_summary.maintenance_bundles}개 기준)</td></tr>
        <tr class='total'><th>전체 합계</th><td><strong>{format_currency(cost_summary.total_cost)}원</strong></td></tr>
      </tbody>
    </table>
    """


def _render_body_comp_table(rows: List[BodyCompRow], cfg: Config) -> str:
    if not rows:
        return ""

    rows_html = "".join(
        f"<tr><td>{r.label}</td><td>{r.weight:.1f}</td><td>{r.skeletal_muscle:.1f}</td>"
        f"<td>{r.fat_mass:.1f}</td><td>{r.body_fat_percent:.1f}%</td></tr>" for r in rows
    )

    visceral_note = ""
    if cfg.visceral_level is not None and rows:
        fat_loss_total = max(rows[0].fat_mass - rows[-1].fat_mass, 0)
        visceral_note = (
            f"체지방량이 약 {fat_loss_total:.1f}kg 감소하면 내장지방 레벨은 대략 1~2 정도 감소할 가능성이 있습니다. (추정치)"
        )

    whr_note = "체지방률 감소에 따라 WHR도 약간 감소하는 경향이 있으며, 실제 값은 인바디로 확인해야 합니다."

    return f"""
    <div class='card'>
      <h2>체성분 예측</h2>
      <table>
        <thead>
          <tr><th>지점</th><th>체중(kg)</th><th>골격근량(kg)</th><th>체지방량(kg)</th><th>체지방률(%)</th></tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
      <p class='note'>모든 체성분 값은 단순 비율 모델에 따른 추정치이며, 시작 시점 수치는 현재 값에서 비율을 역산한 가상의 값입니다. 실제 인바디 측정값과 다를 수 있습니다.</p>
      <p class='note'>{visceral_note}</p>
      <p class='note'>{whr_note}</p>
    </div>
    """


def _activity_label(level: str) -> str:
    mapping = {
        "baseline": "기본(보수적)",
        "none": "운동 거의 안 함",
        "moderate": "운동 보통",
        "active": "운동 열심히",
    }
    return mapping.get(level.lower(), level)


def generate_html(
    cfg: Config,
    plan_rows: List[PlanRow],
    cost_summary: CostSummary,
    body_comp_rows: List[BodyCompRow],
    steps: List[PlanStep],
    weight_projection: List[Dict[str, Any]],
) -> str:
    remaining_loss = max(cfg.current_weight - cfg.target_weight, 0)
    achieved_loss = max(cfg.start_weight - cfg.current_weight, 0)

    display_rows = extend_plan_with_maintenance(cfg, plan_rows)
    timeline = build_timeline(cfg, plan_rows, weight_projection)

    overview = f"""
    <div class='card'>
      <h2>개요</h2>
      <p>시작 체중: {cfg.start_weight} kg / 현재 체중: {cfg.current_weight} kg / 목표 체중: {cfg.target_weight} kg</p>
      <p>이미 감량: {achieved_loss:.1f} kg / 목표까지 남은 감량: {remaining_loss:.1f} kg</p>
      <p>1단계(초기 4주) 감량 가정: {' , '.join(f"{s.name} {s.loss:.1f}kg" for s in steps)}</p>
      <p>페이즈 약화 규칙: 2단계 x0.8, 3단계 x0.6 (초기&gt;중기&gt;후기로 완만해짐) / 활동 수준: {_activity_label(cfg.activity_level)}</p>
      <p>유지 플랜: {cfg.maintenance_start_dose}mg 이후 유지 시작, 유지 용량 {cfg.maintenance_dose}mg, {cfg.maintenance_interval_weeks}주 간격, {cfg.maintenance_months}개월</p>
      {f"<p>시작일 {timeline['start_date']} 기준 유지 종료 예상일: {timeline['estimated_end_date']} ({timeline['d_day']})</p>" if timeline else ''}
    </div>
    """

    plan_section = f"""
    <div class='card'>
      <h2>예상 감량 추이</h2>
      {_render_projection_table(weight_projection)}
    </div>
    """

    cost_section = f"""
    <div class='card'>
      <h2>비용 요약</h2>
      {_render_cost_table(cost_summary)}
      <p class='note'>입력한 현재 체중을 기준으로 이미 거친 단계는 '누적비용'에, 앞으로 필요한 단계는 '예정비용'에 반영했습니다. 유지비용을 더해 총합을 표시합니다.</p>
    </div>
    """

    body_comp_section = _render_body_comp_table(body_comp_rows, cfg)

    pattern_section = """
    <div class='card'>
      <h2>초기~중기 감량 참고치</h2>
      <p class='note'>SURMOUNT 그래프와 실사용 경험을 단순 요약한 참고 범위입니다. 이 계산기는 1→2→3단계로 갈수록 0.8, 0.6배로 자동 완화합니다.</p>
      <table>
        <thead><tr><th>기간</th><th>보편적 감량 경향</th></tr></thead>
        <tbody>
          <tr><td>1~4주</td><td>-2 ~ -4kg</td></tr>
          <tr><td>5~8주</td><td>-2 ~ -5kg</td></tr>
          <tr><td>9~12주</td><td>-3 ~ -6kg</td></tr>
          <tr><td>12주 누적</td><td>-6% ~ -12% (약 -6 ~ -12kg)</td></tr>
        </tbody>
      </table>
      <table>
        <thead><tr><th>7.5mg</th><th>운동 안 함</th><th>운동 보통</th><th>운동 열심히</th></tr></thead>
        <tbody>
          <tr><td>1단계(1~4주)</td><td>-2.0kg</td><td>-3.5kg</td><td>-5.0kg</td></tr>
          <tr><td>2단계(5~8주)</td><td>-1.5kg</td><td>-3.0kg</td><td>-4.0kg</td></tr>
          <tr><td>3단계(9~12주)</td><td>-1.0kg</td><td>-2.0kg</td><td>-3.0kg</td></tr>
          <tr><td>3개월 누적</td><td colspan='3'>현실 범위: -6 ~ -12kg</td></tr>
        </tbody>
      </table>
      <table>
        <thead><tr><th>10mg</th><th>운동 안 함</th><th>운동 보통</th><th>운동 열심히</th></tr></thead>
        <tbody>
          <tr><td>1단계(1~4주)</td><td>-3.0kg</td><td>-4.5kg</td><td>-7.0kg</td></tr>
          <tr><td>2단계(5~8주)</td><td>-2.0kg</td><td>-3.0kg</td><td>-5.0kg</td></tr>
          <tr><td>3단계(9~12주)</td><td>-1.0kg</td><td>-2.0kg</td><td>-3.0kg</td></tr>
          <tr><td>3개월 누적</td><td colspan='3'>현실 범위: -8 ~ -15kg (운동 병행 시 -12~-15kg 흔함)</td></tr>
        </tbody>
      </table>
      <p class='note'>계산기는 1단계 기본값(초기 4주 예상 감량)을 입력/활동수준으로 잡고 2단계는 x0.8, 3단계는 x0.6으로 자동 적용합니다. 별도 입력이 없어도 현실적 패턴을 기본으로 사용합니다.</p>
    </div>
    """

    disclaimer = """
    <div class='card'>
      <h2>주의/면책</h2>
      <p class='note'>※ 이 리포트는 사용자가 입력한 값과 단순 수학적 모델에 기반한 '예상치'입니다. 실제 체중 변화, 혈당·당뇨 상태, 인바디 수치는 개인에 따라 크게 달라질 수 있습니다. 마운자로(티르제파타이드)의 용량 조절, 투여 시작·유지·중단, 다른 약과의 병용 여부 등 모든 의료적 결정은 반드시 담당 의사와 상의하여 결정해야 합니다. 이 프로그램은 의료적 진단이나 처방을 제공하지 않으며, 단지 정보를 정리하고 시각화하는 도구입니다.</p>
      <p class='note'>기본 감량 가정은 SURMOUNT-1(비당뇨 10mg 평균 약 -19.5%/72주, 4주 평균 약 -1~2kg)과 SURMOUNT-2(당뇨 동반 10mg 평균 약 -13.4%/72주, 4주 평균 약 -0.8~1.5kg) 등 3상 연구의 평균치를 100kg 기준으로 환산해 보수적으로 설정했습니다. 개인별 식단·운동·질환 상태에 따라 실제 감량 속도는 크게 달라질 수 있습니다.</p>
    </div>
    """

    style = """
    <style>
      body { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; background: #f5f6fa; margin: 0; padding: 20px; }
      h1 { text-align: center; }
      .card { background: #fff; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 8px; }
      table { width: 100%; border-collapse: collapse; margin-top: 8px; }
      th, td { border: 1px solid #e5e7eb; padding: 8px; text-align: center; }
      thead { background: #f3f4f6; }
      tbody tr:nth-child(even) { background: #fafafa; }
      .note { color: #6b7280; font-size: 0.9em; margin-top: 6px; }
      .total th, .total td { font-weight: 700; background: #fff7ed; }
    </style>
    """

    body = f"""
    <body>
      <h1>마운자로 감량 플랜 리포트</h1>
      {overview}
      {plan_section}
      {cost_section}
      {body_comp_section}
      {pattern_section}
      {disclaimer}
    </body>
    """

    return f"<!DOCTYPE html><html lang='ko'><head><meta charset='UTF-8'><title>마운자로 감량 플랜 리포트</title>{style}</head>{body}</html>"


def generate_mounjaro_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """Generate the Mounjaro weight loss plan and HTML report.

    Args:
        config: Input configuration dictionary.

    Returns:
        A dictionary containing plan_table, body_comp_table, cost_summary, total_cost, and html.
    """

    cfg = apply_defaults(config)
    steps = build_steps(cfg)

    plan_rows = simulate_plan(cfg, steps)
    display_plan_rows = extend_plan_with_maintenance(cfg, plan_rows)
    cost_summary = calculate_costs(cfg, plan_rows, steps)
    weight_projection = build_weight_projection(cfg, plan_rows, steps)
    plan_summary = summarize_plan(cfg, plan_rows, weight_projection)
    body_comp_rows = predict_body_composition(cfg, plan_rows, steps)
    timeline = build_timeline(cfg, plan_rows, weight_projection)

    html = generate_html(cfg, plan_rows, cost_summary, body_comp_rows, steps, weight_projection)

    return {
        "plan_table": [row.__dict__ for row in display_plan_rows],
        "body_comp_table": [row.__dict__ for row in body_comp_rows],
        "plan_summary": plan_summary.__dict__,
        "weight_projection": weight_projection,
        "cost_summary": {
            "adaptation_cost": cost_summary.adaptation_cost,
            "maintenance_cost": cost_summary.maintenance_cost,
            "completed_adaptation_cost": cost_summary.completed_adaptation_cost,
            "upcoming_adaptation_cost": cost_summary.upcoming_adaptation_cost,
            "total_cost": cost_summary.total_cost,
            "maintenance_pens": cost_summary.maintenance_pens,
            "maintenance_bundles": cost_summary.maintenance_bundles,
        },
        "total_cost": cost_summary.total_cost,
        "maintenance_summary": {
            "weeks": cfg.maintenance_months * 4,
            "interval_weeks": cfg.maintenance_interval_weeks,
            "dose": cfg.maintenance_dose,
            "pens": cost_summary.maintenance_pens,
            "bundles": cost_summary.maintenance_bundles,
        },
        "timeline": timeline,
        "html": html,
    }


if __name__ == "__main__":
    example_config = {
        "start_weight": 95,
        "current_weight": 92,
        "target_weight": 80,
        "skeletal_muscle": 35,
        "fat_mass": 30,
    }
    report = generate_mounjaro_report(example_config)
    print(report)
