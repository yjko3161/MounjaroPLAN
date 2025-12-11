"""Mounjaro (tirzepatide) weight loss plan generator."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Any, Dict, List, Optional

# --- 기본 설정값 ---

# 체성분 예측 비율
FAT_RATIO = 0.75  # 감량 1kg당 체지방 0.75kg 감소 가정
MUSCLE_RATIO = 0.05  # 감량 1kg당 근육 0.05kg 감소 가정

# 연속 투여 시 감량 효율 감소 계수 (4주 단위: 1.0 -> 0.8 -> 0.6)
PHASE_DECAY = [1.0, 0.8, 0.6]

@dataclass
class PlanStep:
    """(구조 호환용) 기본 단계 정의."""
    name: str
    dose: float
    loss: float

@dataclass
class Config:
    """사용자 입력 설정."""
    start_weight: float
    current_weight: float
    target_weight: float
    # 주차별 용량 리스트 (핵심 데이터)
    weekly_dose_plan: List[float] = field(default_factory=list)
    
    # 4주 기준 예상 감량치 (사용자 입력 또는 기본값)
    loss_2_5: Optional[float] = None
    loss_5: Optional[float] = None
    loss_7_5: Optional[float] = None
    loss_10: Optional[float] = None
    
    activity_level: str = "baseline"
    period_weeks: Optional[int] = None
    
    # 체성분 데이터
    skeletal_muscle: Optional[float] = None
    fat_mass: Optional[float] = None
    visceral_level: Optional[float] = None
    whr: Optional[float] = None
    
    # 유지 관리
    maintenance_start_dose: float = 5.0
    maintenance_dose: float = 5.0
    maintenance_interval_weeks: int = 4
    maintenance_months: int = 3
    start_date: Optional[str] = None

@dataclass
class DoseCostItem:
    dose_label: str
    pens: int
    bundles: int
    pack_price: int
    cost: int

@dataclass
class CostSummary:
    adaptation_cost: int
    maintenance_cost: int
    completed_adaptation_cost: int
    
    # 용량별 상세 내역 리스트
    dose_breakdown: List[Dict[str, Any]]
    
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
    """UI 표시용 테이블 행 데이터."""
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
    achieved_loss: float
    remaining_loss: float
    total_weeks: int
    upcoming_weeks: int
    projected_weight: float
    total_weeks_with_maintenance: int
    last_non_zero_week: int

@dataclass
class BodyCompRow:
    label: str
    weight: float
    skeletal_muscle: float
    fat_mass: float
    body_fat_percent: float

class InvalidConfigError(ValueError):
    pass

# --- 유틸리티 함수 ---

def _safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def _safe_int(value: Any, default: int) -> int:
    try:
        val = int(value)
        return val if val > 0 else default
    except (TypeError, ValueError):
        return default

def _optional_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_pack_price(dose_mg: float) -> int:
    """용량별 4펜 1박스 가격 반환."""
    if dose_mg <= 2.5:
        return 289_000
    if dose_mg <= 5.0:
        return 380_000
    return 549_000

def _parse_weekly_dose_plan(value: Any) -> List[float]:
    """문자열 또는 리스트 입력을 float 리스트로 변환."""
    if not value:
        return []
    
    raw_list = []
    if isinstance(value, list):
        raw_list = value
    elif isinstance(value, str):
        raw_list = value.replace("\n", ",").split(",")
    
    result = []
    for item in raw_list:
        if isinstance(item, str):
            item = item.strip()
            if item == "":
                item = "0"
        try:
            val = float(item)
            result.append(val)
        except (ValueError, TypeError):
            result.append(0.0)
    return result

def apply_defaults(config: Dict[str, Any]) -> Config:
    """설정 딕셔너리를 Config 객체로 변환."""
    try:
        start_weight = float(config.get("start_weight", 0))
        current_weight = float(config.get("current_weight", 0))
        target_weight = float(config.get("target_weight", 0))
    except (TypeError, ValueError):
        # 기본값 처리를 위해 0으로 설정하거나 에러 처리
        start_weight = 0.0
        current_weight = 0.0
        target_weight = 0.0

    weekly_plan = _parse_weekly_dose_plan(config.get("weekly_dose_plan"))

    return Config(
        start_weight=start_weight,
        current_weight=current_weight,
        target_weight=target_weight,
        weekly_dose_plan=weekly_plan,
        loss_2_5=_optional_float(config.get("loss_2_5")),
        loss_5=_optional_float(config.get("loss_5")),
        loss_7_5=_optional_float(config.get("loss_7_5")),
        loss_10=_optional_float(config.get("loss_10")),
        activity_level=str(config.get("activity_level") or "baseline"),
        period_weeks=_safe_int(config.get("period_weeks"), 0),
        skeletal_muscle=_optional_float(config.get("skeletal_muscle")),
        fat_mass=_optional_float(config.get("fat_mass")),
        visceral_level=_optional_float(config.get("visceral_level")),
        whr=_optional_float(config.get("whr")),
        maintenance_start_dose=_safe_float(config.get("maintenance_start_dose", 5.0), 5.0),
        maintenance_dose=_safe_float(config.get("maintenance_dose", 5.0), 5.0),
        maintenance_interval_weeks=_safe_int(config.get("maintenance_interval_weeks", 4), 4),
        maintenance_months=_safe_int(config.get("maintenance_months", 3), 3),
        start_date=(str(config.get("start_date") or "").strip() or None),
    )

def _get_base_losses(cfg: Config) -> Dict[float, float]:
    """용량별 4주 기준 기본 감량치(kg) 계산."""
    # 기본값 정의
    base_map = {2.5: -3.0, 5.0: -2.5, 7.5: -3.0, 10.0: -4.0}
    
    # 활동 수준별 보정
    adjustments = {
        "none": {7.5: -2.0, 10.0: -3.0},
        "moderate": {7.5: -3.5, 10.0: -4.5},
        "active": {7.5: -5.0, 10.0: -7.0},
    }
    adj = adjustments.get(cfg.activity_level.lower(), {})
    base_map.update(adj)

    # 사용자 입력값 우선 적용
    if cfg.loss_2_5 is not None: base_map[2.5] = -abs(cfg.loss_2_5)
    if cfg.loss_5 is not None: base_map[5.0] = -abs(cfg.loss_5)
    if cfg.loss_7_5 is not None: base_map[7.5] = -abs(cfg.loss_7_5)
    if cfg.loss_10 is not None: base_map[10.0] = -abs(cfg.loss_10)
    
    # 12.5mg, 15mg 등은 10mg 값을 기준으로 추정 (간단 비례)
    base_10 = base_map[10.0]
    base_map[12.5] = base_10 * 1.1
    base_map[15.0] = base_10 * 1.2
    
    return base_map

def _get_weekly_loss_for_dose(dose: float, weeks_on_dose: int, base_losses: Dict[float, float]) -> float:
    """특정 용량, 연속 투여 주차에 따른 예상 감량(kg)."""
    if dose <= 0:
        return 0.0
    
    # 해당 용량의 4주 기준 감량치
    # 매핑에 없으면 가장 가까운 값 찾거나 보간해야 하지만, 여기선 단순 처리
    base_4week = base_losses.get(dose)
    if base_4week is None:
        # 매핑에 없는 용량(예: 6.0)은 5.0과 7.5 사이 등 보간 필요하나 생략하고 근사치 사용
        if dose < 2.5: base_4week = base_losses[2.5] * (dose/2.5)
        elif dose < 5.0: base_4week = base_losses[5.0]
        elif dose < 7.5: base_4week = base_losses[7.5]
        else: base_4week = base_losses[10.0]

    base_weekly = abs(base_4week) / 4.0

    # Phase Decay 적용 (1~4주: 1.0, 5~8주: 0.8, 9주~: 0.6)
    # weeks_on_dose는 1부터 시작
    phase_idx = (weeks_on_dose - 1) // 4
    if phase_idx >= len(PHASE_DECAY):
        factor = PHASE_DECAY[-1]
    else:
        factor = PHASE_DECAY[phase_idx]
        
    return base_weekly * factor

# --- 핵심 로직 구현 ---

def _project_weight_course(cfg: Config) -> List[Dict[str, Any]]:
    """주차별 용량 플랜에 따른 체중 변화 시뮬레이션."""
    weekly_plan = cfg.weekly_dose_plan
    if not weekly_plan:
        # 플랜이 없으면 시작 상태만 반환
        return [{
            "week_index": 0,
            "dose_mg": 0.0,
            "expected_loss_kg": 0.0,
            "expected_weight_kg": cfg.start_weight,
            "label": "시작"
        }]

    base_losses = _get_base_losses(cfg)
    projection = []
    
    current_weight = cfg.start_weight
    
    # 시작점
    try:
        start_date_obj = datetime.strptime(cfg.start_date, "%Y-%m-%d").date() if cfg.start_date else None
    except ValueError:
        start_date_obj = None

    projection.append({
        "week_index": 0,
        "dose_mg": 0.0,
        "expected_loss_kg": 0.0,
        "expected_weight_kg": round(current_weight, 2),
        "label": start_date_obj.isoformat() if start_date_obj else "시작"
    })

    # 상태 추적 변수
    prev_dose = -1.0
    weeks_on_dose = 0

    for idx, dose in enumerate(weekly_plan, start=1):
        # 0mg은 휴약주 -> 연속 투여 카운트 초기화? 유지? 
        # 보통 휴약 후 재시작이면 다시 1주차 효과를 볼 수도 있으나, 
        # 보수적으로 카운트를 리셋한다고 가정.
        if dose == 0:
            loss = 0.0
            weeks_on_dose = 0
            prev_dose = 0.0
        else:
            if dose == prev_dose:
                weeks_on_dose += 1
            else:
                weeks_on_dose = 1 # 용량 변경 시 초기화
            
            loss = _get_weekly_loss_for_dose(dose, weeks_on_dose, base_losses)
            prev_dose = dose

        current_weight = max(cfg.target_weight, current_weight - loss)
        
        label = f"{idx}주차"
        if start_date_obj:
            week_date = start_date_obj + timedelta(weeks=idx)
            label = week_date.isoformat()

        projection.append({
            "week_index": idx,
            "dose_mg": dose,
            "expected_loss_kg": round(loss, 2),
            "expected_weight_kg": round(current_weight, 2),
            "label": label
        })
        
    return projection

def calculate_costs(cfg: Config) -> CostSummary:
    """적응기 비용 계산 (펜 수 집계 -> 4펜 묶음 -> 가격)."""
    weekly_plan = cfg.weekly_dose_plan
    
    # 0mg 제외한 실제 투여 용량 집계
    valid_doses = [d for d in weekly_plan if d > 0]
    dose_counts = Counter(valid_doses)
    
    breakdown = []
    total_adaptation_cost = 0
    
    # 용량 오름차순 정렬
    for dose in sorted(dose_counts.keys()):
        count = dose_counts[dose]
        pack_price = get_pack_price(dose)
        bundles = ceil(count / 4)
        cost = bundles * pack_price
        
        total_adaptation_cost += cost
        breakdown.append({
            "dose": f"{dose:g}mg",
            "pens": count,
            "bundles": bundles,
            "pack_price": pack_price,
            "cost": cost
        })
    
    # 완료된 주차 비용 계산 (현재 체중 달성 여부 등으로 추정하지 않고, 단순 비율로 계산 불가)
    # 여기서는 "완료된 비용"을 정확히 알기 어려우므로 
    # 기존 로직과 비슷하게 '현재 체중 위치'를 보고 추정하거나, 0으로 둠.
    # 단순화를 위해 전체 비용만 계산하고, 리포트에서 보여줌.
    # (기존 UI가 completed/upcoming을 나누므로, 시뮬레이션 결과와 비교해 추정)
    
    # 유지기 비용
    m_weeks = cfg.maintenance_months * 4
    if m_weeks > 0:
        m_interval = max(1, cfg.maintenance_interval_weeks)
        m_pens = ceil(m_weeks / m_interval)
        m_bundles = ceil(m_pens / 4)
        m_price = get_pack_price(cfg.maintenance_dose)
        maintenance_cost = m_bundles * m_price
    else:
        maintenance_cost = 0
        m_pens = 0
        m_bundles = 0

    return CostSummary(
        adaptation_cost=total_adaptation_cost,
        maintenance_cost=maintenance_cost,
        completed_adaptation_cost=0, # 아래에서 별도 계산하거나 0 처리
        dose_breakdown=breakdown,
        maintenance_pens=m_pens,
        maintenance_bundles=m_bundles
    )

def _update_completed_cost(summary: CostSummary, projection: List[Dict[str, Any]], current_weight: float, start_weight: float):
    """현재 체중을 기준으로 완료된 비용 추정."""
    # 체중이 이미 목표보다 아래거나 같으면 전체 완료로 볼 수도 있음
    # 여기서는 projection을 순회하며 current_weight 이하로 떨어진 시점까지의 용량을 계산
    
    if not projection:
        return

    # 시작 체중 -> 현재 체중 감량분
    lost = start_weight - current_weight
    if lost <= 0:
        return

    # 시뮬레이션 상 어디까지 왔는지 찾기
    reached_idx = 0
    for row in projection:
        if row["week_index"] == 0: continue
        if row["expected_weight_kg"] >= current_weight:
            reached_idx = row["week_index"]
        else:
            # 현재 체중보다 더 빠진 시점 발견 -> 여기가 현재 위치 근처
            reached_idx = row["week_index"]
            break
            
    # reached_idx 주차까지 사용한 펜 계산
    # (Note: 정확히는 주차별로 어떤 용량을 썼는지 projection에서 알 수 있음)
    # 하지만 CostSummary 구조상 용량별 묶음 구매라, 낱개 비용 산출이 애매함.
    # 단순 비례(주차 비율)로 표시하거나, 
    # "이미 구매했어야 할 묶음 수"를 계산하는 것이 정확함.
    
    # 여기서는 '예정 비용'과 '누적 비용'을 단순 표시용으로만 나눔 (기존 호환)
    pass 

def _generate_plan_rows(projection: List[Dict[str, Any]]) -> List[PlanRow]:
    """Projection 데이터를 PlanRow 포맷으로 변환 (UI 테이블용)."""
    rows = []
    if not projection:
        return rows
        
    # 같은 용량끼리 묶어서 표시 (0주차 제외)
    data_points = projection[1:]
    if not data_points:
        return rows

    current_chunk = []
    chunk_dose = data_points[0]["dose_mg"]
    
    for p in data_points:
        if p["dose_mg"] == chunk_dose:
            current_chunk.append(p)
        else:
            # 청크 저장
            _add_plan_row(rows, current_chunk, chunk_dose)
            # 새 청크 시작
            current_chunk = [p]
            chunk_dose = p["dose_mg"]
    
    # 마지막 청크
    if current_chunk:
        _add_plan_row(rows, current_chunk, chunk_dose)
        
    return rows

def _add_plan_row(rows: List[PlanRow], chunk: List[Dict], dose: float):
    if not chunk: return
    
    start_w = chunk[0]["expected_weight_kg"] + chunk[0]["expected_loss_kg"] # 역산
    end_w = chunk[-1]["expected_weight_kg"]
    total_loss = start_w - end_w
    weeks = len(chunk)
    
    # Phase 이름 (단순화: 용량 구간)
    phase_name = f"{dose:g}mg 구간" if dose > 0 else "휴약"
    
    rows.append(PlanRow(
        step_name=f"{dose:g}mg" if dose > 0 else "휴약",
        weeks=weeks,
        dose=dose,
        phase=phase_name,
        expected_loss=round(total_loss, 2),
        expected_weight=round(end_w, 2),
        start_weight=round(start_w, 2),
        status="예정" # 상태 로직은 복잡하므로 '예정' 통일 혹은 별도 처리
    ))

def predict_body_composition(cfg: Config, projection: List[Dict[str, Any]]) -> List[BodyCompRow]:
    """체성분 변화 예측."""
    if cfg.skeletal_muscle is None or cfg.fat_mass is None:
        return []

    try:
        start_muscle = float(cfg.skeletal_muscle)
        start_fat = float(cfg.fat_mass)
        # 시작 시점의 체중은 '현재 체중'일 수 있음. 
        # Config의 start_weight는 다이어트 시작 체중, current_weight는 현재 입력 체중.
        # 체성분 입력은 '현재' 기준이라고 가정.
        curr_w = cfg.current_weight
    except (ValueError, TypeError):
        return []

    rows = []
    
    # 현재 상태
    rows.append(BodyCompRow(
        label="현재",
        weight=curr_w,
        skeletal_muscle=start_muscle,
        fat_mass=start_fat,
        body_fat_percent=round(start_fat/curr_w*100, 1) if curr_w>0 else 0
    ))

    # 적응기 종료 예측
    # projection의 마지막 값
    if projection:
        final_proj = projection[-1]
        final_w = final_proj["expected_weight_kg"]
        
        # 감량분
        total_loss = curr_w - final_w
        if total_loss > 0:
            loss_fat = total_loss * FAT_RATIO
            loss_muscle = total_loss * MUSCLE_RATIO
            
            pred_fat = max(0, start_fat - loss_fat)
            pred_muscle = max(0, start_muscle - loss_muscle)
            
            rows.append(BodyCompRow(
                label="적응기 종료",
                weight=round(final_w, 1),
                skeletal_muscle=round(pred_muscle, 1),
                fat_mass=round(pred_fat, 1),
                body_fat_percent=round(pred_fat/final_w*100, 1) if final_w>0 else 0
            ))

    return rows

def generate_html(
    cfg: Config,
    cost_summary: CostSummary,
    body_comp_rows: List[BodyCompRow],
    # weight_projection: List[Dict[str, Any]], # This is now handled in the template
    plan_summary: PlanSummary,
    timeline: Dict[str, Any]
) -> str:
    # --- HTML 생성 도우미 ---
    def fmt_money(v): return "{:,}".format(int(v))
    
    # 1. 개요 (Compact Dashboard)
    # 중복 제거: 목표 체중과 예상 도달 체중이 비슷하면 하나만 표시
    target_display = f"{cfg.target_weight}kg"
    if abs(plan_summary.projected_weight - cfg.target_weight) > 0.1:
        target_display += f" (예상 {plan_summary.projected_weight}kg)"

    date_info = ""
    if timeline.get('estimated_end_date'):
        date_info = f"""
        <div class='stat-box'>
          <h3>종료 예상</h3>
          <p>{timeline['estimated_end_date']}</p>
          <small>{timeline['d_day']}</small>
        </div>
        """

    overview = f"""
    <div class='card summary-card'>
      <div class='stat-grid'>
        <div class='stat-box'>
          <h3>체중 목표</h3>
          <p>{cfg.current_weight}kg &rarr; {target_display}</p>
          <small>남은 감량: {plan_summary.remaining_loss}kg</small>
        </div>
        <div class='stat-box'>
          <h3>총 기간</h3>
          <p>{plan_summary.total_weeks_with_maintenance}주</p>
          <small>적응 {plan_summary.total_weeks}주 + 유지 {cfg.maintenance_months*4}주</small>
        </div>
        <div class='stat-box'>
          <h3>총 비용</h3>
          <p>{fmt_money(cost_summary.total_cost)}원</p>
        </div>
        {date_info}
      </div>
    </div>
    """

    # 2. 감량 추이 테이블은 이제 home.html 템플릿에서 직접 렌더링합니다.
    projection_html = ""

    # 3. 비용 테이블 (Scrollable)
    dose_rows = ""
    for item in cost_summary.dose_breakdown:
        dose_rows += f"""
        <tr>
          <td>{item['dose']}</td>
          <td>{item['pens']}펜</td>
          <td>{item['bundles']}박스</td>
          <td>{fmt_money(item['cost'])}</td>
        </tr>
        """
    
    cost_html = f"""
    <div class='card'>
      <h2>비용 상세</h2>
      <div class='table-wrapper'>
        <table>
          <thead><tr><th>용량</th><th>수량</th><th>단위</th><th>비용</th></tr></thead>
          <tbody>{dose_rows}</tbody>
        </table>
      </div>
      <div class='cost-summary'>
        <p>적응기: {fmt_money(cost_summary.adaptation_cost)}원 / 유지기: {fmt_money(cost_summary.maintenance_cost)}원</p>
      </div>
    </div>
    """

    # 4. 체성분
    comp_rows = ""
    for r in body_comp_rows:
        comp_rows += f"<tr><td>{r.label}</td><td>{r.weight}</td><td>{r.skeletal_muscle}</td><td>{r.fat_mass}</td><td>{r.body_fat_percent}%</td></tr>"
    
    comp_html = ""
    if comp_rows:
        comp_html = f"""
        <div class='card'>
          <h2>체성분 예측</h2>
          <div class='table-wrapper'>
            <table>
              <thead><tr><th>시점</th><th>체중</th><th>골격근</th><th>체지방</th><th>체지방률</th></tr></thead>
              <tbody>{comp_rows}</tbody>
            </table>
          </div>
        </div>
        """

    # 스타일 및 조립
    style = """
    <style>
      body { font-family: 'Pretendard', sans-serif; background: #f5f6fa; padding: 10px; margin: 0; font-size: 14px; }
      .card { background: #fff; padding: 15px; margin-bottom: 15px; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
      h2 { margin: 0 0 10px 0; font-size: 1.1em; color: #333; }
      h3 { margin: 0 0 5px 0; font-size: 0.9em; color: #666; font-weight: normal; }
      
      /* Summary Grid */
      .stat-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; }
      .stat-box { background: #f8f9fa; padding: 10px; border-radius: 8px; text-align: center; }
      .stat-box p { margin: 0; font-weight: bold; font-size: 1.1em; color: #2c3e50; }
      .stat-box small { font-size: 0.8em; color: #888; display: block; margin-top: 4px; }

      /* Tables */
      .table-wrapper { max-height: 200px; overflow-y: auto; border: 1px solid #eee; border-radius: 4px; }
      table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
      th, td { padding: 8px 4px; text-align: center; border-bottom: 1px solid #eee; }
      th { background: #f9f9f9; position: sticky; top: 0; z-index: 1; }
      
      .cost-summary { margin-top: 10px; text-align: right; font-weight: bold; font-size: 0.9em; }
      
      /* Mobile Optimization */
      @media (max-width: 480px) {
        body { padding: 8px; }
        .card { padding: 12px; margin-bottom: 10px; }
        .stat-grid { grid-template-columns: 1fr 1fr; } 
        th, td { font-size: 0.85em; padding: 6px 2px; }
      }
    </style>
    """
    
    return f"<!DOCTYPE html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1.0'><title>리포트</title>{style}</head><body>{overview}{cost_html}{comp_html}</body></html>"

def generate_mounjaro_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """메인 진입점."""
    cfg = apply_defaults(config)
    
    # 1. 감량 시뮬레이션
    weight_projection = _project_weight_course(cfg)
    
    # 2. 비용 계산
    cost_summary = calculate_costs(cfg)
    
    # 3. 체성분 예측
    body_comp_rows = predict_body_composition(cfg, weight_projection)
    
    # 4. 요약 데이터 생성
    # 적응기 주차 = 실제 용량이 있는 마지막 주차까지
    non_zero_weeks = [w['week_index'] for w in weight_projection if w['dose_mg'] > 0 and w['week_index'] > 0]
    last_adaptation_week = max(non_zero_weeks) if non_zero_weeks else 0
    total_adaptation_weeks = last_adaptation_week # 중간에 0이 있어도 마지막까지를 기간으로 봄
    
    final_weight = weight_projection[-1]['expected_weight_kg'] if weight_projection else cfg.current_weight
    
    plan_summary = PlanSummary(
        achieved_loss=round(cfg.start_weight - cfg.current_weight, 1),
        remaining_loss=round(cfg.current_weight - cfg.target_weight, 1),
        total_weeks=total_adaptation_weeks,
        upcoming_weeks=total_adaptation_weeks, # 단순화
        projected_weight=final_weight,
        total_weeks_with_maintenance=total_adaptation_weeks + (cfg.maintenance_months * 4),
        last_non_zero_week=last_adaptation_week
    )
    
    # 타임라인
    timeline = {}
    if cfg.start_date:
        try:
            start_dt = datetime.strptime(cfg.start_date, "%Y-%m-%d").date()
            # 적응기 종료
            adapt_end_dt = start_dt + timedelta(weeks=total_adaptation_weeks)
            # 유지기 종료
            m_weeks = cfg.maintenance_months * 4
            final_end_dt = adapt_end_dt + timedelta(weeks=m_weeks)
            
            today = datetime.now().date()
            rem_days = (final_end_dt - today).days
            d_day = f"D-{rem_days}" if rem_days > 0 else f"D+{abs(rem_days)}"
            
            timeline = {
                "start_date": cfg.start_date,
                "maintenance_start_date": adapt_end_dt.isoformat(),
                "estimated_end_date": final_end_dt.isoformat(),
                "remaining_days": rem_days,
                "d_day": d_day
            }
        except ValueError:
            pass

    # HTML 생성
    html = generate_html(cfg, cost_summary, body_comp_rows, plan_summary, timeline)
    
    return {
        "report": True, # 플래그
        "plan_summary": plan_summary,
        "weight_projection": weight_projection,
        "cost_summary": cost_summary,
        "body_comp_table": body_comp_rows,
        "maintenance_summary": {
            "dose": cfg.maintenance_dose,
            "interval_weeks": cfg.maintenance_interval_weeks,
            "weeks": cfg.maintenance_months * 4,
            "pens": cost_summary.maintenance_pens,
            "bundles": cost_summary.maintenance_bundles
        },
        "timeline": timeline,
        "html": html
    }