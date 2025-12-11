"""Simple Flask web interface for the Mounjaro plan generator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

# mounjaro_plan.py에서 새로 작성한 함수들 임포트
from mounjaro_plan import generate_mounjaro_report

app = Flask(__name__)

HISTORY_FILE = Path("data/history.json")

def _ensure_history_file() -> None:
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    if not HISTORY_FILE.exists():
        HISTORY_FILE.write_text("{}", encoding="utf-8")

def _load_history() -> Dict[str, List[Dict[str, Any]]]:
    _ensure_history_file()
    try:
        return json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}

def _save_history(history: Dict[str, List[Dict[str, Any]]]) -> None:
    HISTORY_FILE.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def _username() -> str:
    return (request.form.get("username") or request.args.get("username") or "").strip()

def _get_user_history(username: str) -> List[Dict[str, Any]]:
    if not username:
        return []
    history = _load_history().get(username, [])
    # 최신 항목이 위로 오도록 정렬
    return sorted(history, key=lambda item: item.get("timestamp", ""), reverse=True)

def _persist_entry(username: str, config: Dict[str, Any], report: Dict[str, Any]) -> None:
    if not username:
        return

    cost_summary = report.get("cost_summary")
    
    # 필요한 정보만 요약 저장
    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "start_weight": config.get("start_weight"),
        "current_weight": config.get("current_weight"),
        "target_weight": config.get("target_weight"),
        "total_cost": cost_summary.total_cost if cost_summary else None,
        "report_html": report.get("html"),
    }
    
    history = _load_history()
    user_history = history.setdefault(username, [])
    user_history.append(summary)
    if len(user_history) > 20:
        user_history[:] = user_history[-20:]
    _save_history(history)

def _build_config_from_form() -> dict:
    """폼 데이터를 딕셔너리로 변환 (weekly_dose_plan 처리 포함)."""
    
    # 1. weekly_dose_plan 파싱 (Hidden Input 우선)
    raw_plan = request.form.get("weekly_dose_plan", "")
    if not raw_plan:
        # 없으면 select 배열 확인
        raw_list = request.form.getlist("weekly_dose_plan[]")
        raw_plan = ",".join(raw_list)
    
    # 2. 나머지 필드
    return {
        "username": _username(),
        "start_weight": request.form.get("start_weight"),
        "current_weight": request.form.get("current_weight"),
        "target_weight": request.form.get("target_weight"),
        "period_weeks": request.form.get("period_weeks"),
        "start_date": request.form.get("start_date"),
        "auto_titration": request.form.get("auto_titration"),
        "weekly_dose_plan": raw_plan, # 문자열 상태로 넘김 (내부 파싱)
        
        "loss_2_5": request.form.get("loss_2_5"),
        "loss_5": request.form.get("loss_5"),
        "loss_7_5": request.form.get("loss_7_5"),
        "loss_10": request.form.get("loss_10"),
        "activity_level": request.form.get("activity_level"),
        
        "skeletal_muscle": request.form.get("skeletal_muscle"),
        "fat_mass": request.form.get("fat_mass"),
        "visceral_level": request.form.get("visceral_level"),
        "whr": request.form.get("whr"),
        
        "maintenance_start_dose": request.form.get("maintenance_start_dose"),
        "maintenance_dose": request.form.get("maintenance_dose"),
        "maintenance_interval_weeks": request.form.get("maintenance_interval_weeks"),
        "maintenance_months": request.form.get("maintenance_months"),
    }

def _plan_context(form_data: Dict[str, Any]) -> Dict[str, Any]:
    """템플릿 렌더링용 헬퍼 데이터."""
    raw_plan = form_data.get("weekly_dose_plan", "")
    if isinstance(raw_plan, list):
        plan_list = [str(x) for x in raw_plan]
        raw_plan = ",".join(plan_list)
    else:
        plan_list = [x.strip() for x in raw_plan.split(",") if x.strip()]
    
    # period_weeks가 없으면 리스트 길이 사용
    try:
        p_weeks = int(form_data.get("period_weeks") or 0)
    except:
        p_weeks = 0
        
    if p_weeks <= 0:
        p_weeks = len(plan_list) if plan_list else 24
        
    # 리스트 길이가 부족하면 마지막 값으로 채우기 (UI 표시용)
    if plan_list:
        last = plan_list[-1]
        if len(plan_list) < p_weeks:
            plan_list.extend([last] * (p_weeks - len(plan_list)))
    else:
        plan_list = ["5"] * p_weeks
        
    # Dose Counts (UI 프리셋용)
    counts = {}
    for d in plan_list:
        counts[d] = counts.get(d, 0) + 1

    return {
        "weekly_plan_raw": raw_plan,
        "weekly_plan_list": plan_list,
        "period_weeks": p_weeks,
        "dose_counts": counts,
        "dose_options": ["0", "2.5", "5", "7.5", "10", "12.5", "15"],
    }

@app.route("/")
def index():
    username = _username()
    # 초기 진입 시 빈 폼
    return render_template(
        "home.html",
        report=None,
        report_html="",
        form_data={},
        history=_get_user_history(username),
        username=username,
        active_tab="input",
        **_plan_context({})
    )

@app.route("/generate", methods=["POST"])
def generate():
    config = _build_config_from_form()
    username = config["username"]
    
    # 리포트 생성
    report = generate_mounjaro_report(config)
    
    # 저장
    _persist_entry(username, config, report)
    
    return render_template(
        "home.html",
        report=report,
        report_html=report.get("html"),
        form_data=config,
        history=_get_user_history(username),
        username=username,
        active_tab="results",
        **_plan_context(config)
    )

@app.route("/api/report", methods=["POST"])
def api_report():
    payload = request.get_json(force=True, silent=True) or {}
    username = payload.get("username", "")
    report = generate_mounjaro_report(payload)
    _persist_entry(username, payload, report)
    return jsonify(report)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)