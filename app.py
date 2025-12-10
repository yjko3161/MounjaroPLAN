"""Simple Flask web interface for the Mounjaro plan generator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template, request

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

    history = _load_history()
    user_history = history.setdefault(username, [])

    summary = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "start_weight": config.get("start_weight"),
        "current_weight": config.get("current_weight"),
        "target_weight": config.get("target_weight"),
        "total_cost": report.get("total_cost"),
        "report_html": report.get("html"),
    }

    user_history.append(summary)
    # 최근 20개까지만 보관
    if len(user_history) > 20:
        user_history[:] = user_history[-20:]

    _save_history(history)


def _form_float(name: str) -> float | None:
    """Safely parse optional float values from form inputs."""

    value = request.form.get(name)
    if value in (None, ""):
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _form_int(name: str) -> int | None:
    value = request.form.get(name)
    if value in (None, ""):
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _build_config() -> dict:
    return {
        "start_weight": _form_float("start_weight"),
        "current_weight": _form_float("current_weight"),
        "target_weight": _form_float("target_weight"),
        "activity_level": request.form.get("activity_level", "baseline"),
        "loss_2_5": _form_float("loss_2_5"),
        "loss_5": _form_float("loss_5"),
        "loss_7_5": _form_float("loss_7_5"),
        "loss_10": _form_float("loss_10"),
        "period_weeks": _form_int("period_weeks"),
        "weekly_dose_plan": request.form.get("weekly_dose_plan"),
        "skeletal_muscle": _form_float("skeletal_muscle"),
        "fat_mass": _form_float("fat_mass"),
        "visceral_level": _form_float("visceral_level"),
        "whr": _form_float("whr"),
        "maintenance_start_dose": _form_float("maintenance_start_dose"),
        "maintenance_dose": _form_float("maintenance_dose"),
        "maintenance_interval_weeks": _form_float("maintenance_interval_weeks"),
        "maintenance_months": _form_float("maintenance_months"),
        "start_date": request.form.get("start_date"),
    }


def _render_home(report: Dict[str, Any] | None, form_data: Dict[str, Any] | None, username: str) -> str:
    history = _get_user_history(username)
    return render_template(
        "home.html",
        report=report,
        report_html=(report or {}).get("html"),
        form_data=form_data or {},
        history=history,
        username=username,
        active_tab="results" if report else "input",
    )


@app.route("/")
def index():
    username = _username()
    return _render_home(report=None, form_data={}, username=username)


@app.route("/generate", methods=["POST"])
def generate():
    config = _build_config()
    username = _username()
    report = generate_mounjaro_report(config)
    _persist_entry(username, config, report)
    return _render_home(report=report, form_data=request.form, username=username)


@app.route("/api/report", methods=["POST"])
def api_report():
    payload = request.get_json(force=True, silent=True) or {}
    username = (payload.pop("username", "") or "").strip()
    report = generate_mounjaro_report(payload)
    _persist_entry(username, payload, report)
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
