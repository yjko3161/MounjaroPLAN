"""Simple Flask web interface for the Mounjaro plan generator."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from flask import Flask, jsonify, render_template_string, request

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


def _build_config() -> dict:
    return {
        "start_weight": _form_float("start_weight"),
        "current_weight": _form_float("current_weight"),
        "target_weight": _form_float("target_weight"),
        "loss_2_5": _form_float("loss_2_5"),
        "loss_5": _form_float("loss_5"),
        "loss_7_5": _form_float("loss_7_5"),
        "loss_10": _form_float("loss_10"),
        "skeletal_muscle": _form_float("skeletal_muscle"),
        "fat_mass": _form_float("fat_mass"),
        "visceral_level": _form_float("visceral_level"),
        "whr": _form_float("whr"),
        "maintenance_start_dose": _form_float("maintenance_start_dose"),
        "maintenance_dose": _form_float("maintenance_dose"),
        "maintenance_interval_weeks": _form_float("maintenance_interval_weeks"),
        "maintenance_months": _form_float("maintenance_months"),
    }


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang=\"ko\">
<head>
  <meta charset=\"UTF-8\" />
  <title>마운자로 감량 플랜 생성기</title>
  <style>
    :root {
      --bg: #f5f6fa;
      --card: #fff;
      --accent: #2563eb;
      --muted: #6b7280;
      --border: #e5e7eb;
    }
    * { box-sizing: border-box; }
    body { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; background: var(--bg); margin: 0; padding: 20px; color: #111827; }
    h1 { text-align: center; margin-bottom: 12px; }
    .layout { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); }
    .card { background: var(--card); padding: 16px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 10px; }
    form .group { display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 12px; }
    label { display: block; font-weight: 600; margin-bottom: 4px; }
    input { width: 100%; padding: 8px; border: 1px solid var(--border); border-radius: 6px; }
    button { margin-top: 12px; padding: 10px 16px; background: var(--accent); color: #fff; border: none; border-radius: 8px; cursor: pointer; font-weight: 700; }
    button:hover { background: #1d4ed8; }
    .note { color: var(--muted); font-size: 0.9em; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; margin-top: 8px; }
    th, td { border: 1px solid var(--border); padding: 8px; text-align: center; }
    thead { background: #f3f4f6; }
    tbody tr:nth-child(even) { background: #fafafa; }
    .muted { color: var(--muted); }
    .history-report { margin-top: 8px; background: #f9fafb; padding: 10px; border-radius: 6px; }
  </style>
</head>
<body>
  <h1>마운자로 감량 플랜 생성기</h1>
  <div class=\"layout\">
    <section class=\"card\">
      <h2>계산기</h2>
      <form method=\"post\" action=\"/generate\">
        <div class=\"group\">
          <div><label>사용자 이름</label><input type=\"text\" name=\"username\" value=\"{{ username }}\" placeholder=\"보고서를 저장할 이름\" /></div>
          <div><label>시작 체중(kg)</label><input type=\"number\" step=\"0.1\" name=\"start_weight\" value=\"{{ form_data.get('start_weight','') }}\" required /></div>
          <div><label>현재 체중(kg)</label><input type=\"number\" step=\"0.1\" name=\"current_weight\" value=\"{{ form_data.get('current_weight','') }}\" required /></div>
          <div><label>목표 체중(kg)</label><input type=\"number\" step=\"0.1\" name=\"target_weight\" value=\"{{ form_data.get('target_weight','') }}\" required /></div>
          <div><label>2.5mg 예상 감량(4주)</label><input type=\"number\" step=\"0.1\" name=\"loss_2_5\" value=\"{{ form_data.get('loss_2_5','') }}\" placeholder=\"-4\" /></div>
          <div><label>5mg 예상 감량(4주)</label><input type=\"number\" step=\"0.1\" name=\"loss_5\" value=\"{{ form_data.get('loss_5','') }}\" placeholder=\"-3\" /></div>
          <div><label>7.5mg 예상 감량(4주)</label><input type=\"number\" step=\"0.1\" name=\"loss_7_5\" value=\"{{ form_data.get('loss_7_5','') }}\" placeholder=\"-2.5\" /></div>
          <div><label>10mg 예상 감량(4주)</label><input type=\"number\" step=\"0.1\" name=\"loss_10\" value=\"{{ form_data.get('loss_10','') }}\" placeholder=\"-2.5\" /></div>
          <div><label>골격근량(kg)</label><input type=\"number\" step=\"0.1\" name=\"skeletal_muscle\" value=\"{{ form_data.get('skeletal_muscle','') }}\" /></div>
          <div><label>체지방량(kg)</label><input type=\"number\" step=\"0.1\" name=\"fat_mass\" value=\"{{ form_data.get('fat_mass','') }}\" /></div>
          <div><label>내장지방 레벨</label><input type=\"number\" step=\"0.1\" name=\"visceral_level\" value=\"{{ form_data.get('visceral_level','') }}\" /></div>
          <div><label>WHR</label><input type=\"number\" step=\"0.01\" name=\"whr\" value=\"{{ form_data.get('whr','') }}\" /></div>
          <div><label>유지 시작 용량(mg)</label><input type=\"number\" step=\"0.1\" name=\"maintenance_start_dose\" value=\"{{ form_data.get('maintenance_start_dose','') }}\" placeholder=\"5\" /></div>
          <div><label>유지 용량(mg)</label><input type=\"number\" step=\"0.1\" name=\"maintenance_dose\" value=\"{{ form_data.get('maintenance_dose','') }}\" placeholder=\"5\" /></div>
          <div><label>유지 주기(주)</label><input type=\"number\" step=\"1\" name=\"maintenance_interval_weeks\" value=\"{{ form_data.get('maintenance_interval_weeks','') }}\" placeholder=\"4\" /></div>
          <div><label>유지 기간(개월)</label><input type=\"number\" step=\"1\" name=\"maintenance_months\" value=\"{{ form_data.get('maintenance_months','') }}\" placeholder=\"3\" /></div>
        </div>
        <p class=\"note\">필수 항목(시작/현재/목표 체중)만 입력하면 기본 가정으로 리포트를 생성합니다. 사용자 이름을 입력하면 데이터가 저장되고 이력이 표시됩니다.</p>
        <button type=\"submit\">리포트 생성 및 저장</button>
      </form>
    </section>
    <section class=\"card\">
      <h2>계산 결과</h2>
      {% if report_html %}
        {{ report_html|safe }}
      {% else %}
        <p class=\"muted\">아직 생성된 리포트가 없습니다. 왼쪽 폼을 입력해 감량 플랜을 계산하세요.</p>
      {% endif %}
    </section>
  </div>
  <section class=\"card\">
    <h2>사용자별 이력</h2>
    <p class=\"note\">최근 20개의 리포트가 저장됩니다. 사용자 이름을 입력한 뒤 리포트를 생성하면 해당 이름으로 묶어 저장됩니다.</p>
    {% if username %}
      <p><strong>{{ username }}</strong>님의 저장된 리포트</p>
    {% endif %}
    {% if history %}
      <table>
        <thead><tr><th>생성 시각(UTC)</th><th>시작/현재/목표(kg)</th><th>총비용(원)</th><th>상세</th></tr></thead>
        <tbody>
          {% for item in history %}
            <tr>
              <td>{{ item.timestamp }}</td>
              <td>{{ item.start_weight }} → {{ item.current_weight }} → {{ item.target_weight }}</td>
              <td>{{ '{:,}'.format(item.total_cost or 0) }}</td>
              <td>
                <details>
                  <summary>리포트 보기</summary>
                  <div class=\"history-report\">{{ item.report_html|safe }}</div>
                </details>
              </td>
            </tr>
          {% endfor %}
        </tbody>
      </table>
    {% else %}
      <p class=\"muted\">표시할 저장 이력이 없습니다. 사용자 이름을 입력하고 리포트를 생성해 보세요.</p>
    {% endif %}
  </section>
</body>
</html>
"""


def _render_home(report_html: str | None, form_data: Dict[str, Any] | None, username: str) -> str:
    history = _get_user_history(username)
    return render_template_string(
        HTML_TEMPLATE,
        report_html=report_html,
        form_data=form_data or {},
        history=history,
        username=username,
    )


@app.route("/")
def index():
    username = _username()
    return _render_home(report_html=None, form_data={}, username=username)


@app.route("/generate", methods=["POST"])
def generate():
    config = _build_config()
    username = _username()
    report = generate_mounjaro_report(config)
    _persist_entry(username, config, report)
    return _render_home(report_html=report["html"], form_data=request.form, username=username)


@app.route("/api/report", methods=["POST"])
def api_report():
    payload = request.get_json(force=True, silent=True) or {}
    username = (payload.pop("username", "") or "").strip()
    report = generate_mounjaro_report(payload)
    _persist_entry(username, payload, report)
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
