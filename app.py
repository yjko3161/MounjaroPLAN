"""Simple Flask web interface for the Mounjaro plan generator."""
from __future__ import annotations

from flask import Flask, jsonify, render_template_string, request

from mounjaro_plan import generate_mounjaro_report

app = Flask(__name__)


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


@app.route("/")
def index():
    return render_template_string(
        """
        <!DOCTYPE html>
        <html lang="ko">
        <head>
          <meta charset="UTF-8" />
          <title>마운자로 감량 플랜 생성기</title>
          <style>
            body { font-family: 'Pretendard', 'Noto Sans KR', sans-serif; background: #f5f6fa; margin: 0; padding: 20px; }
            h1 { text-align: center; }
            form { background: #fff; padding: 16px 20px; margin-bottom: 16px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-radius: 8px; }
            .group { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 12px; }
            label { display: block; font-weight: 600; margin-bottom: 4px; }
            input { width: 100%; padding: 8px; border: 1px solid #e5e7eb; border-radius: 6px; }
            button { margin-top: 12px; padding: 10px 16px; background: #2563eb; color: #fff; border: none; border-radius: 6px; cursor: pointer; }
            button:hover { background: #1d4ed8; }
            .note { color: #6b7280; font-size: 0.9em; margin-top: 6px; }
            #report { margin-top: 16px; }
          </style>
        </head>
        <body>
          <h1>마운자로 감량 플랜 생성기</h1>
          <form method="post" action="/generate">
            <div class="group">
              <div><label>시작 체중(kg)</label><input type="number" step="0.1" name="start_weight" required /></div>
              <div><label>현재 체중(kg)</label><input type="number" step="0.1" name="current_weight" required /></div>
              <div><label>목표 체중(kg)</label><input type="number" step="0.1" name="target_weight" required /></div>
              <div><label>2.5mg 예상 감량(4주)</label><input type="number" step="0.1" name="loss_2_5" placeholder="-4" /></div>
              <div><label>5mg 예상 감량(4주)</label><input type="number" step="0.1" name="loss_5" placeholder="-3" /></div>
              <div><label>7.5mg 예상 감량(4주)</label><input type="number" step="0.1" name="loss_7_5" placeholder="-2.5" /></div>
              <div><label>10mg 예상 감량(4주)</label><input type="number" step="0.1" name="loss_10" placeholder="-2.5" /></div>
              <div><label>골격근량(kg)</label><input type="number" step="0.1" name="skeletal_muscle" /></div>
              <div><label>체지방량(kg)</label><input type="number" step="0.1" name="fat_mass" /></div>
              <div><label>내장지방 레벨</label><input type="number" step="0.1" name="visceral_level" /></div>
              <div><label>WHR</label><input type="number" step="0.01" name="whr" /></div>
              <div><label>유지 시작 용량(mg)</label><input type="number" step="0.1" name="maintenance_start_dose" placeholder="5" /></div>
              <div><label>유지 용량(mg)</label><input type="number" step="0.1" name="maintenance_dose" placeholder="5" /></div>
              <div><label>유지 주기(주)</label><input type="number" step="1" name="maintenance_interval_weeks" placeholder="4" /></div>
              <div><label>유지 기간(개월)</label><input type="number" step="1" name="maintenance_months" placeholder="3" /></div>
            </div>
            <p class="note">필수 항목(시작/현재/목표 체중)만 입력하면 기본 가정으로 리포트를 생성합니다.</p>
            <button type="submit">리포트 생성</button>
          </form>
          <div id="report">{{ report|safe }}</div>
        </body>
        </html>
        """,
        report=None,
    )


@app.route("/generate", methods=["POST"])
def generate():
    config = _build_config()
    report = generate_mounjaro_report(config)
    return render_template_string(
        "{{ report|safe }}",
        report=report["html"],
    )


@app.route("/api/report", methods=["POST"])
def api_report():
    config = request.get_json(force=True, silent=True) or {}
    report = generate_mounjaro_report(config)
    return jsonify(report)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
