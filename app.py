import os
import re
import uuid
from pathlib import Path
from flask import Flask, render_template, request, url_for, flash, redirect
from werkzeug.utils import secure_filename
from PIL import Image
import sympy as sp
from sympy import symbols, Eq, solve, simplify, sympify

from gtts import gTTS
from helpers import ocr_from_pil, explain_human_readable

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "static" / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
app.secret_key = "inklens-secret"

ALLOWED_EXT = {"png", "jpg", "jpeg"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def preprocess_expression(expr: str) -> str:
    if not expr:
        return expr
    expr = expr.replace("^", "**")
    expr = expr.replace("−", "-")
    expr = re.sub(r"\s+", "", expr)
    expr = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z)])(?=\d|[A-Za-z(])", "*", expr)
    return expr


def safe_sympify(s: str, local_dict=None):
    try:
        return sympify(s, locals=local_dict)
    except Exception:
        from sympy.parsing.sympy_parser import parse_expr
        return parse_expr(s, local_dict=local_dict)


def text_to_math_speech(text: str) -> str:
    replacements = {
        "=": " equals ",
        "→": " gives ",
        "**2": " squared ",
        "**3": " cubed ",
        "\\sqrt": " square root of ",
        "\\frac": " fraction ",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = re.sub(r"\*\*(\d+)", lambda m: f" to the power of {m.group(1)}", text)
    text = re.sub(r"\\frac{([^}]*)}{([^}]*)}", r"\1 divided by \2", text)
    text = text.replace("*", " ")
    text = text.replace("{", " ").replace("}", " ")

    return text.strip()


def build_narration(steps):
    narration_parts = []
    for s in steps:
        text_part = s["text"]
        latex_part = s["latex"]
        math_part = text_to_math_speech(latex_part)
        narration_parts.append(f"{text_part}. {math_part}.")
    
    return "\n".join(narration_parts)


def solve_equation_step_by_step(equation_str: str):
    steps = []

    if not equation_str.strip():
        return [{"text": "Invalid equation: empty input.", "latex": ""}], ""

    x = symbols('x')
    equation_str = preprocess_expression(equation_str)
    if '=' not in equation_str:
        lhs = safe_sympify(equation_str, local_dict={'x': x})
        rhs = 0
        eq = Eq(lhs, rhs)
    else:
        lhs_raw, _, rhs_raw = equation_str.partition('=')
        lhs = safe_sympify(lhs_raw, local_dict={'x': x})
        rhs = safe_sympify(rhs_raw, local_dict={'x': x})
        eq = Eq(lhs, rhs)

    poly = sp.expand(lhs - rhs)
    steps.append({"text": "Start with the equation:", "latex": sp.latex(eq)})
    steps.append({"text": "Rearrange into standard form:", "latex": sp.latex(poly) + " = 0"})

    vars = list(eq.free_symbols)

    if len(vars) == 1:
        var = vars[0]
        degree = sp.degree(poly, var)

        if degree == 2:
            factored = sp.factor(poly)
            if factored != poly:
                steps.append({"text": "Factorize the quadratic:", "latex": sp.latex(factored)})
            else:
                a, b, c = sp.Poly(poly, var).all_coeffs()
                steps.append({"text": "Quadratic formula:", "latex": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"})
                steps.append({"text": "Substitute values:", "latex": sp.latex(var) + " = \\frac{-(" + sp.latex(b) + ") \\pm \\sqrt{(" + sp.latex(b) + ")^2 - 4(" + sp.latex(a) + ")(" + sp.latex(c) + ")}}{2(" + sp.latex(a) + ")}"})
                disc = sp.simplify(b**2 - 4*a*c)
                steps.append({"text": "Simplify discriminant:", "latex": sp.latex(var) + " = \\frac{" + sp.latex(-b) + " \\pm \\sqrt{" + sp.latex(disc) + "}}{" + sp.latex(2*a) + "}"})

        sols = solve(eq, var)
        for i, sol in enumerate(sols, 1):
            steps.append({"text": f"Solution {i}:", "latex": sp.latex(sp.simplify(sol))})

    else:
        factored = sp.factor(poly)
        steps.append({"text": "Factorize the expression:", "latex": sp.latex(factored)})

        sols = sp.solve(eq, dict=True)
        for i, sol in enumerate(sols, 1):
            steps.append({"text": f"Solution {i}:", "latex": sp.latex(sol)})

    narration = build_narration(steps)
    return steps, narration


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_expr = (request.form.get("manual_expr") or "").strip()

        if not raw_expr and "image" in request.files:
            f = request.files["image"]
            if f and allowed_file(f.filename):
                fname = secure_filename(f.filename)
                fpath = UPLOAD_DIR / f"{uuid.uuid4().hex}_{fname}"
                f.save(str(fpath))
                try:
                    image = Image.open(fpath).convert("RGB")
                    raw_expr = ocr_from_pil(image)
                except Exception as e:
                    flash(f"OCR failed: {e}", "error")
            else:
                flash("Please upload a PNG/JPG image.", "error")

        if raw_expr:
            return redirect(url_for("result", expr=raw_expr))

    return render_template("index.html")


@app.route("/result")
def result():
    raw_expr = request.args.get("expr", "")
    steps, narration_text = solve_equation_step_by_step(raw_expr)

    audio_url = None
    if narration_text.strip():
        try:
            mp3_name = f"{uuid.uuid4().hex}.mp3"
            mp3_path = AUDIO_DIR / mp3_name
            gTTS(text=narration_text, lang="en").save(str(mp3_path))
            audio_url = url_for("static", filename=f"audio/{mp3_name}")
        except Exception as e:
            flash(f"TTS generation failed: {e}", "error")

    try:
        parsed_expr = safe_sympify(preprocess_expression(raw_expr.split("=")[0]), local_dict={'x': symbols('x')})
        parsed_latex = sp.latex(parsed_expr)
    except Exception:
        parsed_latex = raw_expr

    return render_template(
        "result.html",
        raw_expr=raw_expr,
        parsed_latex=parsed_latex,
        steps=steps,      
        audio_url=audio_url
    )


if __name__ == "__main__":
    app.run(debug=True)      