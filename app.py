import os
import re
import uuid
import json
import traceback
from pathlib import Path
from typing import Tuple, List, Dict

from flask import (
    Flask,
    render_template,
    request,
    url_for,
    flash,
    redirect,
    session,
    jsonify,
)
from werkzeug.utils import secure_filename
from PIL import Image
import sympy as sp
from sympy import symbols, Eq, solve, simplify, sympify
from gtts import gTTS
from pdf2image import convert_from_path
import fitz  # PyMuPDF
import logging

# Custom helpers
from helpers import ocr_from_pil
from equation_extractor import EquationExtractor

# -------------------- Paths --------------------
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
AUDIO_DIR = BASE_DIR / "static" / "audio"
UPLOAD_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- Flask Setup --------------------
app = Flask(__name__)
app.secret_key = os.environ.get("INKLENS_SECRET", "inklens-secret")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB upload limit

ALLOWED_EXT = {"png", "jpg", "jpeg", "pdf"}

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- Utility Functions --------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT


def preprocess_expression(expr: str) -> str:
    """Clean and prepare expression for sympy parsing"""
    if not expr:
        return expr
    expr = expr.replace("^", "**").replace("−", "-")
    expr = re.sub(r"\s+", "", expr)
    expr = re.sub(r"(?<=\d)(?=[A-Za-z(])", "*", expr)
    expr = re.sub(r"(?<=[A-Za-z)])(?=\d|[A-Za-z(])", "*", expr)
    return expr


def safe_sympify(s: str, local_dict=None):
    """Parse a string safely to sympy expression"""
    try:
        return sympify(s, locals=local_dict)
    except Exception:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
        )

        transforms = standard_transformations + (implicit_multiplication_application,)
        return parse_expr(s, local_dict=local_dict, transformations=transforms)


def text_to_math_speech(text: str) -> str:
    """Convert LaTeX/math string to human-readable speech"""
    if not text:
        return ""
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
    text = text.replace("*", " ").replace("{", " ").replace("}", " ")
    return text.strip()


def build_narration(steps):
    narration_parts = []
    for s in steps:
        text_part = s.get("text", "")
        latex_part = s.get("latex", "")
        narration_parts.append(f"{text_part}. {latex_part}.")
    return "\n".join(narration_parts).strip()


# -------------------- PDF OCR --------------------
def extract_text_from_pdf_direct(pdf_path: Path) -> str:
    """Extract text directly from PDF using PyMuPDF (fallback method)"""
    try:
        doc = fitz.open(str(pdf_path))
        text_all = ""
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text")
            if text:
                text_all += " " + text
        doc.close()
        return text_all.strip()
    except Exception:
        logger.exception("PDF text extraction failed")
        return ""


def ocr_from_pdf(pdf_path: Path) -> Tuple[str, List[Dict[str, str]]]:
    """Extract text and equations from PDF pages using enhanced OCR"""
    try:
        logger.info(f"Processing PDF: {pdf_path}")
        text_all = ""

        direct_text = extract_text_from_pdf_direct(pdf_path)
        if direct_text and len(direct_text.strip()) > 10:
            text_all = direct_text
        else:
            from pdf2image import convert_from_path

            try:
                images = convert_from_path(str(pdf_path), dpi=300)
            except Exception:
                logger.exception("PDF-to-image conversion failed")
                return "", []

            for i, img in enumerate(images):
                try:
                    text = ocr_from_pil(img)
                    if text:
                        text_all += " " + text
                except Exception:
                    logger.exception(f"OCR failed on page {i+1}")
                    continue

        text_all = text_all.strip()
        if not text_all:
            return "", []

        extractor = EquationExtractor()
        equations = extractor.extract_equations(text_all)
        return text_all, equations

    except Exception:
        logger.exception("Error in ocr_from_pdf")
        return "", []


# -------------------- Solver --------------------
def solve_equation_step_by_step(equation_str: str):
    steps = []
    if not equation_str or not equation_str.strip():
        return [{"text": "Invalid equation: empty input.", "latex": ""}], ""

    x = symbols("x")
    equation_str = preprocess_expression(equation_str)

    try:
        if "=" not in equation_str:
            lhs = safe_sympify(equation_str, local_dict={"x": x})
            rhs = sp.Integer(0)
        else:
            lhs_raw, _, rhs_raw = equation_str.partition("=")
            lhs = safe_sympify(lhs_raw, local_dict={"x": x})
            rhs = safe_sympify(rhs_raw, local_dict={"x": x})
        eq = Eq(lhs, rhs)
    except Exception:
        logger.exception("Failed to parse equation")
        return [{"text": "Failed to parse the equation.", "latex": equation_str}], ""

    poly = sp.expand(lhs - rhs)
    steps.append({"text": "Start with the equation", "latex": sp.latex(eq)})
    steps.append({"text": "Rearrange into standard form", "latex": sp.latex(poly) + " = 0"})

    var = list(eq.free_symbols)[0] if eq.free_symbols else x
    try:
        degree = sp.degree(poly, var)
    except Exception:
        degree = None

    if degree == 2:
        a, b, c = sp.Poly(poly, var).all_coeffs()
        steps.append({"text": "Quadratic formula", "latex": r"x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}"})
        steps.append(
            {
                "text": "Substitute values",
                "latex": f"x = \\frac{{-({sp.latex(b)}) \\pm \\sqrt{{({sp.latex(b)})^2 - 4({sp.latex(a)})({sp.latex(c)})}}}}{{2({sp.latex(a)})}}",
            }
        )
        disc = sp.simplify(b**2 - 4 * a * c)
        steps.append({"text": "Simplify discriminant", "latex": f"Discriminant = {sp.latex(disc)}"})

    try:
        sols = solve(eq, var)
        for i, sol in enumerate(sols, 1):
            steps.append({"text": f"Solution {i}", "latex": f"x = {sp.latex(sol)}"})
    except Exception:
        steps.append({"text": "Could not compute solutions", "latex": ""})

    narration = build_narration(steps)
    return steps, narration


# -------------------- Routes --------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        raw_expr = (request.form.get("manual_expr") or "").strip()
        extracted_equations = []
        pdf_text = ""

        f = request.files.get("image")
        if not raw_expr and f:
            if f and f.filename and allowed_file(f.filename):
                fname = secure_filename(f.filename)
                fpath = UPLOAD_DIR / f"{uuid.uuid4().hex}_{fname}"
                f.save(str(fpath))

                try:
                    if fname.lower().endswith("pdf"):
                        pdf_text, extracted_equations = ocr_from_pdf(fpath)
                        if extracted_equations:
                            raw_expr = extracted_equations[0].get("equation", "")
                            flash(f"Found {len(extracted_equations)} equation(s)", "success")
                        elif pdf_text:
                            raw_expr = pdf_text
                            flash("No equations detected, using extracted text", "warning")
                        else:
                            flash("No text could be extracted from PDF", "error")
                    else:
                        img = Image.open(fpath).convert("RGB")
                        raw_expr = ocr_from_pil(img)
                        if not raw_expr.strip():
                            flash("No text could be extracted from image", "warning")
                except Exception as e:
                    logger.exception("File processing failed")
                    flash(f"Processing failed: {e}", "error")
                finally:
                    try:
                        fpath.unlink()
                    except:
                        pass
            else:
                flash("Please upload a PNG/JPG/JPEG/PDF file.", "error")

        if raw_expr:
            session["extracted_equations"] = extracted_equations
            session["pdf_text"] = pdf_text
            return redirect(url_for("result", expr=raw_expr))

    return render_template("index.html")


@app.route("/result")
def result():
    raw_expr = request.args.get("expr", "")
    extracted_equations = session.pop("extracted_equations", []) or []
    pdf_text = session.pop("pdf_text", "") or ""

    steps, narration_text = solve_equation_step_by_step(raw_expr)

    audio_url = None
    if narration_text.strip():
        try:
            mp3_name = f"{uuid.uuid4().hex}.mp3"
            mp3_path = AUDIO_DIR / mp3_name
            gTTS(text=narration_text, lang="en").save(str(mp3_path))
            audio_url = url_for("static", filename=f"audio/{mp3_name}")
        except Exception:
            flash("TTS generation failed", "warning")

    return render_template(
        "result.html",
        raw_expr=raw_expr,
        steps=steps,
        audio_url=audio_url,
        extracted_equations=extracted_equations,
        pdf_text=pdf_text,
    )


@app.route("/pdf_guide")
def pdf_guide():
    return render_template("pdf_guide.html")


# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
