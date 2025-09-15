from PIL import Image
import numpy as np
import sympy as sp
import pytesseract
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application
)

TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

def ocr_from_pil(image: Image.Image) -> str:
    """Basic OCR for math-ish text using Tesseract."""
    gray = image.convert("L")
    arr = np.array(gray)
    arr = (arr > 200) * 255  # light threshold
    bin_im = Image.fromarray(arr.astype(np.uint8))
    text = pytesseract.image_to_string(bin_im, config="--psm 6")
    return (text or "").strip()

def parse_math(expr_str: str):
    """Parse string into SymPy expression or Equality."""
    expr_str = expr_str.replace("^", "**")
    if "=" in expr_str:
        lhs, rhs = expr_str.split("=", 1)
        lhs = parse_expr(lhs, transformations=TRANSFORMS)
        rhs = parse_expr(rhs, transformations=TRANSFORMS)
        return sp.Eq(lhs, rhs)
    else:
        return parse_expr(expr_str, transformations=TRANSFORMS)

def classify_task(expr_str: str):
    s = expr_str.replace(" ", "").lower()
    if "=" in s:
        return "solve"
    if s.startswith("diff(") or "diff(" in s:
        return "derivative"
    if s.startswith("integrate(") or "integrate(" in s:
        return "integral"
    if s.startswith("simplify(") or "simplify(" in s:
        return "simplify"
    if s.startswith("expand(") or "expand(" in s:
        return "expand"
    return "auto"

def solve_math(expr_str: str):
    task = classify_task(expr_str)
    steps = []

    try:
        if task == "solve":
            eq = parse_math(expr_str)
            syms = list(eq.free_symbols) or [sp.Symbol("x")]
            sol = sp.solve(eq, syms[0], dict=True)
            steps.append(f"Recognized equation; attempt to solve for {syms[0]}.")
            steps.append("Isolate variable and apply algebraic methods.")
            return {"task": "solve", "input": eq, "result": sol, "steps": steps}

        elif task == "derivative":
            expr = parse_math(expr_str)
            res = sp.simplify(expr) 
            steps.append("Detected derivative operation via diff().")
            steps.append("Applied standard differentiation rules.")
            return {"task": "derivative", "input": expr, "result": res, "steps": steps}

        elif task == "integral":
            expr = parse_math(expr_str)
            res = sp.simplify(expr) 
            steps.append("Detected integral operation via integrate().")
            steps.append("Applied standard antiderivative rules.")
            return {"task": "integral", "input": expr, "result": res, "steps": steps}

        elif task == "simplify":
            expr = parse_math(expr_str)
            inner = expr.args[0] if hasattr(expr, "args") and expr.func.__name__ == "simplify" else expr
            simp = sp.simplify(inner)
            steps.append("Simplification requested.")
            steps.append("Used algebraic identities and cancellations.")
            return {"task": "simplify", "input": inner, "result": simp, "steps": steps}

        elif task == "expand":
            expr = parse_math(expr_str)
            inner = expr.args[0] if hasattr(expr, "args") and expr.func.__name__ == "expand" else expr
            expanded = sp.expand(inner)
            steps.append("Expansion requested.")
            steps.append("Distributed products and used binomial expansion.")
            return {"task": "expand", "input": inner, "result": expanded, "steps": steps}

        else:
            expr = parse_math(expr_str)
            simp = sp.simplify(expr)
            steps.append("No explicit op; attempted simplify/evaluate.")
            return {"task": "auto", "input": expr, "result": simp, "steps": steps}

    except Exception as e:
        return {"task": "error", "input": expr_str, "result": None, "steps": [f"Error: {e}"]}

def explain_human_readable(result_dict):
    """Human-friendly, rule-based explanation (LLM-ready hook)."""
    task = result_dict["task"]
    steps = result_dict["steps"]
    inp = result_dict["input"]
    res = result_dict["result"]

    def L(x):
        try:
            return sp.latex(x)
        except Exception:
            return str(x)

    lines = []
    if task == "solve":
        lines.append(f"We are solving the equation: {L(inp)}.")
        for s in steps:
            lines.append(f"- {s}")
        if isinstance(res, list) and res:
            lines.append("Solutions:")
            for d in res:
                items = [f"{L(k)} = {L(v)}" for k, v in d.items()]
                lines.append("- " + ", ".join(items))
        else:
            lines.append("No closed-form solution found.")
    elif task in ("derivative", "integral", "simplify", "expand", "auto"):
        lines.append(f"Input: {L(inp)}.")
        for s in steps:
            lines.append(f"- {s}")
        lines.append(f"Result: {L(res)}.")
    else:
        lines.append("There was an error interpreting or solving the input.")
        lines.extend(steps)

    return "\n".join(lines)
