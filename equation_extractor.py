import re

class EquationExtractor:
    """
    Simple equation extractor for Inklens AI.
    Detects math-like lines that contain '=' or arithmetic operators.
    """

    def __init__(self):
        # Regex for expressions that look like equations
        self.eq_pattern = re.compile(
            r'([A-Za-z0-9\+\-\*/\^\(\)\s]+=[A-Za-z0-9\+\-\*/\^\(\)\s]+)'
        )

    def extract_equations(self, text: str):
        """
        Return a list of dictionaries containing detected equations.
        Each entry: {'equation': str, 'confidence': float, 'line_number': int}
        """
        equations = []
        if not text:
            return equations

        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            line = line.strip()
            if not line:
                continue

            matches = self.eq_pattern.findall(line)
            for eq in matches:
                clean_eq = eq.strip()
                if not clean_eq:
                    continue
                confidence = 0.8
                if "=" in clean_eq and any(ch.isdigit() for ch in clean_eq):
                    confidence = 0.95
                equations.append({
                    "equation": clean_eq,
                    "confidence": confidence,
                    "line_number": i,
                })
        return equations
