# sreejita/narrative/formatter.py
from typing import List
from .schema import ExecutiveFinding


def executive_summary(findings: List[ExecutiveFinding]) -> List[str]:
    summary = []
    for f in findings:
        line = f"{f.title}: {f.explanation}"
        if f.impact:
            line += f" ({f.impact})"
        summary.append(line)
    return summary 
