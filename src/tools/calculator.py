"""calculator tool: safe arithmetic without eval()."""

import ast
import math
import operator

# The only operations allowed. Anything else in the expression is rejected.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

MAX_EXPRESSION_LENGTH = 200


def _evaluate(node: ast.AST) -> float:
    """Walk the parsed expression tree, allowing only numbers and the operations above."""
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        return _BINARY_OPS[type(node.op)](_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    raise ValueError(f"Not allowed in a calculation: {type(node).__name__}")


def calculate(expression: str) -> float:
    """Evaluate an arithmetic expression like '(99.91 - 98.43) * 2'.

    Raises ValueError, ZeroDivisionError, OverflowError or SyntaxError on bad input.
    """
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)")
    tree = ast.parse(expression, mode="eval")
    result = _evaluate(tree.body)
    if not isinstance(result, float) or not math.isfinite(result):
        raise ValueError("Result is not a finite real number")
    return result