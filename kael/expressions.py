"""Safe parsing and evaluation of symbolic conductivity expressions."""
import ast
import numpy as np

VARIABLES = ("phi", "nv", "chi")
FUNCTIONS = {"exp": np.exp, "abs": np.abs, "log": np.log, "sqrt": np.sqrt}
BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply, ast.Div: np.divide, ast.Pow: np.power}


class Expression:
    """Expression in descriptors phi, nv, chi and free parameters p0, p1, ..."""

    def __init__(self, source, max_nodes=64):
        self.source = source
        self.tree = ast.parse(source, mode="eval").body
        self.nodes, params, self.variables = 0, set(), set()

        def check(node):
            self.nodes += 1
            if isinstance(node, ast.Name):
                if node.id in VARIABLES:
                    self.variables.add(node.id)
                elif node.id.startswith("p") and node.id[1:].isdigit():
                    params.add(int(node.id[1:]))
                else:
                    raise ValueError(f"Unknown symbol {node.id}")
            elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
                pass
            elif isinstance(node, ast.BinOp) and type(node.op) in BINARY:
                check(node.left), check(node.right)
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                check(node.operand)
            elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                  and node.func.id in FUNCTIONS and len(node.args) == 1 and not node.keywords):
                check(node.args[0])
            else:
                raise ValueError("Unsupported syntax")
        check(self.tree)
        if self.nodes > max_nodes:
            raise ValueError("Expression too long")
        if params != set(range(len(params))):
            raise ValueError("Parameters must be p0, p1, ... without gaps")
        if not self.variables:
            raise ValueError("Expression must use at least one descriptor")
        self.n_params = len(params)

    def __call__(self, params, data):
        env = {v: np.asarray(data[v], float) for v in self.variables}
        env.update({f"p{i}": float(v) for i, v in enumerate(params)})

        def visit(node):
            if isinstance(node, ast.Name):
                return env[node.id]
            if isinstance(node, ast.Constant):
                return float(node.value)
            if isinstance(node, ast.UnaryOp):
                value = visit(node.operand)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.Call):
                return FUNCTIONS[node.func.id](visit(node.args[0]))
            return BINARY[type(node.op)](visit(node.left), visit(node.right))
        with np.errstate(all="ignore"):
            out = visit(self.tree)
        n = len(next(iter(env.values()))) if np.ndim(next(iter(env.values()))) else 1
        return np.broadcast_to(out, (n,)).astype(float)
