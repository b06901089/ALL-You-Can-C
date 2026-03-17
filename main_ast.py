import ast
import inspect
import textwrap
import cv2
import numpy as np
import functools


# cv2.imshow supports:
#   - 2D (H, W) — grayscale
#   - 3D (H, W, 3) — BGR color
#   - 3D (H, W, 4) — BGRA (with alpha channel)

def _show(val, name):
    if not isinstance(val, np.ndarray):
        return
    if not (val.ndim == 2 or (val.ndim == 3 and val.shape[2] in (1, 3, 4))):
        print(f"Warning: [{name}] ndim={val.ndim} shape={val.shape} not supported by imshow, skipping")
        return
    dim_info = " x ".join(str(s) for s in val.shape)
    cv2.imshow(f"[{name}]: ndim={val.ndim}, shape=({dim_info})", val)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class _InjectShow(ast.NodeTransformer):
    """After every assignment, inject a _show(target, 'name') call."""

    def _make_show_call(self, var_name):
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="_show", ctx=ast.Load()),
                args=[
                    ast.Name(id=var_name, ctx=ast.Load()),
                    ast.Constant(value=var_name),
                ],
                keywords=[],
            )
        )

    def visit_FunctionDef(self, node):
        new_body = []
        for stmt in node.body:
            new_body.append(stmt)
            if isinstance(stmt, ast.Assign):
                for target in stmt.targets:
                    if isinstance(target, ast.Name):
                        new_body.append(self._make_show_call(target.id))
            elif isinstance(stmt, ast.AugAssign):
                if isinstance(stmt.target, ast.Name):
                    new_body.append(self._make_show_call(stmt.target.id))
        node.body = new_body
        return node


def show_steps(func):
    source = inspect.getsource(func)
    source = textwrap.dedent(source)

    tree = ast.parse(source)

    # Remove the @show_steps decorator so we don't recurse infinitely
    func_def = tree.body[0]
    func_def.decorator_list = []

    # Inject _show calls after each assignment
    tree = _InjectShow().visit(func_def)
    # Wrap back in a module for compile()
    module = ast.Module(body=[tree], type_ignores=[])
    ast.fix_missing_locations(module)

    code = compile(module, filename=inspect.getfile(func), mode="exec")

    # Build a namespace with the original globals + _show
    ns = {**func.__globals__, "_show": _show}
    exec(code, ns)
    instrumented = ns[func.__name__]

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return instrumented(*args, **kwargs)

    return wrapper


@show_steps
def process_image(input_path: str, output_path: str):
    image = cv2.imread(input_path)
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imwrite(output_path, image)


if __name__ == "__main__":
    process_image("img/unsplash-cat-exmaple.jpg", "img/output.jpg")
