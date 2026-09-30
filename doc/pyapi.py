"""Renders the Python API reference (RST) from the type stubs in python/silt."""

import ast
import pathlib
import re

_SKIP_DUNDERS = {"__repr__", "__str__", "__hash__", "__init__", "__doc__"}


class StubError(RuntimeError):
    pass


def _clean(text):
    # Package-relative names read better than their import paths.
    return re.sub(r"\b(?:silt\.silt|silt|builtins)\.", "", text)


def _decorators(node):
    return {ast.unparse(d) for d in node.decorator_list}


def _signature(node, drop_first):
    args = ast.unparse(node.args)
    if drop_first:
        args = re.sub(r"^\s*(?:self|cls)\s*(?::[^,]*)?(?:,\s*|$)", "", args)
    out = f"{node.name}({args})"
    if node.returns is not None:
        out += f" -> {ast.unparse(node.returns)}"
    return _clean(out)


def _indent(text, n):
    pad = " " * n
    return "\n".join(pad + line if line else line for line in text.splitlines())


def _doc(node):
    return ast.get_docstring(node, clean=True) or ""


def _merged_docs(nodes):
    docs = []
    for node in nodes:
        doc = _doc(node)
        if doc and doc not in docs:
            docs.append(doc)
    return "\n\n".join(docs)


def _multi_signature(directive, sigs, options, doc, indent=0):
    pad = " " * indent
    lines = [f"{pad}.. py:{directive}:: {sigs[0]}"]
    lines += [f"{pad}   {s}" for s in sigs[1:]]
    lines += [f"{pad}   {opt}" for opt in options]
    lines.append("")
    if doc:
        lines.append(_indent(doc, indent + 3))
        lines.append("")
    return "\n".join(lines)


def _group_functions(body):
    groups = {}
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            groups.setdefault(node.name, []).append(node)
    return groups


def _render_function(nodes, indent=0, in_class=False):
    first = nodes[0]
    decs = _decorators(first)
    is_property = "property" in decs or any(d.endswith(".setter") for d in decs)
    if is_property:
        getter = next((n for n in nodes if "property" in _decorators(n)), first)
        opts = []
        if getter.returns is not None:
            opts.append(f":type: {_clean(ast.unparse(getter.returns))}")
        return _multi_signature("property", [first.name], opts, _merged_docs([getter]), indent)
    options = []
    if "staticmethod" in decs:
        options.append(":staticmethod:")
    if "classmethod" in decs:
        options.append(":classmethod:")
    drop = in_class and "staticmethod" not in decs
    sigs = [_signature(n, drop) for n in nodes]
    return _multi_signature("method" if in_class else "function", sigs, options, _merged_docs(nodes), indent)


def _render_class(node):
    is_enum = any("Enum" in ast.unparse(b) for b in node.bases)
    groups = _group_functions(node.body)
    init = groups.get("__init__", [])
    sigs = [_signature(n, True).replace("__init__", node.name, 1).split(" -> ")[0] for n in init] or [node.name]
    doc = _doc(node) or _merged_docs(init)
    out = [_multi_signature("class", sigs, [], doc)]
    for stmt in node.body:
        if is_enum and isinstance(stmt, ast.Assign):
            name = ast.unparse(stmt.targets[0])
            out.append(_multi_signature("attribute", [name], [f":value: {ast.unparse(stmt.value)}"], "", 3))
    for name, nodes in groups.items():
        if name in _SKIP_DUNDERS or (name.startswith("_") and not name.startswith("__")):
            continue
        out.append(_render_function(nodes, indent=3, in_class=True))
    return "\n".join(out)


def _render_data(name, node):
    opts = []
    if isinstance(node, ast.AnnAssign):
        opts.append(f":type: {_clean(ast.unparse(node.annotation))}")
        if node.value is not None:
            opts.append(f":value: {_clean(ast.unparse(node.value))}")
    return _multi_signature("data", [name], opts, "")


def _module_members(tree):
    classes, functions, data = {}, {}, {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            data[node.target.id] = node
    functions = _group_functions(tree.body)
    return classes, functions, data


def _all_names(tree):
    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", "") == "__all__":
            return list(ast.literal_eval(node.value))
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "__all__":
            return list(ast.literal_eval(node.value))
    raise StubError("__init__.pyi defines no __all__")


def public_names(package_stub):
    """Names the package stub declares public, via its __all__."""
    return [n for n in _all_names(ast.parse(pathlib.Path(package_stub).read_text(encoding="utf-8")))
            if not n.startswith("_")]


def render(package_stub, extension_stub):
    """RST for every public name; raises StubError if a name has no definition."""
    pkg = _module_members(ast.parse(pathlib.Path(package_stub).read_text(encoding="utf-8")))
    ext = _module_members(ast.parse(pathlib.Path(extension_stub).read_text(encoding="utf-8")))
    names = public_names(package_stub)

    # Package-level definitions shadow same-named extension ones.
    classes, functions, data = {}, {}, {}
    for table, layers in ((classes, (ext[0], pkg[0])), (functions, (ext[1], pkg[1])), (data, (ext[2], pkg[2]))):
        for layer in layers:
            table.update(layer)

    missing = [n for n in names if n not in classes and n not in functions and n not in data]
    if missing:
        raise StubError(
            "public names without a stub definition (stubs out of date? rebuild the "
            f"extension to regenerate python/silt/*.pyi): {', '.join(sorted(missing))}"
        )

    out = [".. py:currentmodule:: silt", ""]
    out += ["Types", "~~~~~", ""]
    out += [_render_class(classes[n]) for n in sorted(names) if n in classes]
    out += ["Constants", "~~~~~~~~~", ""]
    out += [_render_data(n, data[n]) for n in sorted(names) if n in data]
    out += ["Functions", "~~~~~~~~~", ""]
    out += [_render_function(functions[n]) for n in sorted(names) if n in functions]
    return "\n".join(out).rstrip() + "\n"


def write(package_stub, extension_stub, output):
    text = render(package_stub, extension_stub)
    output = pathlib.Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not output.exists() or output.read_text(encoding="utf-8") != text:
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import sys

    sys.stdout.write(render(sys.argv[1], sys.argv[2]))
