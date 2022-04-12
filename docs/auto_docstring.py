"""Generate the code reference pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

for path in sorted(Path("aucmedi").rglob("*.py")):
    module_path = path.relative_to("aucmedi").with_suffix("")
    doc_path = path.relative_to("aucmedi").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = ("aucmedi",) + tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)
