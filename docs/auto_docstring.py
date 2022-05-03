#==============================================================================#
#  Author:       Dominik Müller                                                #
#  Copyright:    2022 IT-Infrastructure for Translational Medical Research,    #
#                University of Augsburg                                        #
#                                                                              #
#  This program is free software: you can redistribute it and/or modify        #
#  it under the terms of the GNU General Public License as published by        #
#  the Free Software Foundation, either version 3 of the License, or           #
#  (at your option) any later version.                                         #
#                                                                              #
#  This program is distributed in the hope that it will be useful,             #
#  but WITHOUT ANY WARRANTY; without even the implied warranty of              #
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               #
#  GNU General Public License for more details.                                #
#                                                                              #
#  You should have received a copy of the GNU General Public License           #
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.       #
#==============================================================================#
#-----------------------------------------------------#
#                   Library imports                   #
#-----------------------------------------------------#
from pathlib import Path
import mkdocs_gen_files

#-----------------------------------------------------#
#                    Configuration                    #
#-----------------------------------------------------#
src_dir = "aucmedi"

#-----------------------------------------------------#
#                       Runner                        #
#-----------------------------------------------------#
""" Generate code reference pages and navigation

    Based on the recipe of mkdocstrings:
    https://github.com/mkdocstrings/mkdocstrings

    Credits:
    Timothée Mazzucotelli
    https://github.com/pawamoy
"""
# Iterate over each Python file
for path in sorted(Path(src_dir).rglob("*.py")):
    # Get path in module, documentation and absolute
    module_path = path.relative_to(src_dir).with_suffix("")
    doc_path = path.relative_to(src_dir).with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    # Handle edge cases
    parts = (src_dir,) + tuple(module_path.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1] == "__main__":
        continue

    # Write docstring documentation to disk via parser
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}")
    # Update parser
    mkdocs_gen_files.set_edit_path(full_doc_path, path)
