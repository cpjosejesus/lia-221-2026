"""Generate the Google Colab notebooks of every ec2 folder.

Each numbered folder gets up to two self-contained notebooks:
  colab_1_step_by_step.ipynb   the step files (01_*.cu, 02_*.cu, ...) in order. Each step is introduced
                               and followed by Markdown taken from the README's "Learning path" table:
                               | Step | File | What it adds | What to look at in the output |
  colab_2_full_programs.ipynb  the complete programs and the bike-sharing examples, each introduced by
                               its row of the README's files table: | `file.cu` | description |

Run after editing any .cu file or README table:  python3 ec2/make_colab.py
"""
import json
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
HEADER = (HERE / "common" / "cuda_compat.h").read_text(encoding="utf-8")
STEP_FILE = re.compile(r"^\d\d_\w+\.cu$")
LEARNING_PATH_ROW = re.compile(r"^\|\s*(\d+)\s*\|\s*\[`(\d\d_\w+\.cu)`\]\([^)]*\)\s*\|(.+)\|\s*$")
FILES_TABLE_ROW = re.compile(r"^\|\s*`(\w+\.cu)`\s*\|\s*(.+?)\s*\|\s*$")
STEPS_NOTEBOOK = "colab_1_step_by_step.ipynb"
FULL_NOTEBOOK = "colab_2_full_programs.ipynb"

HOW_THE_CELLS_WORK = """### How this notebook works

Every program comes as a pair of cells:

1. a cell that starts with `%%writefile name.cu`: running it **saves** the CUDA program as a file on the
   Colab machine (it does not run anything yet);
2. a cell that starts with `!nvcc`: it **compiles** that file with NVIDIA's compiler and **runs** it on
   the GPU. The `!` sends the line to the machine's shell.

Every program computes its result on the GPU **and** with a simple loop on the CPU, compares the two and
prints **PASS** when they agree (**FAIL** otherwise). Read the code cell before running it, then read
the output: the Markdown after each run says what to look at.

You can edit any `%%writefile` cell (change a size, a constant, a kernel) and run the pair again."""

SETUP = """## Setup: get a GPU

A new Colab notebook runs on a machine **without a GPU**. "Enabling the GPU" is not something the code
does: it means asking Colab for a machine that has an NVIDIA GPU, with the NVIDIA driver and the CUDA
toolkit (the `nvcc` compiler) already installed.

**Do this once, before running anything:** *Runtime → Change runtime type → Hardware accelerator: T4 GPU
→ Save*. Colab restarts the machine; files written before the change are lost, so run the notebook from
the top afterwards.

The next cell checks it worked. `nvidia-smi` lists the GPU (usually a *Tesla T4*) and `nvcc --version`
prints the CUDA compiler version. If you see *command not found* or *No devices were found*, the runtime
has no GPU: change it as above."""

HEADER_EXPLANATION = """## The `cuda_compat.h` header: what it is and when you need it

Every program in this course starts with `#include "cuda_compat.h"`. The header exists so that **the
same `.cu` file runs everywhere**, with or without an NVIDIA GPU:

| Where you run | Compiler | What `cuda_compat.h` does |
|---|---|---|
| Google Colab with a T4 GPU (this notebook) | `nvcc` | Almost nothing: it includes the real CUDA runtime, so the code runs on the real GPU. |
| Your own computer **with** an NVIDIA GPU | `nvcc` | Exactly the same as on Colab. |
| A laptop **without** an NVIDIA GPU (any Mac, most laptops) | `g++` or `clang++` | It **emulates** CUDA on the CPU: each block's threads are real CPU threads, `__syncthreads()` is a barrier, `__shared__` memory is shared, atomics and warp shuffles work. Results are correct; times are meaningless. |

It also gives three small helpers that the programs use with `nvcc` too:

- `launch(kernel, grid, block, args...)`: with `nvcc` it is **exactly** `kernel<<<grid, block>>>(args...)`.
  The `<<<...>>>` syntax only exists in `nvcc`, so a file that used it could not be compiled with `g++`;
  `launch` keeps one source for both.
- `CHECK(call)`: stops with the error message, file and line if a CUDA call fails (introduced in
  `01_first_kernels`, step 8).
- `cdiv(a, b)`: the ceiling division used to compute how many blocks a grid needs.

**On a computer with an NVIDIA GPU there is nothing to enable.** Install the NVIDIA driver and the CUDA
Toolkit (`nvcc`), then compile exactly as the cells below do:
`nvcc -O2 -arch=native -I../common file.cu -o file && ./file` (`-arch=native` compiles for the GPU in
that machine). You still include `cuda_compat.h`, because the programs call `launch` and `CHECK`; under
`nvcc` it does not emulate anything. If you prefer plain CUDA, write `kernel<<<grid, block>>>(args...)`
instead of `launch(...)` and the header is no longer needed for the launch.

**Without an NVIDIA GPU**, compile the same file with
`g++ -std=c++17 -O2 -pthread -x c++ -I../common file.cu -o file && ./file`.

The next cell saves the header on the Colab machine. You do not need to read it: just run it."""


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


def notebook(cells):
    return {"cells": cells,
            "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 0}


def readme_title_and_overview(readme):
    lines = readme.splitlines()
    title = lines[0].lstrip("# ").strip() if lines else ""
    paragraph = []
    for line in lines[1:]:
        text = line.strip()
        if text.startswith("#") and paragraph:
            break
        if not text:
            if paragraph:
                break
            continue
        if text.startswith(">") or text.startswith("|") or text.startswith("#"):
            continue
        paragraph.append(text)
    return title, " ".join(paragraph)


def learning_path(readme):
    rows = {}
    for line in readme.splitlines():
        match = LEARNING_PATH_ROW.match(line)
        if match:
            cells = [c.strip() for c in match.group(3).split("|")]
            adds = cells[0] if cells else ""
            look_at = cells[1] if len(cells) > 1 else ""
            rows[match.group(2)] = (int(match.group(1)), adds, look_at)
    return rows


def files_table(readme):
    rows = {}
    for line in readme.splitlines():
        match = FILES_TABLE_ROW.match(line)
        if match:
            rows[match.group(1)] = match.group(2)
    return rows


def program_cells(source):
    return [
        code(f"%%writefile {source.name}\n" + source.read_text(encoding="utf-8")),
        code(f"!nvcc -O2 -arch=native -I. {source.name} -o {source.stem} && ./{source.stem}"),
    ]


def opening(title, overview, folder, kind_text):
    intro = f"# {title}\n\n"
    if overview:
        intro += f"{overview}\n\n"
    intro += (f"{kind_text}\n\nThe full theory, the exercises and their solutions are in "
              f"`ec2/{folder}/README.md`. The slides of the session are the companion of this notebook.")
    return [markdown(intro), markdown(HOW_THE_CELLS_WORK), markdown(SETUP),
            code("!nvidia-smi\n!nvcc --version | tail -2"), markdown(HEADER_EXPLANATION),
            code("%%writefile cuda_compat.h\n" + HEADER)]


def closing(folder, next_hint):
    text = ("## What you did\n\n"
            "Every program above was checked against the CPU. Go back to any `%%writefile` cell, change "
            "something (a size, a block dimension, a constant) and run the pair again to see what changes.\n\n"
            f"The exercises of this topic, with solutions, are at the end of `ec2/{folder}/README.md`.\n\n"
            f"**Next:** {next_hint}")
    return [markdown(text)]


def steps_notebook(folder, title, overview, steps, descriptions, next_hint):
    total = len(steps)
    cells = opening(title, overview, folder.name,
                    f"This notebook goes **step by step**: {total} small programs, each adding one idea to the "
                    "previous one, from the simplest kernel up. The inputs are tiny and printed, so you can "
                    "follow the algorithm by hand.")
    for source in steps:
        number, adds, look_at = descriptions.get(source.name, (int(source.name[:2]), "", ""))
        intro = f"## Step {number} of {total}: `{source.name}`\n\n"
        if adds:
            intro += f"{adds}\n\n"
        intro += "Run the next cell to save the program, then the one after it to compile and run it."
        cells.append(markdown(intro))
        cells.extend(program_cells(source))
        if look_at:
            cells.append(markdown(f"**What to look at in the output:** {look_at}"))
    cells.extend(closing(folder.name, next_hint))
    return cells


def full_notebook(folder, title, overview, programs, descriptions, next_hint):
    cells = opening(title, overview, folder.name,
                    "This notebook runs the **complete programs**: every kernel of the chapter on large "
                    "inputs, compared side by side with timings, and the **bike-sharing examples**, which "
                    "apply the same pattern to another domain to give you ideas for your project. Do the "
                    f"step-by-step notebook (`{STEPS_NOTEBOOK}`) first. Timings are only meaningful on the GPU.")
    for source in programs:
        intro = f"## `{source.name}`\n\n"
        if source.name in descriptions:
            intro += f"{descriptions[source.name]}\n\n"
        if source.stem.startswith("bikeshare_"):
            intro += ("This is a bike-sharing example: the README section *Ideas for your project* explains it "
                      "and lists where the same pattern may appear in your project.\n\n")
        intro += "Run the next cell to save the program, then the one after it to compile and run it."
        cells.append(markdown(intro))
        cells.extend(program_cells(source))
    cells.extend(closing(folder.name, next_hint))
    return cells


def write(path, cells):
    path.write_text(json.dumps(notebook(cells), indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{path.relative_to(HERE.parent)}")


folders = sorted(p for p in HERE.iterdir() if p.is_dir() and re.match(r"^\d\d_", p.name))
for index, folder in enumerate(folders):
    readme_path = folder / "README.md"
    readme = readme_path.read_text(encoding="utf-8") if readme_path.exists() else f"# {folder.name}"
    title, overview = readme_title_and_overview(readme)
    sources = sorted(folder.glob("*.cu"))
    steps = [s for s in sources if STEP_FILE.match(s.name)]
    programs = [s for s in sources if not STEP_FILE.match(s.name)]
    following = folders[index + 1].name if index + 1 < len(folders) else None
    after_folder = (f"open `ec2/{following}/{STEPS_NOTEBOOK}`." if following
                    else "you have reached the end of the path.")

    (folder / "colab.ipynb").unlink(missing_ok=True)

    if steps:
        next_hint = (f"open `{FULL_NOTEBOOK}` in this same folder." if programs else after_folder)
        descriptions = learning_path(readme)
        write(folder / STEPS_NOTEBOOK, steps_notebook(folder, title, overview, steps, descriptions, next_hint))
        missing = [s.name for s in steps if s.name not in descriptions]
        if missing:
            print(f"  warning: no Learning path row in {folder.name}/README.md for {missing}")
        without_look = [s.name for s in steps if s.name in descriptions and not descriptions[s.name][2]]
        if without_look:
            print(f"  note: no 'What to look at' column in {folder.name}/README.md for {len(without_look)} steps")

    if programs:
        write(folder / FULL_NOTEBOOK,
              full_notebook(folder, title, overview, programs, files_table(readme), after_folder))
        undescribed = [s.name for s in programs if s.name not in files_table(readme)]
        if undescribed:
            print(f"  warning: no files-table row in {folder.name}/README.md for {undescribed}")
