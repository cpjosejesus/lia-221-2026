"""Generate one Google Colab notebook per pattern folder of ec2/.

Each notebook is self-contained: it writes common/cuda_compat.h and the folder's .cu files into
the Colab machine with %%writefile cells, then compiles each one with nvcc and runs it. Upload the
notebook to Colab, choose Runtime > Change runtime type > GPU, and run all cells.

Run after editing any .cu file:  python3 ec2/make_colab.py
"""
import json
import pathlib

here = pathlib.Path(__file__).resolve().parent
header = (here / "common" / "cuda_compat.h").read_text(encoding="utf-8")
folders = ["conv", "stencil", "histogram", "reduction", "scan", "filter", "merge", "sorting"]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.splitlines(keepends=True)}


for name in folders:
    folder = here / name
    sources = sorted(folder.glob("*.cu"))
    if not sources:
        print(f"skip {name}: no .cu files")
        continue
    title = (folder / "README.md").read_text(encoding="utf-8").splitlines()[0].lstrip("# ").strip() \
        if (folder / "README.md").exists() else name
    cells = [
        md(f"# {title}\n\nRuns every program of `ec2/{name}` on a Colab GPU. "
           "First choose **Runtime → Change runtime type → GPU (T4 is enough)**, then **Runtime → Run all**.\n\n"
           f"The theory is in `ec2/{name}/README.md`. The same `.cu` files also run on a laptop without a GPU: "
           "`clang++ -std=c++17 -O2 -pthread -x c++ -I../common FILE.cu -o FILE && ./FILE`."),
        code("!nvidia-smi\n!nvcc --version | tail -2"),
        md("The compatibility header: plain CUDA under `nvcc`, a CPU emulation under `g++`/`clang++`."),
        code("%%writefile cuda_compat.h\n" + header),
    ]
    for src in sources:
        exe = src.stem
        cells.append(md(f"## `{src.name}`"))
        cells.append(code(f"%%writefile {src.name}\n" + src.read_text(encoding="utf-8")))
        cells.append(code(f"!nvcc -O2 -arch=native -I. {src.name} -o {exe} && ./{exe}"))
    nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                                         "kernelspec": {"display_name": "Python 3", "name": "python3"},
                                         "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    out = folder / "colab.ipynb"
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{out.relative_to(here.parent)}: {len(sources)} programs")
