"""Assemble 13_memory_locality/slides.html from the session-12 deck framework plus this folder's
slides (body.html) and animations (anim.js). Run: python3 13_memory_locality/src/build.py"""
import pathlib

here = pathlib.Path(__file__).resolve().parent
deck12 = (here.parent.parent / "12_gpu_architecture_scheduling" / "slides.html").read_text(encoding="utf-8")

style = deck12[deck12.index("<style>"):deck12.index("</style>") + len("</style>")]
extra_css = """
svg .hl{fill:var(--acc)}
svg .soft{fill:var(--acc-soft);stroke:#9db7e3}
svg .soft2{fill:var(--acc2-soft);stroke:#d9b06a}
.cfg .presets button.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.cfg-out table{font-size:.95em}
[data-anim=hazards] table td{padding:.35em .9em .35em 0}
.cfg .cap{font-size:.86em}
.cfg [data-legend] table{font-size:.9em;margin:.3em 0}
.cfg [data-legend] th{font-size:.82em}
.cfg .dctl{font-size:.75em}
.cfg-big small{font-size:.4em}
"""
style = style.replace("</style>", extra_css + "</style>")

script = deck12[deck12.index("<script>\n(function () {"):]
helpers = script[:script.index("  const ANIM = {};") + len("  const ANIM = {};\n")]
nav = script[script.index("  // ------------------------------------------------------------ deck navigation"):]

body = (here / "body.html").read_text(encoding="utf-8")
anim = (here / "anim.js").read_text(encoding="utf-8") + "\n" + (here / "anim_mem.js").read_text(encoding="utf-8")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Memory and Data Locality</title>
{style}
</head>
<body>
{body}
{helpers}
{anim}
{nav}</body>
</html>
"""
out = here.parent / "slides.html"
out.write_text(page, encoding="utf-8")
print(out, page.count('<section class="slide'), "slides")
