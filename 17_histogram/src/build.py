"""Assemble 17_histogram/slides.html from the session-12 deck framework plus this folder's
slides (body.html) and animations (anim.js). Run: python3 17_histogram/src/build.py"""
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
.cols.wc{grid-template-columns:minmax(0,1.75fr) minmax(0,1fr)}
@media (max-width:820px){.cols.wc{grid-template-columns:1fr}}
.cfg-out table td,.cfg-out table th{padding:.25em .7em .25em 0}
.cfg-out tr.hot td{background:var(--acc-soft)}
.hl-line{background:var(--acc2-soft);font-weight:600}
"""
style = style.replace("</style>", extra_css + "</style>")

script = deck12[deck12.index("<script>\n(function () {"):]
helpers = script[:script.index("  const ANIM = {};") + len("  const ANIM = {};\n")]
nav = script[script.index("  // ------------------------------------------------------------ deck navigation"):]

body = (here / "body.html").read_text(encoding="utf-8")
anim = (here / "anim.js").read_text(encoding="utf-8")

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Parallel histogram: atomics and privatization</title>
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
