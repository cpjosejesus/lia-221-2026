"""Assemble 16_stencil/slides.html from the session-12 deck framework plus this folder's
slides (body.html) and animations (anim.js). Run: python3 16_stencil/src/build.py"""
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
.cfg .cap{font-size:.86em}
.cols.wc{grid-template-columns:minmax(0,1.8fr) minmax(0,1fr)}
@media (max-width:820px){.cols.wc{grid-template-columns:1fr}}
pre.tight{font-size:.5em;line-height:1.4}
.formula{font-family:var(--mono);font-size:.8em;background:var(--code);padding:.45em .8em;border-radius:3px;margin:.2em 0}
.cfg .presets .plbl{font-family:var(--mono);color:var(--muted);align-self:center;min-width:5.5em}
[data-anim=res] table td,[data-anim=res] table th{padding:.3em .7em .3em 0}
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
<title>Stencil computation on the GPU</title>
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
