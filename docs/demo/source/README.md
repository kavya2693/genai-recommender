# Video and audit report sources

Everything needed to rebuild the silent walkthrough (`docs/demo/demo.mp4`) and the audit report
(`docs/demo/AUDIT_REPORT.*`) from this repository alone. Paths below are relative to the repo root.

## What is here

| Path | What it is |
|---|---|
| `index.html` | Root of the video: theme (colours, type), and the timing of every slide |
| `compositions/bg.html`, `compositions/chrome.html` | Background grid, and the persistent project label, slide counter and progress bar |
| `compositions/s01-context.html` | Slide 01, the industry and four everyday problems |
| `compositions/s02-hook.html` | Slide 02, the project in one sentence |
| `compositions/explainer.html` | Slides 03-04, the white "how it works" chapter (edit its `CONFIG` block) |
| `compositions/s03-architecture.html` | Slide 05, the architecture diagram (`NODES` and wires `W`) |
| `compositions/s04-routes.html` | Slide 06, the BERT4Rec training curve (`HIST`) and one llama3.2 request trace |
| `compositions/value.html` | Slide 07, the outreach plan and field-team scene (data injected by `tools/inject_sheet.py`) |
| `compositions/s05-results.html`, `s06-twist.html`, `s07-end.html` | Slides 08-10, results, what didn't win, end card |
| `assets/outreach_plan.json` | One region's outreach plan, exported by `tools/export_outreach_plan.py` |
| `audit/facts.yaml` | Every figure (F-xx), assumption (A-xx), slide text and check expression used by the report |
| `tools/build_audit_report.py` | Builds the slide stills, re-checks every figure against the repo outputs, writes MD, DOCX and PDF |
| `tools/export_outreach_plan.py` | Re-runs the repo's own pipeline (seed 7) and plans one touch per prescriber in one region |
| `tools/inject_sheet.py` | Inlines that JSON into `compositions/value.html` (renders cannot fetch files) |
| `package.json`, `hyperframes.json`, `meta.json` | HyperFrames project files (CLI pinned to 0.8.156) |

Fonts (Inter, JetBrains Mono, Montserrat) are referenced by name and embedded by the HyperFrames
renderer; all three are SIL Open Font License. The repo has no plot images, so slide 06 draws its
curve from `outputs/bert4rec_training.json` (checked by the report, figure F-33).

## Requirements

- Node 22 and `npx` (HyperFrames 0.8.156 is fetched by `npx`), plus Chrome/Chromium for rendering.
- `ffmpeg` for compressing the video and extracting slide stills.
- LibreOffice (`soffice`) to convert the report to PDF.
- The repo's own environment (`uv sync`) for `make demo` and `export_outreach_plan.py`; `make demo`
  also needs a local Ollama server with `ollama pull llama3.2`.
- For `build_audit_report.py` and `inject_sheet.py` (any Python 3.11+, not added to the repo's
  `pyproject.toml`): `pip install pyyaml pandas python-docx pillow numpy matplotlib`.

## Rebuild the numbers, the video and the report

```bash
# 1. repo outputs (about 4 minutes on 4 CPU threads, including three llama3.2 drafts)
make demo

# 2. inputs for the video (the export retrains every model with the same seeds, about 6 minutes)
PYTHONPATH=src .venv/bin/python docs/demo/source/tools/export_outreach_plan.py \
  docs/demo/source/assets/outreach_plan.json --region north
python3 docs/demo/source/tools/inject_sheet.py docs/demo/source/assets/outreach_plan.json \
  docs/demo/source/compositions/value.html --rows 0,1,2,9,15,66

# 3. render (85.8 s, 1920x1080, 30 fps, silent) and compress
cd docs/demo/source
npx --yes hyperframes@0.8.156 check
npx --yes hyperframes@0.8.156 render -f 30 -q looks -o renders/genai-recommender-master.mp4
cd -
ffmpeg -y -i docs/demo/source/renders/genai-recommender-master.mp4 -c:v libx264 -preset slow -crf 25 \
  -pix_fmt yuv420p -movflags +faststart -an docs/demo/demo.mp4

# 4. GIFs (value slide and architecture loop)
F="fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle"
ffmpeg -y -ss 55.5 -t 12 -i docs/demo/source/renders/genai-recommender-master.mp4 -vf "$F" -loop 0 docs/demo/demo.gif
ffmpeg -y -ss 30.6 -t 7.8 -i docs/demo/source/renders/genai-recommender-master.mp4 -vf "$F" -loop 0 docs/demo/architecture.gif

# 5. audit report: slide stills, automated checks, AUDIT_REPORT.md/.docx/.pdf
python3 docs/demo/source/tools/build_audit_report.py docs/demo/source/audit/facts.yaml \
  --master docs/demo/source/renders/genai-recommender-master.mp4
```

`build_audit_report.py` exits with code 1 if any figure on screen no longer matches the repo outputs.
Training and latency depend on the machine; the report lists which figures and slides each input
affects (change-impact matrix).

The `renders/` folder (about 13 MB) is not committed; step 3 recreates it.
