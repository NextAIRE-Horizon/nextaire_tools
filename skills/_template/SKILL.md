---
name: your-skill-name
description: One or two plain-language sentences. Say what the skill does and when to use it, e.g. "Clean a raw air-quality CSV file (missing values, outliers). Use when the user has station data and wants a cleaned file or a quality report."
---

<!--
Template for a new skill. Copy this folder, rename it, and fill in each section.
Delete these comments when done.

- The folder name must equal `name` above (lowercase, digits, hyphens).
- `description` is how the agent decides to use the skill. Write it for a
  non-programmer's request, not for developers.
- Keep this file short. Move long detail into `references/`.
- Put runnable code in `scripts/` (tested, calls `nextaire_tools`) so the agent
  runs it as-is instead of writing code on the fly.
-->

# Skill title

One sentence on the outcome the user gets.

## What to ask the user

Ask only what you cannot find out yourself. Use plain words, offer defaults, and do not guess.

1. Question one (default: ...)
2. Question two (default: ...)

## Setup

1. Check for Python 3.10 or newer. If it is missing, tell the user how to install it.
2. Create and activate a virtual environment.
3. Install the library and any extras this skill needs:

   ```bash
   pip install "nextaire_tools[extra]"
   ```

## Steps

1. Step one. Name the exact command or script, e.g. `python scripts/run.py --input ... --output ...`.
2. Step two.
3. Check the result (row counts, no all-empty columns, plots look sensible) before reporting.

## What to tell the user at the end

- Where the output files are.
- A short plain-language summary of what was done and found.
- Any warnings or limits they should know about.

## Troubleshooting

| Problem | What it means | What to do |
|---|---|---|
| `MissingDependencyError` | An optional package is not installed | Install the extra named in the message |
| Example problem | Example cause | Example fix |

## Rules

- Never commit or share the user's data, `.nc`/`.grib` files, or credentials.
- Use time-ordered validation for any model; never shuffle time-series data.
- Do not modify the user's original files; write results to a new file.
