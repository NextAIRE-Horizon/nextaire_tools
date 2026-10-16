# Agent skills

Ready-made instructions that let an AI agent do common `nextaire_tools` tasks for you.
You do not need to know Python. Describe what you want, and the agent follows the skill.

## What is a skill?

A skill is a folder with a `SKILL.md` file. It tells the agent what to ask you, what to install,
what to run, and how to explain the result. Some skills also include tested scripts.

## How to use a skill

1. Download or clone this repository.
2. Give the skill folder to your agent. Either:
   - copy the folder (e.g. `skills/clean-air-quality-data/`) into your agent's skills directory
     (for Droid: `~/.factory/skills/` for all projects, or `.factory/skills/` in one project), or
   - tell the agent the path to the folder and ask it to follow `SKILL.md`.
3. Ask in plain words, e.g. "Clean my station file `data/graz.csv` and show me the outliers."
4. Answer the agent's questions. It sets up Python and runs everything.

## Available skills

| Skill | What it does |
|---|---|
| _none yet_ | |

## Writing a new skill

1. Copy `_template/` to `skills/<your-skill-name>/`. The folder name must match `name` in the frontmatter.
2. Fill in `SKILL.md`. Write for someone who does not code.
3. Put any runnable code in `scripts/`, built on `nextaire_tools`, and test it on synthetic data
   (see `reproductions/_synthetic.py`).
4. Add the skill to the table above and an entry under `[Unreleased]` in `CHANGELOG.md`.

Folders starting with `_` are not skills and are ignored.

### Layout of a skill

```
skills/<skill-name>/
  SKILL.md        required: frontmatter (name, description) + instructions
  scripts/        optional: runnable code the agent executes as-is
  references/     optional: longer docs the agent reads when needed
  assets/         optional: sample data, report templates
```

### Guidelines

- `description` states what the skill does and when to use it, in plain words.
- The agent does the technical work. Never ask the user to run commands or edit code.
- Ask only for what the agent cannot find itself, and offer defaults.
- Cover common failures, such as missing optional extras or Copernicus credentials
  (`CDSAPI_KEY` or `~/.cdsapirc`).
- Never commit data, `.nc`/`.grib` files, or credentials.
- Time-series models must use time-ordered validation, never shuffled splits.
- One skill per use case. Keep `SKILL.md` short.
