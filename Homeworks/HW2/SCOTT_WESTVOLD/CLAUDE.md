# CLAUDE.md — HW2 (Scott Westvold)

This file describes how I want Claude to work with me on this assignment.

## Working model: small, scoped tasks

I own the **architecture and design**. Claude implements **small, well-defined pieces** that I ask for, one at a time.

- **I decide** the overall structure: which files, modules, classes, and functions exist, how they fit together, and what each one is responsible for.
- **I decide** how specific pieces should be done when I care about it: the approach, data structures, libraries, function signatures, and return types.
- **Claude builds** the logic inside the small pieces I hand off, such as a single function, a helper, a short block, or a fix.

## Rules for Claude

1. **One piece at a time.** Only do the specific task I asked for. Don't write the whole assignment, a whole file, or several functions at once unless I explicitly ask.
2. **Don't make architecture decisions.** Don't add new files, classes, modules, or functions, and don't restructure or rename existing code, unless I ask. If you think the design should change, suggest it and let me decide.
3. **Follow my specs.** When I give a signature, approach, or constraint, stick to it. If something is unclear or there's a real tradeoff, ask a short question instead of guessing.
4. **Stay in scope.** Don't "improve" code outside the piece I asked about. Point out bugs or issues you notice elsewhere, but don't fix them unprompted.
5. **Explain briefly.** After each piece, give a short explanation of how the logic works and any assumptions or edge cases, so I understand and can check it.
6. **Keep it readable.** Write clear, commented Python with good names that matches the style of my existing code. Use type hints and docstrings where they make sense.
7. **Stop and hand back.** After finishing a piece, stop and wait for my next instruction. Don't move on to the next step by yourself.

## Hard boundaries (never break these)

- **Never create, edit, move, or delete anything outside this folder** (`Homeworks/HW2/SCOTT_WESTVOLD`). This includes lecture materials and repo-level files. Reading course materials (e.g. `Lectures/`, `README.md`, the assignment instructions) for reference is fine, but nothing outside this folder gets changed.
- **Never read, open, search, or list anything in another student's folder, under any circumstances.** That means any `Homeworks/HW*/<name>` folder other than `SCOTT_WESTVOLD`, including through git history, diffs, or search tools. My work must be my own. If a search would turn up other students' files, narrow it to this folder or to the course materials.
- **Never run any git command, ever. No exceptions.** This means no `git` or `gh` command of any kind, read-only ones included. Never look at git history (`git log`, `git show`, `git blame`) or diffs (`git diff`), and never run status, add, commit, push, pull, fetch, branch, checkout, switch, stash, reset, restore, merge, rebase, or PRs. Don't get around this with other tools, like reading files inside the `.git` folder. If a task seems to need git, stop and tell me, and I'll do it myself. I handle all git and submission.
- Don't copy `.env` files or `data.db` into the homework.

## Example requests

- "Write a function `load_prices(path: str) -> pd.DataFrame` that reads the CSV and parses the date column as the index."
- "Fill in the body of `compute_returns`: use log returns, drop the first NaN row."
- "This function is throwing a KeyError. Find why and fix only that."
