# CLAUDE.md

This is my HW1 for the UCB MFE Python Pre-Program, Fall 2026. Arav Shanghavi.

The assignment lives in `Lectures/Lecture 1/hw1.md`. Shared starter files are one level up in `Homeworks/HW1/`.

Branch is `hw_1_arav_shanghavi`. The PR goes into `main` as `HW1: Arav Shanghavi`, assigned to `kevinramlal` and `raghuramkowdeed`.

## What's here

| File | What it is |
| --- | --- |
| `git.txt` | My name and favorite movie, on one line |
| `uv.txt` | The uv commands I ran, plus the latest `requests` version I ended up with |
| `game_of_life.py` | My `evolve` function, with the two given tests left as they were |
| `CLAUDE.md` | This file |

## Game of Life

`evolve` takes a grid of 0s and 1s and hands back the next generation as a new grid. The input is left alone.

A live cell stays alive with 2 or 3 live neighbors. A dead cell comes alive with exactly 3. Everything else is dead next round. Each cell looks at the 8 cells around it, and anything off the edge counts as dead. No wrapping.

The first test is a 2x2 block, which never changes. The second is a vertical line of three that should flip sideways.

## Running it

From this folder, on Windows:

```
HW1\Scripts\activate
python game_of_life.py
```

If nothing prints, both asserts passed.

## Ground rules for Claude Code

Only touch files in this folder. The shared starter files and other people's folders stay as they are.

Keep both test cases in `game_of_life.py`. Keep the code plain, with no comments.

Never commit the `HW1/` environment, a `.venv/`, `.env` files or anything with credentials in it.
