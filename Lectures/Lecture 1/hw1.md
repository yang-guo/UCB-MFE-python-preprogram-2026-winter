# HW1

**Due: Monday, October 19, 2026, at 11:59 p.m. Pacific Time (PDT, UTC-7).**
This is two weeks after the Basic Python lecture on October 5.

See [the starter files and submission layout](../../Homeworks/HW1/README.md).

## Get permissions
- Step 1: Join the Python Preprogram 2026 Slack channel
- Step 2: Register a GitHub account if you haven't done so
- Step 3: Fill out [this form](https://forms.gle/zzNRy7nAszwbqqPy5), so we can grant you access to the GitHub repo.
- PLEASE DO THIS ASAP, AS THIS IS A DEPENDENCY FOR EVERYTHING IN THE FUTURE.

## Get access to the GitHub Repo
- Get ready
    - Connect to GitHub with SSH, follow the instruction [here](https://docs.github.com/en/authentication/connecting-to-github-with-ssh)
    - The goal is to be able to clone the repo on your local via `git clone git@github.com:yang-guo/UCB-MFE-python-preprogram-2026-winter.git` command
        - If the above command is not working, you are not done with the setup.
        - If you need help, please ask in the Slack channel.

## Git
- create a branch of the format: `hw_1_<FIRST_NAME>_<LAST_NAME>` (no space!)
- From the repository root, create your personal submission folder and copy the starter file. Replace `<FIRST_NAME>` and `<LAST_NAME>` with your own names in these commands (use a terminal such as macOS/Linux Terminal or Git Bash on Windows):

```bash
git switch -c hw_1_<FIRST_NAME>_<LAST_NAME>
mkdir -p Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>
cp Homeworks/HW1/game_of_life.py Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/game_of_life.py
```

- Put your work in your personal folder; leave the shared starter file and requirements file unchanged.
- Create a file in `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/git.txt` with the following text:
    - `<FIRST_NAME> <LAST_NAME>, <YOUR_FAV_MOVIE>`

## Claude Code
- Install Claude Code by following the instructions [here](https://docs.anthropic.com/en/docs/claude-code/getting-started)
- Navigate to your homework directory `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/`
- Run `claude` to initialize Claude Code in your directory
- Create a `CLAUDE.md` file in your homework directory with:
    - A brief description of your homework submission
    - Any notes or context you want Claude Code to know about your work
- This file serves as proof that you have successfully installed and initialized Claude Code

## uv (Python Package Manager)
- Install `uv` by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/)
- Create a file in `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/uv.txt`, write the commands you used to achieve the following:
    - Create a virtual environment called `HW1` with Python 3.12.9 inside your personal submission folder
    - Install the packages from [Homeworks/HW1/requirements.txt](../../Homeworks/HW1/requirements.txt) into that environment
    - Downgrade the `requests` library to version 2.24.0
    - Upgrade the `requests` library to the latest
    - What's the latest version of `requests`?
- **IMPORTANT**: Do NOT include your virtual environment folder (`HW1/` or `.venv/`) in your submission. It should stay on your local machine only. The repository ignores these environments.
- Keep this exercise separate from the course-wide environment: use your homework environment when installing, downgrading, and upgrading packages.
- The downgrade is a package-management exercise only: `requests==2.24.0` installs, but its old dependencies cannot be imported on Python 3.12. Use `uv pip show requests` with your homework environment activated to inspect the installed version, then upgrade before using the library.

## Python
- make a copy of `Homeworks/HW1/game_of_life.py` to `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/game_of_life.py`
- read this wikipedia of Game of Life: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
- Implement the function `evolve`, which takes a board and returns the next stage of the board.
- The implemented function should be able to pass the 2 test cases in the file.
- please run your file `python game_of_life.py` locally to make sure it passes all the test cases before submission.
    - PLEASE MAKE SURE YOU DO THIS STEP.
    - Run this command from your personal submission folder with your homework environment activated. The starter intentionally fails until you implement `evolve`; keep both supplied test cases.

## Submission Checklist
Your submission folder `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/` should contain:
- [ ] `git.txt` - Your name and favorite movie
- [ ] `uv.txt` - Commands used for uv package management
- [ ] `CLAUDE.md` - Proof of Claude Code installation
- [ ] `game_of_life.py` - Your implementation of the evolve function

## Submission
- **Due: Monday, October 19, 2026, at 11:59 p.m. Pacific Time (PDT, UTC-7).**
- From the repository root, stage only your personal submission folder, review your changes, commit, and push your branch:

```bash
git add Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/
git diff --cached
git commit -m "Submit HW1 for <FIRST_NAME> <LAST_NAME>"
git push -u origin hw_1_<FIRST_NAME>_<LAST_NAME>
```

- Open a pull request into this repository's `main` branch, titled `HW1: <FIRST_NAME> <LAST_NAME>`.
- Assign the PR to Kevin Ramlal (`kevinramlal`) and Benjamin Steel (`elfiasco`).
- Check that the PR includes all four checklist files, with no virtual environment, credentials, or changes to another student's folder or the shared files.
- The course's [late policy](../../README.md#late-policy) applies.
