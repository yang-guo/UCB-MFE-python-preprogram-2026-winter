# HW1
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
    - Create a virtual environment called `HW1` with Python 3.12.9
    - Install the packages from the requirement file in `Homeworks/HW1/requirements.txt`
    - Downgrade the `requests` library to version 2.24.0
    - Upgrade the `requests` library to the latest
    - What's the latest version of `requests`?
- **IMPORTANT**: Do NOT include your virtual environment folder (`.venv`) in your submission. It should stay on your local machine only.

## Python
- make a copy of `Homeworks/HW1/game_of_life.py` to `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/game_of_life.py`
- read this wikipedia of Game of Life: https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life
- Implement the function `evolve`, which takes a board and returns the next stage of the board.
- The implemented function should be able to pass the 2 test cases in the file.
- please run your file `python game_of_life.py` locally to make sure it passes all the test cases before submission.
    - PLEASE MAKE SURE YOU DO THIS STEP.

## Submission Checklist
Your submission folder `Homeworks/HW1/<FIRST_NAME>_<LAST_NAME>/` should contain:
- [ ] `git.txt` - Your name and favorite movie
- [ ] `uv.txt` - Commands used for uv package management
- [ ] `CLAUDE.md` - Proof of Claude Code installation
- [ ] `game_of_life.py` - Your implementation of the evolve function

## Submission
- **Due date: October 19, 2026** (two weeks after the Basic Python lecture on October 5).
- commit your changes and submit a pull request, assign to Kevin Ramlal (`kevinramlal`) and Benjamin Steel (`elfiasco`)
