#!/bin/zsh
cd -- "${0:A:h}/demo"
export PATH="$HOME/.local/bin:$PATH"
uv run --locked jupyter lab ../Lesson_9_Demos.ipynb
