# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Repository Overview

This is the UCB MFE Python Pre-Program course repository for 2026. The course runs from October 5 to November 5, 2026 (Fall 2026), covering Python fundamentals, data analysis, modeling, and advanced topics for MFE students.

## Repository Structure

```
/
├── Lectures/     # Lecture materials organized by date/module
└── README.md     # Course schedule and policies
```

## Course Context

- **Audience**: UCB MFE students preparing for quantitative finance roles
- **Focus**: Python for data science and quantitative development
- **Homework submission**: Via GitHub pull requests
- **Homework deadlines**: Two weeks (14 calendar days) after the module's last lecture; see README.md for the dates
- **Grading**: Pass/No-Pass based on homework completion
- **Late policy**: 3 free late days (24-hour periods), no extensions, max 3 days late

## Git Workflow

Students submit homework via pull requests. When reviewing or working with student submissions:
- Each homework corresponds to a module (Basic Python, Data Analysis, Advanced Python, Modeling, Productionization, Advanced Topics); numbered lecture parts belong to the same module
- PRs should be reviewed for correctness, code quality, and adherence to Python best practices
- Students are learning, so explanations should be clear and educational

## Python Environment

- Repository uses standard Python gitignore patterns
- `.env` files are local-only and must not be copied into course materials
- `data.db` is gitignored (local database file, distributed via bCourse)
- Development environment: primarily macOS, but should work on Windows/Linux

## Common Patterns

Since this is a teaching repository:
- Code should be clear and well-commented for educational purposes
- Examples should demonstrate best practices for data science and quantitative finance
- Jupyter notebooks will likely be used extensively for lectures and homework
- Testing and code quality are important topics to model for students
