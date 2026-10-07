# UCB-MFE-python-preprogram-2026-winter

## Class Schedule

Fall 2026: October 5-November 5. All times are Pacific Time (PT).

| Day      | Date       | Time        | Lecturer | Topics |
| -------- | ---------- | ----------- | -------- | ------ |
| Monday   | 10/5/2026  | 9am - 12pm  | Aneesh   | Basics: Python syntax, Git, CLI, AI tools |
| Thursday | 10/8/2026  | 9am - 12pm  | Yang     | [Data Analysis I & II: pandas, plotting](Lectures/Lecture%202/readme.md) |
| Monday   | 10/12/2026 | 9am - 12pm  | Aneesh   | [LLM Classification & Agent Evaluation](Lectures/Lecture%203/readme.md) |
| Thursday | 10/15/2026 | 9am - 12pm  | Yang     | [Data Analysis III: SQL, data quality, automated reports](Lectures/Lecture%204/readme.md) |
| Monday   | 10/19/2026 | 9am - 12pm  | Aneesh   | Advanced Python I: OOP, environments, regex |
| Thursday | 10/22/2026 | 9am - 12pm  | Aneesh   | Advanced Python II: async, functional programming |
| Monday   | 10/26/2026 | 9am - 12pm  | Yang     | Modeling I: scikit-learn basics |
| Thursday | 10/29/2026 | 9am - 12pm  | Yang     | Modeling II: scikit-learn advanced |
| Monday   | 11/2/2026  | 9am - 12pm  | Aneesh   | Productionization: testing, Docker, CI |
| Thursday | 11/5/2026  | 9am - 12pm  | Yang     | Advanced topic: on demand |

## Homework
- one homework per module
- Homework is due two weeks (14 calendar days) after the module's last lecture.
- submission via github pull request (see Lecture 1)
- Please assign PRs to Kevin Ramlal (`kevinramlal`) and Benjamin Steel (`elfiasco`)

| Module | Last Lecture | Homework Due |
| ------ | ------------ | ------------ |
| Basic Python | 10/5/2026 | 10/19/2026 |
| Data Analysis | 10/15/2026 | 10/29/2026 |
| Advanced Python | 10/22/2026 | 11/5/2026 |
| Modeling | 10/29/2026 | 11/12/2026 |
| Productionization | 11/2/2026 | 11/16/2026 |
| Advanced Topics | 11/5/2026 | 11/19/2026 |

## Goal
- Level up your coding skills
- Help you to succeed in an entry-level / internship position of data scientist / quant developer

## How to get the most of this course
- Attend live sessions and ask a lot of questions
- Review class materials and do homework right after the class, as learning curve is pretty steep.

## Operating System
- Most of the class materials are prepared with macOS 
- All class materials should work seamlessly with both Windows and Linux systems

## Extra Materials
- Relevant materials will be [linked](...) inline.
- It's highly recommended, but not a must to go over the linked materials. The course is self-contained.

## Grading
- Pass / No-Pass
- In order to pass, students needs to complete all homeworks on-time.
- Do your homework as soon as possible.

## Late Policy
- Each student will be given **three** free late days, with each late day being equivalent to 24 hours of lateness. 
- Assignments are considered either on time or late by whole days; for instance, an assignment is either on time, 1 day late, 2 days late, etc. 
- Due to the leniency of this policy, no extensions will be granted for assignments under any circumstances. 
- No assignment will be accepted if it is submitted more than 3 days after its due date.

## Communication
- Most of the communication and announcement will be sent via Slack.
- Most of the questions should be asked & answered in the group channel.
- Yang, Aneesh & GSIs will routinely check the channel to make sure all questions are answered.

## Instructors
- [Yang Guo](https://www.linkedin.com/in/yang-guo-19494ba/)
- [Aneesh Sachdeva](https://www.linkedin.com/in/aneeshsachdeva/)

## GSIs
- [Benjamin Steel](https://www.linkedin.com/in/benjaminasteel/)
- [Kevin Ramlal](https://www.linkedin.com/in/kevinramlal/)



## Running the notebooks

- From the project root, run `uv sync --locked` and select the project's `.venv` Python interpreter as your notebook kernel.
- Run each notebook with its lecture folder as the working directory. The reorganized data-analysis notebooks also support the repository root and their reference folders.
- Data analysis has two live notebooks on October 8 (Lecture 2) and three on October 15 (Lecture 4), with a ten-minute break each day. Each notebook has a continuous lesson with visible code and worked examples, followed by an Exercises section. Expanded material is optional reference.
- Lecture 2 includes a historical sample and a synthetic classroom CSV; default runs work without an API key. Its Jupyter tutorial is optional preparation.
- Worked examples use different inputs or questions from the final exercises. Exercise prompts, starter cells, and feedback stay together at the end; the lesson runs independently of student answers.
- Lecture 4's default report batch deliberately includes one invalid ticker: three reports succeed, one failure is recorded, and the comparison uses successful outputs only.
- Lecture 4 requires the bCourses database at `Lectures/Lecture 4/data/data.db`.
- Lecture 6's HTTP examples require internet access. Its multiprocessing example uses the adjacent `lecture6_workers.py` module.
- Lecture 8 uses its built-in demo if the separate homework database at `Lectures/Lecture 8/data/data.db` is absent.
- Database files, local credentials, and generated reports are not tracked.
- Verify the reorganized modules with `uv run python scripts/validate_data_analysis.py` (or add `--core-only` for the five live notebooks). The validator uses fresh kernels and a temporary database backup.
