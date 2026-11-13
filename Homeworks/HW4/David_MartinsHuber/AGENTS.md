# HW4 implementation rules

- Do not copy `Lectures/Lecture 8/1_backtesting.ipynb` into the HW4 submission. Use its structure and concepts for inspiration, then implement the HW4 backtester independently.
- Reuse the lecture data-loading code directly. Keep it in a callable Python module in this submission folder so both `Research.ipynb` and the backtesting code can import the same functions.
- Cite the reused lecture data-loading code in the module and in the submission documentation.
- Keep `load_data` and `load_membership` verbatim from `Lectures/modeling/course_data.py`, including defaults, assertions, and warnings. Do not add custom loading or cleaning behavior. Pass database paths and date cutoffs explicitly at call sites. Our independent implementation is the backtester.
