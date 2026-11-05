# Data for the harness lesson

`prices.csv` contains generated prices for AAPL, MSFT, and JPM. There are 86 weekdays per ticker from January 2 through April 30, 2024, including US market holidays. Raw and adjusted prices are equal by construction.

Construction used Python's `random.Random(26)`, iterating tickers in AAPL, MSFT, JPM order. Starting prices were 100, 80, and 60. After the first observation, prices multiplied by `exp(random.gauss(mu, sigma))`, with `(mu, sigma)` equal to `(0.0005, 0.018)`, `(0.0008, 0.012)`, and `(0.0002, 0.009)`. Values were rounded to six decimal places. The CSV itself is the fixed fixture used by tests and the example report.

`holdings.csv` contains the portfolio's lots. The starting cash balance is a project parameter, defaulting to 5,000.

`sources.json` describes how the sample data was constructed and includes a May 10 AAPL note for the date-filtering example. `course_sources.json` describes the Lecture 4 database. To investigate an issuer's business, add filing excerpts using the same record structure.

Course mode reads the bCourses SQLite database from the Lecture 4 folder. See [Lecture 4](../../Lecture%204/readme.md) for its schema and data-quality checks.

`retrieval_lab_sources.json` contains five short classroom excerpts with invented debt and cash figures. Scripts 06 and 08 use them to make ticker filtering, publication dates, response budgets, and synonym misses easy to inspect.
