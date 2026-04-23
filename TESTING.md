When you run pytest with the --cov-report=term-missing flag, the coverage tool generates that summary table to tell you exactly how thoroughly your tests are exercising your code.

Here is exactly what each column means:

Name: The file path of the specific Python module being evaluated (e.g., src/pipeline/common.py).

Stmts (Statements): The total number of valid, executable lines of code in that file. Blank lines, comments (#), and docstrings (""") are automatically ignored and don't count toward this number.

Miss (Missing Statements): The number of executable lines that were never run during your test suite. If a test never triggered a specific if block or except clause, those lines are counted here as "missed."

Cover (Coverage Percentage): The math calculating your score: (Stmts - Miss) / Stmts. This is the percentage of the file that was successfully executed by at least one test.

Missing (Missing Lines): It lists the exact line numbers in your Python file that the tests failed to hit.
