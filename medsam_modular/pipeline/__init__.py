"""Pipeline stage entry points.

The runner still owns orchestration, but stage-specific modules provide stable
boundaries for gradually moving the heavier implementation out of runner.py.
"""

