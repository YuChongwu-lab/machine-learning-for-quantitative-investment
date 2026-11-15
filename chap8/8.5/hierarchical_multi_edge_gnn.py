"""Thin entry point, kept under the book-matching filename.

See run_experiment.py for the actual implementation; this just runs the
default experiment (all edge types + softmax long-only).
"""

from config import DEFAULT_EXPERIMENT
from run_experiment import run

if __name__ == "__main__":
    run(DEFAULT_EXPERIMENT)
