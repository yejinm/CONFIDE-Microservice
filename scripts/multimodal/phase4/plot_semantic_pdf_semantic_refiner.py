"""Compatibility entry point for the paper's Semantic Refiner terminology.

The implementation historically used the name DADE for the component that the
paper calls Semantic Refiner (SR). Keep this wrapper so readers can find and run
the figure script using the paper term.
"""

from __future__ import annotations

from plot_semantic_pdf_dade import main


if __name__ == "__main__":
    main()
