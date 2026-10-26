# ARC Symbolic Reasoning

A knowledge-based AI project investigating symbolic reasoning, hypothesis maintenance, and incremental concept refinement for the Abstraction and Reasoning Corpus (ARC).

Developed as part of Georgia Tech's CS 4635/7637 Knowledge-Based AI course.

## Research Direction

The project investigates whether information from a version space can guide an ARC solver in refining its conceptual vocabulary when existing transformation hypotheses fail.

The approach combines three KBAI techniques:

- **Generate-and-Test:** Search over executable symbolic transformations.
- **Version Spaces:** Maintain hypotheses consistent with observed demonstrations.
- **Incremental Concept Learning:** Explore how new relational concepts could be constructed when existing representations are insufficient.

The first two components are implemented in the current prototype. Incremental concept learning remains a planned extension.

## Preliminary Results

Evaluated on 400 public ARC-AGI-1 training tasks:

| Configuration | Tasks Solved |
|---|---|
| Pixel-only transformations | 9/400 (2.25%) |
| Pixel + object transformations | 14/400 (3.50%) |

A separate Gradescope evaluation returned 0/400, demonstrating the limitations of the current hypothesis library.

## Current Limitations

The prototype uses a restricted collection of single-step transformations and predefined object representations. Future work will investigate transformation composition, relational concepts, and version-space-guided refinement.

## Status

Active research project. Preliminary symbolic baseline implemented; concept-learning experiments pending.
