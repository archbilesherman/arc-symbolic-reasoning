"""Connected multicolor objects and a bounded, task-local candidate vocabulary.

A multicolor object is a foreground connected component; colored adjacency alone
is not assumed to be the uniquely correct interpretation of a scene. The solver
may continue to use the original monochromatic representation independently.
"""
from dataclasses import dataclass
from functools import lru_cache
from collections import deque

import numpy as np


@dataclass(frozen=True)
class MultiColorObject:
    id: int
    colored_cells: tuple[tuple[int, int, int], ...]

    @property
    def cells(self):
        return tuple((r, c) for r, c, _ in self.colored_cells)

    @property
    def area(self):
        return len(self.colored_cells)

    @property
    def bounding_box(self):
        rows = [r for r, _, _ in self.colored_cells]
        cols = [c for _, c, _ in self.colored_cells]
        return min(rows), min(cols), max(rows), max(cols)

    @property
    def height(self):
        top, _, bottom, _ = self.bounding_box
        return bottom - top + 1

    @property
    def width(self):
        _, left, _, right = self.bounding_box
        return right - left + 1

    @property
    def color_set(self):
        return frozenset(color for _, _, color in self.colored_cells)

    @property
    def color_histogram(self):
        return tuple((color, sum(v == color for _, _, v in self.colored_cells))
                     for color in sorted(self.color_set))

    @property
    def normalized_shape(self):
        top, left, _, _ = self.bounding_box
        return tuple(sorted((r - top, c - left) for r, c in self.cells))

    @property
    def normalized_pattern(self):
        top, left, _, _ = self.bounding_box
        return tuple(sorted((r - top, c - left, v)
                            for r, c, v in self.colored_cells))


@lru_cache(maxsize=256)
def _extract_cached(shape, data, connectivity):
    height, width = shape
    arr = np.frombuffer(data, dtype=np.int64).reshape(shape)
    visited = np.zeros(shape, dtype=bool)
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
    objects = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or arr[row, col] == 0:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []
            while queue:
                r, c = queue.popleft()
                cells.append((r, c, int(arr[r, c])))
                for dr, dc in directions:
                    nr, nc = r + dr, c + dc
                    if (0 <= nr < height and 0 <= nc < width
                            and not visited[nr, nc] and arr[nr, nc] != 0):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            objects.append(MultiColorObject(len(objects), tuple(sorted(cells))))
    return tuple(objects)


def extract_multicolor_objects(grid, connectivity=4):
    """Return foreground-connected regions while preserving each pixel's color."""
    if connectivity not in (4, 8):
        raise ValueError('connectivity must be 4 or 8')
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2 or 0 in arr.shape or any(x > 30 for x in arr.shape):
        raise ValueError('ARC grid must be a nonempty, at-most-30x30 array')
    if np.any((arr < 0) | (arr > 9)):
        raise ValueError('ARC colors must be between 0 and 9')
    arr = np.ascontiguousarray(arr)
    return list(_extract_cached(arr.shape, arr.tobytes(), connectivity))


def propose_multicolor_hypotheses(examples, max_candidates=128):
    """Generate a small pool using colored patterns and shape abstractions.

    Both 4- and 8-connectivity receive a share of the search budget. Proposals
    are ranked by description complexity, rather than generated in an
    unrestricted product of object features and transformations.
    """
    from .hypotheses import make_hypothesis
    from .operators import MultiColorOperator
    from .selectors import MultiColorPredicate

    if max_candidates < 1:
        raise ValueError('max_candidates must be >= 1')
    colors_out = sorted({int(v) for ex in examples
                         for v in np.unique(ex['output']) if v != 0})
    by_connectivity = []
    for connectivity in (4, 8):
        objects = [obj for ex in examples
                   for obj in extract_multicolor_objects(ex['input'], connectivity)]
        mixed = [obj for obj in objects if len(obj.color_set) > 1]
        if not mixed:
            continue
        color_sets = sorted({tuple(sorted(o.color_set)) for o in mixed})[:6]
        patterns = sorted({o.normalized_pattern for o in mixed},
                          key=lambda s:(len(s),s))[:4]
        shapes = sorted({o.normalized_shape for o in mixed},
                        key=lambda s:(len(s),s))[:4]
        colors_present = sorted({v for o in mixed for v in o.color_set})[:6]
        predicates = [MultiColorPredicate()]
        predicates += [MultiColorPredicate(color_set=frozenset(c)) for c in color_sets]
        predicates += [MultiColorPredicate(pattern=p) for p in patterns]
        predicates += [MultiColorPredicate(includes_color=c) for c in colors_present]
        predicates += [MultiColorPredicate(shape=s) for s in shapes]
        local = {}
        for pred in predicates:
            for action in ('crop', 'keep', 'erase'):
                op = MultiColorOperator(action, pred, connectivity)
                local[op] = make_hypothesis(op)
            for color in colors_out[:3]:
                op = MultiColorOperator('recolor', pred, connectivity,
                                         target_color=color)
                local[op] = make_hypothesis(op)
        for rank in ('largest', 'smallest'):
            for action in ('crop','keep','erase'):
                op = MultiColorOperator(action, MultiColorPredicate(),
                                         connectivity, rank=rank)
                local[op] = make_hypothesis(op)
        by_connectivity.append(sorted(local.values(),
                                      key=lambda h:(h.complexity,h.label)))
    results = []
    for index in range(max((len(group) for group in by_connectivity), default=0)):
        for group in by_connectivity:
            if index < len(group) and len(results) < max_candidates:
                results.append(group[index])
        if len(results) >= max_candidates:
            break
    return results