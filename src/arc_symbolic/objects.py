from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ARCObject:
    """A monochromatic connected component in an ARC grid."""

    id: int
    color: int
    cells: tuple[tuple[int, int], ...]

    @property
    def area(self) -> int:
        return len(self.cells)

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Inclusive bounds: (top, left, bottom, right)."""
        rows, cols = zip(*self.cells)
        return min(rows), min(cols), max(rows), max(cols)

    @property
    def height(self) -> int:
        top, _, bottom, _ = self.bounding_box
        return bottom - top + 1

    @property
    def width(self) -> int:
        _, left, _, right = self.bounding_box
        return right - left + 1

    @property
    def centroid(self) -> tuple[float, float]:
        """Average cell position: (row, column)."""
        row = sum(r for r, _ in self.cells) / self.area
        col = sum(c for _, c in self.cells) / self.area
        return row, col

    @property
    def normalized_shape(self) -> tuple[tuple[int, int], ...]:
        """Translation-invariant coordinates relative to top-left."""
        top, left, _, _ = self.bounding_box
        return tuple(
            sorted((r - top, c - left) for r, c in self.cells)
        )


def extract_objects(grid, connectivity: int = 4) -> list[ARCObject]:
    """
    Extract monochromatic connected components from an ARC grid.

    Background color 0 is ignored.
    Connectivity must be 4 or 8.
    Object IDs follow row-major discovery order.
    """

    if connectivity not in (4, 8):
        raise ValueError("Connectivity must be 4 or 8")

    arr = np.asarray(grid, dtype=np.int64)

    if arr.ndim != 2 or 0 in arr.shape:
        raise ValueError("ARC grid must be a nonempty 2D array")

    if np.any((arr < 0) | (arr > 9)):
        raise ValueError("ARC colors must be between 0 and 9")

    height, width = arr.shape
    visited = np.zeros(arr.shape, dtype=bool)

    neighbors = [
        (-1, 0), (1, 0),
        (0, -1), (0, 1),
    ]

    if connectivity == 8:
        neighbors += [
            (-1, -1), (-1, 1),
            (1, -1), (1, 1),
        ]

    objects = []

    for row in range(height):
        for col in range(width):

            if arr[row, col] == 0 or visited[row, col]:
                continue

            color = int(arr[row, col])
            queue = deque([(row, col)])
            visited[row, col] = True
            cells = []

            while queue:
                r, c = queue.popleft()
                cells.append((r, c))

                for dr, dc in neighbors:
                    nr, nc = r + dr, c + dc

                    if not (0 <= nr < height and 0 <= nc < width):
                        continue

                    if visited[nr, nc]:
                        continue

                    if arr[nr, nc] != color:
                        continue

                    visited[nr, nc] = True
                    queue.append((nr, nc))

            objects.append(ARCObject(id=len(objects), color=color, cells=tuple(sorted(cells))))

    return objects