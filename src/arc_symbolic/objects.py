from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import numpy as np

@dataclass(frozen=True)
class ARCObject:
    id: int
    color: int
    cells: tuple[tuple[int, int], ...]
    @property
    def area(self): return len(self.cells)
    @property
    def bounding_box(self):
        rows, cols = zip(*self.cells)
        return min(rows), min(cols), max(rows), max(cols)
    @property
    def height(self):
        top, _, bottom, _ = self.bounding_box
        return bottom-top+1
    @property
    def width(self):
        _, left, _, right = self.bounding_box
        return right-left+1
    @property
    def centroid(self):
        return sum(r for r,c in self.cells)/self.area, sum(c for r,c in self.cells)/self.area
    @property
    def normalized_shape(self):
        top,left,_,_=self.bounding_box
        return tuple(sorted((r-top,c-left) for r,c in self.cells))

def extract_objects(grid, connectivity=4):
    if connectivity not in (4,8): raise ValueError('Connectivity must be 4 or 8')
    arr=np.asarray(grid,dtype=np.int64)
    if arr.ndim!=2 or 0 in arr.shape: raise ValueError('ARC grid must be a nonempty 2D array')
    if np.any((arr<0)|(arr>9)): raise ValueError('ARC colors must be between 0 and 9')
    return list(_cached_monochrome_objects(arr.shape, np.ascontiguousarray(arr).tobytes(), connectivity))

@lru_cache(maxsize=512)
def _cached_monochrome_objects(shape, grid_bytes, connectivity):
    """Memoize repeated scene parsing across candidate evaluations."""
    arr=np.frombuffer(grid_bytes,dtype=np.int64).reshape(shape)
    height,width=arr.shape
    visited=np.zeros(arr.shape,dtype=bool)
    neighbors=[(-1,0),(1,0),(0,-1),(0,1)]
    if connectivity==8: neighbors += [(-1,-1),(-1,1),(1,-1),(1,1)]
    objects=[]
    for row in range(height):
        for col in range(width):
            if arr[row,col]==0 or visited[row,col]:continue
            color=int(arr[row,col]);q=deque([(row,col)]);visited[row,col]=True;cells=[]
            while q:
                r,c=q.popleft();cells.append((r,c))
                for dr,dc in neighbors:
                    nr,nc=r+dr,c+dc
                    if 0<=nr<height and 0<=nc<width and not visited[nr,nc] and arr[nr,nc]==color:
                        visited[nr,nc]=True;q.append((nr,nc))
            objects.append(ARCObject(len(objects),color,tuple(sorted(cells))))
    return tuple(objects)