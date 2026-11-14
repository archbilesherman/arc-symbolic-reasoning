"""Two-operation program semantics and bounded counterexample-guided search."""
import numpy as np
import pytest

from src.arc_symbolic.composition import find_composite_survivors
from src.arc_symbolic.pixels import PixelOperator
from src.arc_symbolic.programs import CompositeProgram


def _recolor_crop_examples():
    first = [[0,0,0,0,0],[0,1,1,0,3],[0,1,1,0,0],[0,0,0,0,0]]
    second = [[0,3,0,0,0],[0,1,1,0,0],[0,1,1,0,0],[0,0,0,0,0]]
    result = [[2,2],[2,2]]
    return [dict(input=grid,output=result) for grid in (first,second)]


def test_composition_executes_in_order():
    program = CompositeProgram((
        PixelOperator('geometry',operation='rotate_90'),
        PixelOperator('recolor',source=1,target=2),
    ))
    assert program.apply([[1,0,1],[0,1,0]]).tolist() == [[2,0],[0,2],[2,0]]
    with pytest.raises(ValueError):
        CompositeProgram((PixelOperator('geometry',operation='identity'),))


def test_composition_finds_recolor_then_crop():
    examples = _recolor_crop_examples()
    found, report = find_composite_survivors(examples, max_prefixes=20,
                                               max_pair_evaluations=256)
    assert found
    assert report['pair_evaluations'] <= 256
    assert all(all(np.array_equal(h.apply(e['input']),e['output'])
                   for e in examples) for h in found)
    # No training output is needed when subsequently applying learned programs.
    assert any(np.array_equal(h.apply(examples[1]['input']),examples[1]['output'])
               for h in found)


def test_composition_finds_geometry_and_recolor():
    matrices = [[[1,0,1],[0,1,0]],[[0,1,0],[1,0,1]]]
    examples = []
    for grid in matrices:
        expected = np.rot90(np.asarray(grid),1)
        expected[expected==1] = 2
        examples.append({'input':grid,'output':expected.tolist()})
    found, _ = find_composite_survivors(examples)
    assert found
    assert all(np.array_equal(h.apply(examples[0]['input']), examples[0]['output'])
               for h in found)


def test_composition_budget_and_repeatability():
    examples = _recolor_crop_examples()
    a, m1 = find_composite_survivors(examples,max_prefixes=8,max_pair_evaluations=21)
    b, m2 = find_composite_survivors(examples,max_prefixes=8,max_pair_evaluations=21)
    assert m1 == m2
    assert [h.label for h in a] == [h.label for h in b]
    assert m1['prefixes_selected'] <= 8
    assert m1['pair_evaluations'] <= 21
    with pytest.raises(ValueError):
        find_composite_survivors(examples,max_pair_evaluations=0)


def test_composition_finds_recolor_and_evidence_guided_translation():
    first = np.zeros((5,5),dtype=int)
    first[1:3,1:3] = 1
    second = np.zeros((5,5),dtype=int)
    second[0:2,2:4] = 1
    examples = []
    for before in (first, second):
        after = np.zeros_like(before)
        for r,c in np.argwhere(before == 1):
            after[r+1,c] = 2
        examples.append({'input':before.tolist(), 'output':after.tolist()})
    candidates, report = find_composite_survivors(examples)
    assert candidates
    assert report['pair_evaluations'] <= 256
    assert any('translate' in candidate.label for candidate in candidates)