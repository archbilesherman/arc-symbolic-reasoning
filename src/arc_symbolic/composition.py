"""Evidence-guided two-operation search with explicit evaluation budgets.

Composition is not the Cartesian product of the V4 candidate library. Candidate
prefixes are scored on training inputs, then a small suffix vocabulary is
constructed from the intermediate grids. Only exact programs are returned.
"""
from collections import Counter

import numpy as np

from .hypotheses import make_hypothesis
from .operators import ObjectOperator, MultiColorOperator
from .pixels import PixelOperator, GEOMETRIC_OPS
from .selectors import AttributePredicate, MultiColorPredicate
from .programs import CompositeProgram, _valid_intermediate


def _foreground_crop(grid):
    cells = np.argwhere(np.asarray(grid) != 0)
    if not len(cells):
        return None
    top, left = cells.min(axis=0)
    bottom, right = cells.max(axis=0)
    return np.asarray(grid)[top:bottom+1, left:right+1]


def _similarity_error(predicted, expected):
    """Low-cost heuristic; never substitutes for exact replay verification."""
    a = np.asarray(predicted)
    b = np.asarray(expected)
    if a.shape == b.shape:
        exact = np.mean(a != b)
        occupancy = np.mean((a != 0) != (b != 0))
        return float(min(exact, 0.12 + occupancy))
    crop = _foreground_crop(a)
    if crop is not None and crop.shape == b.shape:
        return float(0.06 + min(np.mean(crop != b),
                                0.12 + np.mean((crop != 0) != (b != 0))))
    # Shape differences can be fixed by geometry/crop; keep the signal weak.
    area_a = (crop.size if crop is not None else a.size)
    area_b = b.size
    size_gap = abs(area_a-area_b)/max(area_a,area_b,1)
    return 0.70 + 0.25 * size_gap


def _pool_color_ops(examples):
    inputs = sorted({int(v) for e in examples for v in np.unique(e['input']) if v})
    outputs = sorted({int(v) for e in examples for v in np.unique(e['output']) if v})
    return inputs, outputs


def _prefix_pool(examples, memories=(), max_programs=240):
    """Construct a diverse but finite prefix vocabulary from observed colors."""
    source_colors, output_colors = _pool_color_ops(examples)
    pool = []
    seen = set()

    def add(op):
        if op not in seen and len(pool) < max_programs:
            seen.add(op)
            pool.append(op)

    # Memory contributes known near misses without treating them as correct.
    for item in memories:
        program = item.hypothesis.program
        if isinstance(program, (ObjectOperator, PixelOperator, MultiColorOperator)):
            add(program)
    for geom in GEOMETRIC_OPS:
        if geom != 'identity':
            add(PixelOperator('geometry', operation=geom))
    for c in source_colors:
        for target in output_colors:
            if c != target:
                add(PixelOperator('recolor', source=c, target=target))
    # Multicolor prefixes are evidence-driven and remain a small fraction.
    from .multicolor import extract_multicolor_objects
    for connectivity in (4, 8):
        mixed = [obj for ex in examples
                 for obj in extract_multicolor_objects(ex['input'], connectivity)
                 if len(obj.color_set) > 1]
        if mixed:
            add(MultiColorOperator('erase', MultiColorPredicate(), connectivity))
            add(MultiColorOperator('keep', MultiColorPredicate(), connectivity))
            add(MultiColorOperator('crop', MultiColorPredicate(), connectivity))
            for colors in sorted({tuple(sorted(o.color_set)) for o in mixed})[:3]:
                predicate = MultiColorPredicate(color_set=frozenset(colors))
                for action in ('erase','keep','crop'):
                    add(MultiColorOperator(action, predicate, connectivity))
    for connectivity in (4, 8):
        for color in [None] + source_colors:
            attrs = None if color is None else AttributePredicate(color=color)
            for action in ('erase','keep','crop'):
                add(ObjectOperator(action, attributes=attrs, connectivity=connectivity))
            if color is not None:
                for target in output_colors[:4]:
                    if color != target:
                        add(ObjectOperator('recolor', attributes=attrs,
                                           connectivity=connectivity,
                                           target_color=target))
        for rank in ('largest','smallest'):
            for action in ('erase','keep','crop'):
                add(ObjectOperator(action, rank=rank, connectivity=connectivity))

    return pool


def _suffix_pool(intermediate, examples, limit=56):
    """Propose small second-step vocabulary conditional on intermediate grids."""
    expected = [np.asarray(ex['output']) for ex in examples]
    intermediate = np.asarray(intermediate)
    source = sorted({int(v) for v in np.unique(intermediate) if v})
    targets = sorted({int(v) for out in expected for v in np.unique(out) if v})
    needs_resize = any(out.shape != intermediate.shape for out in expected)
    options = []
    seen = set()

    def add(op):
        if op not in seen:
            seen.add(op)
            options.append(op)

    if needs_resize:
        # An unrestricted foreground crop is often the missing second step.
        for connectivity in (4, 8):
            add(ObjectOperator('crop', connectivity=connectivity))
            add(MultiColorOperator('crop', MultiColorPredicate(), connectivity))
            for color in source:
                add(ObjectOperator('crop', attributes=AttributePredicate(color=color),
                                   connectivity=connectivity))
        for geom in GEOMETRIC_OPS:
            if geom != 'identity':
                add(PixelOperator('geometry', operation=geom))
    else:
        # Prefer correcting appearance or orientation before cropping.
        for c in source:
            for target in targets:
                if c != target:
                    add(PixelOperator('recolor', source=c, target=target))
                    add(ObjectOperator('recolor', attributes=AttributePredicate(color=c),
                                       target_color=target))
        # Infer displacements from matching object shapes in the intermediate
        # and demonstrated output; do not search arbitrary grid offsets.
        from .generator import _motion_offsets
        offsets = _motion_offsets([{'input':intermediate.tolist(),
                                    'output':expected[0].tolist()}],
                                  connectivity=4, limit=4)
        for offset in offsets:
            for color in source[:4]:
                attrs = AttributePredicate(color=color)
                add(ObjectOperator('translate', attributes=attrs,
                                   offset=offset))
                add(ObjectOperator('duplicate', attributes=attrs,
                                   offset=offset))
        for geom in GEOMETRIC_OPS:
            if geom != 'identity':
                add(PixelOperator('geometry', operation=geom))
        for color in source:
            add(ObjectOperator('erase', attributes=AttributePredicate(color=color)))
        add(ObjectOperator('crop'))
    return options[:limit]


def find_composite_survivors(
    examples, *, memories=(), max_prefixes=20,
    max_pair_evaluations=256, max_prefix_pool=240,
):
    """Search two-step programs; count exact executions and honor hard budgets.

    The search uses only training inputs/outputs. Test labels are inaccessible.
    Return (exact_survivors, diagnostics).
    """
    if not examples:
        return [], {'prefix_evaluations':0, 'pair_evaluations':0,
                    'prefixes_selected':0, 'composite_survivors':0}
    if min(max_prefixes, max_pair_evaluations, max_prefix_pool) < 1:
        raise ValueError('All search budgets must be positive')

    initial = np.asarray(examples[0]['input'], dtype=np.int64)
    desired = np.asarray(examples[0]['output'], dtype=np.int64)
    candidates = []
    prefix_evals = 0
    for prefix in _prefix_pool(examples, memories, max_prefix_pool):
        intermediate = prefix.apply(initial)
        prefix_evals += 1
        if not _valid_intermediate(intermediate):
            continue
        if np.array_equal(intermediate, initial):
            continue
        score = _similarity_error(intermediate, desired)
        candidates.append((score, make_hypothesis(prefix).complexity,
                           repr(prefix), prefix, np.asarray(intermediate)))
    candidates.sort(key=lambda x:(x[0],x[1],x[2]))

    # No single family should monopolize the beam (e.g. rank/color keep).
    selected = []
    family_counts = Counter()
    for record in candidates:
        prefix = record[3]
        family = ((type(prefix).__name__, prefix.action)
                  if hasattr(prefix, 'action') else (type(prefix).__name__, prefix.kind))
        if family_counts[family] >= 4:
            continue
        selected.append(record)
        family_counts[family] += 1
        if len(selected) >= max_prefixes:
            break

    # Round robin across prefixes so the search budget is not exhausted by
    # suffixes of the first prefix. The first successful exact candidates win.
    trials = []
    for _, _, _, prefix, intermediate in selected:
        suffixes = _suffix_pool(intermediate, examples)
        trials.append((prefix, suffixes))

    survivors = []
    seen_programs = set()
    pair_evals = 0
    for turn in range(max((len(x[1]) for x in trials), default=0)):
        for prefix, suffixes in trials:
            if turn >= len(suffixes):
                continue
            if pair_evals >= max_pair_evaluations:
                break
            suffix = suffixes[turn]
            program = CompositeProgram((prefix, suffix))
            if program in seen_programs:
                continue
            seen_programs.add(program)
            pair_evals += 1
            # Exact verification uses all demonstrations and fails fast.
            valid = True
            for example in examples:
                output = program.apply(example['input'])
                if output is None or not np.array_equal(output, example['output']):
                    valid = False
                    break
            if valid:
                survivors.append(make_hypothesis(program))
        if pair_evals >= max_pair_evaluations:
            break
    survivors.sort(key=lambda h:(h.complexity,h.label))
    return survivors, {
        'prefix_evaluations':prefix_evals,
        'prefixes_selected':len(selected),
        'pair_evaluations':pair_evals,
        'composite_survivors':len(survivors),
    }