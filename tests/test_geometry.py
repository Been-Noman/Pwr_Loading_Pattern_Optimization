import numpy as np
import pytest
import geometry as g

def test_core_geometry_and_counts():
    assert g.N_CELLS == 193
    assert g.GRID_SIDE == 15
    assert (g.N_FRESH, g.N_ONCE, g.N_TWICE) == (65, 64, 64)
    assert len(g.POS) == 193
    assert len({tuple(v) for v in g.POS.values()}) == 193
    assert len(g.ALL_FUEL_COORDS) + len(g.WATER_POSITIONS) == 225

def test_grid_inverse_and_d4_hash():
    rng = np.random.default_rng(1234)
    p = np.concatenate([np.ones(65), np.full(64, 2), np.full(64, 3)]).astype(int)
    rng.shuffle(p)
    assert np.array_equal(g.grid_to_chromosome(g.chromosome_to_grid(p)), p)
    assert len(g.pattern_hash(p)) == 64
    assert len(g.d4_variants(p)) <= 8
    assert all(g.d4_hash(p) == g.d4_hash(v) for v in g.d4_variants(p))

def test_wrong_inventory_is_rejected():
    p = np.ones(g.N_CELLS, dtype=int)
    x = g.count_violations(p)
    assert x['total'] > 0
    assert not g.is_hard_feasible(p)
    with pytest.raises(ValueError):g.count_violations(p[:-1])

def test_onehot_channels():
    p=np.concatenate([np.ones(65),np.full(64,2),np.full(64,3)]).astype(int)
    x=g.patterns_to_onehot_grid(p)
    assert x.shape == (1,15,15,3)
    assert x.sum() == 193
