"""Tests for the extended Schelling model in src/extended_schelling.py.

The tests check that the model rules described in the notebook (Section 2) are
implemented as specified, including boundary and exceptional cases:

* initialisation (states, empty share, openness values, marriage cap)
* satisfaction formulas for every household type, the wrap-around boundary,
  agents with no neighbours, and the happiness threshold (S >= p)
* the segregation measure on layouts with known answers
* relocation (happy agents stay, unhappy agents move, a full grid is handled)
* neighbour-duration tracking and marriage eligibility
* pair selection (no one married twice, marriage cap, openness weighting)
* household formation (AA / BB / AB types, new single, openness of AB couples)
* invariants that must hold for whole runs, and reproducibility with a seed

Run from the project root with:  pytest tests/
"""

import numpy as np
import pytest

from extended_schelling import Person, Schelling

OPENNESS_LEVELS = {round(0.1 * k, 1) for k in range(1, 10)}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build(layout, p=0.5, m=0.0, openness=0.5):
    """Create a Schelling model whose grid is set to a hand-made layout.

    layout: square 2-D list of cell states (0 empty, 1 A, 2 B, 3 AA, 4 BB, 5 AB)
    Every occupant gets the same openness, and the marriage cap is
    recomputed from the layout's population (as __init__ would do).
    """
    arr = np.array(layout, dtype=np.int8)
    np.random.seed(0)
    model = Schelling(n=arr.shape[0], p=p, m=m)
    model.array = arr.copy()
    model.people, model.people_by_id, model.next_person_id = {}, {}, 0
    for row, col in zip(*np.nonzero(arr)):
        loc = (int(row), int(col))
        person = Person(loc, int(arr[loc]), model.next_person_id)
        person.openness = openness
        model.next_person_id += 1
        model.people[loc] = person
        model.people_by_id[person.id] = person
    model.marriage_limit = int(len(model.people) * m)
    model.marriages_done = 0
    return model


def assert_consistent(model):
    """The grid array and the Person dictionaries must describe the same state."""
    occupied = {(int(r), int(c)) for r, c in zip(*np.nonzero(model.array))}
    assert set(model.people) == occupied
    assert len(model.people_by_id) == len(model.people)
    for loc, person in model.people.items():
        assert person.loc == loc
        assert model.array[loc] == person.group
        assert model.people_by_id[person.id] is person


def group_counts(model):
    return {g: int(np.sum(model.array == g)) for g in range(6)}


def satisfaction_at(model, loc):
    *_, satisfaction = model.count_neighbors()
    return satisfaction[loc]


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def test_initial_grid_states_and_population():
    np.random.seed(1)
    model = Schelling(n=100, p=0.5, m=0.3)
    assert set(np.unique(model.array)) <= {0, 1, 2}
    assert len(model.people) == np.count_nonzero(model.array)
    assert 0.08 < np.mean(model.array == 0) < 0.12          # about 10% empty
    assert len({p.id for p in model.people.values()}) == len(model.people)
    assert model.marriage_limit == int(len(model.people) * 0.3)
    assert model.marriages_done == 0
    assert_consistent(model)


def test_random_openness_uses_nine_levels():
    np.random.seed(2)
    model = Schelling(n=60, p=0.5)
    values = {round(p.openness, 1) for p in model.people.values()}
    assert values <= OPENNESS_LEVELS
    assert len(values) == 9


def test_fixed_openness_applies_to_every_agent():
    np.random.seed(3)
    model = Schelling(n=30, p=0.5, m=0.5, fixed_openness=0.8)
    assert all(p.openness == 0.8 for p in model.people.values())


# ---------------------------------------------------------------------------
# Satisfaction, happiness and segregation
# ---------------------------------------------------------------------------

# Centre cell (2, 2) has neighbours: A, B, AB, AA, BB, B and two empty cells,
# so n_A = 2, n_B = 3, n_AB = 1 and N = 6.  Openness is 0.4.
NEIGHBOURHOOD = [
    [0, 0, 0, 0, 0],
    [0, 1, 2, 5, 0],
    [0, 3, None, 4, 0],
    [0, 0, 2, 0, 0],
    [0, 0, 0, 0, 0],
]


@pytest.mark.parametrize("centre, expected", [
    (1, (2 + 1 + 0.4 * 3) / 6),           # A:  (n_A + n_AB + o n_B) / N
    (2, (3 + 1 + 0.4 * 2) / 6),           # B:  (n_B + n_AB + o n_A) / N
    (3, (2 + 1) / 6),                     # AA: (n_A + n_AB) / N
    (4, (3 + 1) / 6),                     # BB: (n_B + n_AB) / N
    (5, (1.5 * 1 + 0.4 * (2 + 3)) / 6),   # AB: (1.5 n_AB + o (n_A + n_B)) / N
])
def test_satisfaction_formula_for_each_household_type(centre, expected):
    layout = [row[:] for row in NEIGHBOURHOOD]
    layout[2][2] = centre
    model = build(layout, openness=0.4)
    assert satisfaction_at(model, (2, 2)) == pytest.approx(expected)


def test_neighbourhood_wraps_around_the_edges():
    # A at the corner (0, 0) has an A at (0, 1) and a B at (4, 4),
    # which is only a neighbour because the grid is a torus.
    layout = [[0] * 5 for _ in range(5)]
    layout[0][0], layout[0][1], layout[4][4] = 1, 1, 2
    model = build(layout, openness=0.0)
    assert satisfaction_at(model, (0, 0)) == pytest.approx(0.5)


def test_agent_without_neighbours_has_zero_satisfaction():
    layout = [[0] * 5 for _ in range(5)]
    layout[2][2] = 1
    model = build(layout)
    empty, _, _, frac_same, satisfaction = model.count_neighbors()
    assert satisfaction[2, 2] == 0
    assert frac_same[2, 2] == 0
    assert np.isnan(frac_same[0, 0]) and empty[0, 0]   # empty cells are NaN


def test_happy_when_satisfaction_equals_threshold():
    layout = [[0] * 5 for _ in range(5)]
    layout[2][2], layout[2][3], layout[2][1] = 1, 1, 2      # S_A = 1/2 with o = 0
    model = build(layout, p=0.5, openness=0.0)
    *_, satisfaction = model.count_neighbors()
    model.update_happiness(satisfaction)
    assert model.people[(2, 2)].happy is True
    model.p = 0.51
    model.update_happiness(satisfaction)
    assert model.people[(2, 2)].happy is False


def test_segregation_on_known_layouts():
    checkerboard = [[1 if (r + c) % 2 == 0 else 2 for c in range(4)] for r in range(4)]
    assert build(checkerboard).segregation() == pytest.approx(0.5)

    all_a = [[1] * 4 for _ in range(4)]
    all_a[0][0] = 0
    assert build(all_a).segregation() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Relocation
# ---------------------------------------------------------------------------

def test_happy_agents_stay_and_unhappy_agents_move():
    np.random.seed(4)
    model = Schelling(n=30, p=0.6, m=0.0)
    *_, satisfaction = model.count_neighbors()
    model.update_happiness(satisfaction)
    before = {p.id: (p.loc, p.happy) for p in model.people.values()}
    counts_before = group_counts(model)

    model.step()

    for pid, (loc, happy) in before.items():
        new_loc = model.people_by_id[pid].loc
        if happy:
            assert new_loc == loc
        else:
            assert new_loc != loc
    assert group_counts(model) == counts_before        # m = 0: nobody marries
    assert_consistent(model)


def test_full_grid_without_empty_cells_does_not_change():
    checkerboard = [[1 if (r + c) % 2 == 0 else 2 for c in range(4)] for r in range(4)]
    model = build(checkerboard, p=0.9)                   # everyone unhappy
    before = model.array.copy()
    model.step()
    assert np.array_equal(model.array, before)
    assert_consistent(model)


# ---------------------------------------------------------------------------
# Neighbour tracking and marriage eligibility
# ---------------------------------------------------------------------------

def two_neighbours(groups=(1, 1), n=6):
    layout = [[0] * n for _ in range(n)]
    layout[2][2], layout[2][3] = groups
    return layout


def test_neighbour_rounds_increase_by_one_per_step():
    model = build(two_neighbours(), p=0.0)               # p = 0: everyone stays
    first, second = model.people[(2, 2)], model.people[(2, 3)]
    for k in range(1, 5):
        model.step()
        assert first.neighbors[second.id] == k
        assert second.neighbors[first.id] == k


def test_candidates_need_three_consecutive_rounds():
    model = build(two_neighbours(), p=0.0)
    first, second = model.people[(2, 2)], model.people[(2, 3)]
    for _ in range(2):
        model.step()
    assert model.find_candidates(first.id) == []
    model.step()
    assert model.find_candidates(first.id) == [second.id]
    assert model.find_all_candidate_pairs() == [(min(first.id, second.id), max(first.id, second.id))]


def test_unhappy_agents_are_not_candidates():
    model = build(two_neighbours(), p=0.0)
    for _ in range(3):
        model.step()
    first, second = model.people[(2, 2)], model.people[(2, 3)]
    second.happy = False
    assert model.find_candidates(first.id) == []
    assert model.find_candidates(second.id) == []


def test_couples_are_not_candidates():
    model = build(two_neighbours(groups=(1, 3)), p=0.0)  # a single A next to an AA couple
    for _ in range(3):
        model.step()
    assert model.find_all_candidate_pairs() == []


def test_no_marriages_when_m_is_zero():
    np.random.seed(5)
    model = Schelling(n=30, p=0.5, m=0.0)
    for _ in range(30):
        model.step()
    assert model.marriage_limit == 0
    assert model.marriages_done == 0
    assert np.sum(model.array >= 3) == 0


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def chain_of_singles(groups, openness=0.5, m=1.0):
    """A row of singles; possible pairs are all adjacent pairs in the row."""
    layout = [[0] * 8 for _ in range(8)]
    for col, g in enumerate(groups):
        layout[3][col] = g
    model = build(layout, m=m, openness=openness)
    ids = [model.people[(3, c)].id for c in range(len(groups))]
    pairs = [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]
    return model, pairs


def test_nobody_is_selected_twice():
    model, pairs = chain_of_singles([1, 1, 2, 2, 1, 2])
    for seed in range(50):
        np.random.seed(seed)
        chosen = model.choose_marriage_pairs(pairs)
        people = [pid for pair in chosen for pid in pair]
        assert len(people) == len(set(people))
        assert set(chosen) <= set(pairs)


def test_selection_respects_the_marriage_cap():
    model, pairs = chain_of_singles([1, 1, 2, 2, 1, 1])
    model.marriage_limit = model.marriages_done + 1
    np.random.seed(0)
    assert len(model.choose_marriage_pairs(pairs)) <= 1
    model.marriages_done = model.marriage_limit
    assert model.choose_marriage_pairs(pairs) == []


def cross_group_share(openness, draws=600):
    """Share of draws in which the middle A marries the B rather than the other A."""
    model, pairs = chain_of_singles([1, 1, 2], openness=openness)
    same_pair, cross_pair = pairs
    cross = 0
    for seed in range(draws):
        np.random.seed(seed)
        chosen = model.choose_marriage_pairs(pairs)
        cross += cross_pair in chosen
    return cross / draws


def test_cross_group_pairs_are_weighted_by_openness():
    # weight of the A-B pair = o * o, weight of the A-A pair = 1
    assert cross_group_share(0.1) < 0.04                 # expected 0.01 / 1.01, about 1%
    assert 0.40 < cross_group_share(1.0) < 0.60          # expected 1 / 2


@pytest.mark.xfail(raises=ValueError, strict=True,
                   reason="Known limitation: with openness 0 and only cross-group pairs "
                          "available, all selection weights are 0 and np.random.choice "
                          "receives NaN probabilities. Openness must therefore be > 0; "
                          "all experiments use openness >= 0.1.")
def test_zero_openness_with_only_cross_group_pairs():
    model, pairs = chain_of_singles([1, 2], openness=0.0)
    np.random.seed(0)
    with np.errstate(invalid="ignore"):
        model.choose_marriage_pairs(pairs)


# ---------------------------------------------------------------------------
# Household formation
# ---------------------------------------------------------------------------

def test_household_types_new_singles_and_openness():
    layout = [[0] * 8 for _ in range(8)]
    layout[0][0], layout[0][1] = 1, 1        # A + A
    layout[3][0], layout[3][1] = 2, 2        # B + B
    layout[6][0], layout[6][1] = 1, 2        # A + B
    model = build(layout, m=1.0)
    model.people[(6, 0)].openness = 0.3
    model.people[(6, 1)].openness = 0.7
    pair = lambda r: (model.people[(r, 0)].id, model.people[(r, 1)].id)
    pairs = [pair(0), pair(3), pair(6)]
    old_ids = {pid for p in pairs for pid in p}
    occupied_before = np.count_nonzero(model.array)

    np.random.seed(0)
    assert model.complete_marriages(pairs) == 3
    assert model.marriages_done == 3
    assert np.count_nonzero(model.array) == occupied_before
    assert old_ids.isdisjoint(model.people_by_id)
    assert_consistent(model)

    def cells(row):
        return sorted(int(model.array[row, c]) for c in (0, 1))
    assert cells(0) == [1, 3]                # AA couple + new single A
    assert cells(3) == [2, 4]                # BB couple + new single B
    assert 5 in cells(6) and set(cells(6)) - {5} <= {1, 2}

    ab = next(p for p in model.people.values() if p.group == 5)
    assert ab.openness == pytest.approx(0.3)             # min of the partners
    assert set(ab.member_ids) == set(pair_ids for pair_ids in pairs[2])
    aa = next(p for p in model.people.values() if p.group == 3)
    assert aa.openness is None


def test_new_single_gets_fixed_openness():
    model = build([[1, 1, 0], [0, 0, 0], [0, 0, 0]], m=1.0)
    model.fixed_openness = 0.6
    ids = tuple(p.id for p in model.people.values())
    np.random.seed(0)
    model.complete_marriages([ids])
    single = next(p for p in model.people.values() if p.group in (1, 2))
    assert single.openness == 0.6


def test_completion_stops_at_the_marriage_cap_and_skips_invalid_pairs():
    model, pairs = chain_of_singles([1, 1, 2, 2, 1, 1])
    model.marriage_limit = 1
    assert model.complete_marriages([pairs[0], pairs[2], pairs[4]]) == 1

    model, pairs = chain_of_singles([1, 1, 2])
    model.complete_marriages([pairs[0]])
    assert model.complete_marriages([pairs[1]]) == 0     # first person no longer exists


# ---------------------------------------------------------------------------
# Whole runs: invariants and reproducibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    dict(p=0.76, m=0.54),
    dict(p=0.3, m=1.0),
    dict(p=0.76, m=1.0, fixed_openness=1.0),
    dict(p=0.9, m=0.5, fixed_openness=0.1),
])
def test_invariants_hold_over_a_run(kwargs):
    np.random.seed(6)
    model = Schelling(n=30, **kwargs)
    empty_before = np.sum(model.array == 0)
    population = len(model.people)
    for _ in range(40):
        seg = model.step()
        assert 0.0 <= seg <= 1.0
        assert seg == pytest.approx(model.segregation())
        assert np.sum(model.array == 0) == empty_before
        assert len(model.people) == population
        assert np.sum(model.array >= 3) == model.marriages_done   # couples never split
        assert model.marriages_done <= model.marriage_limit
    assert_consistent(model)


def test_same_seed_gives_identical_runs():
    def run():
        np.random.seed(42)
        model = Schelling(n=30, p=0.76, m=0.54)
        segs = [model.step() for _ in range(25)]
        return model.array.copy(), segs, model.marriages_done
    a, b = run(), run()
    assert np.array_equal(a[0], b[0])
    assert a[1] == b[1] and a[2] == b[2]
