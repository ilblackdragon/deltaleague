"""Generate matched reasoning steps with exact labels and disjoint contexts."""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random


@dataclass
class Pair:
    contexts: list[str]
    answers: list[str]


def modular(rng):
    divisor = rng.randint(3, 9)
    a, b = rng.randint(5, 65), rng.randint(3, 24)
    offsets = rng.sample(range(divisor), 2)
    context = lambda c: f"Compute the remainder of ({a} times {b} plus {c}) upon division by {divisor}. After the division, the remainder is"
    return Pair([context(c) for c in offsets], [str((a*b+c) % divisor) for c in offsets])


def equation(rng):
    coefficient, offset = rng.randint(2, 19), rng.randint(-12, 24)
    roots = rng.sample(range(10), 2)
    context = lambda x: f"Solve {coefficient}x {offset:+d} = {coefficient*x+offset} for x. Isolating the variable gives x ="
    return Pair([context(x) for x in roots], list(map(str, roots)))


def inequality(rng):
    a, b, c = rng.randint(2, 12), rng.randint(2, 20), rng.randint(1, 12)
    lhs = a*b-c
    thresholds = [lhs-rng.randint(1, 6), lhs+rng.randint(1, 6)]
    small = rng.choice([True, False])
    relation = 'less' if small else 'greater'
    context = lambda t: f"Consider the claim that {a} times {b} minus {c} is strictly {relation} than {t}. After evaluating both sides, the claim is"
    return Pair([context(t) for t in thresholds], [str(lhs<t if small else lhs>t).lower() for t in thresholds])


def boolean(rng):
    names = rng.sample(['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'], 3)
    a, b = [rng.choice([True, False]) for _ in range(2)]
    op = rng.choice(['and', 'or'])
    nested = (a and b) if op == 'and' else (a or b)
    same = rng.choice([True, False])
    comparison = 'has the same truth value as' if same else 'has a different truth value from'
    p, q, r = names
    context = lambda c: f"Let {p} be {str(a).lower()}, {q} be {str(b).lower()}, and {r} be {str(c).lower()}. The claim is that ({p} {op} {q}) {comparison} {r}. The truth value of the claim is"
    return Pair([context(c) for c in [False, True]], [str((nested==c)==same).lower() for c in [False, True]])


def spatial(rng):
    center, first, other = rng.sample(['Ada', 'Ben', 'Cara', 'Dan', 'Eli', 'Fay', 'Gus', 'Hana', 'Ivy', 'Jo'], 3)
    direction = rng.choice(['left', 'right'])
    context = lambda side: f"{first}, {center} and {other} occupy three seats in a row. {center} sits in the middle. {first} is immediately to the {side} of {center}. The claim that {other} is to the {direction} of {first} is"
    return Pair([context(side) for side in ['left', 'right']], [str(direction!=side).lower() for side in ['left', 'right']])


def permutation(rng):
    values = rng.sample(range(10), 3)
    i, j, k = rng.sample(range(3), 3)
    slots = ['first', 'second', 'third']
    prompts, answers = [], []
    for last in [j, k]:
        prompts.append(f"A row of tiles reads {values[0]}, {values[1]}, {values[2]} from left to right. Swap the {slots[j]} and {slots[k]} tiles, then swap the {slots[i]} and {slots[last]} tiles. The number now in the {slots[i]} position is")
        row = values.copy()
        row[j], row[k] = row[k], row[j]
        row[i], row[last] = row[last], row[i]
        answers.append(str(row[i]))
    return Pair(prompts, answers)


FAMILIES = [modular, equation, inequality, boolean, spatial, permutation]


def prepare(out):
    rng = random.Random(20261123)
    occupied = set()
    data = {}
    for split, per_family in [('train', 16), ('diagnostic', 4)]:
        rows = []
        for family in FAMILIES:
            while sum(r['family']==family.__name__ for r in rows) < per_family:
                pair = family(rng)
                assert len(set(pair.contexts)) == len(set(pair.answers)) == 2
                if occupied.intersection(pair.contexts):
                    continue
                occupied.update(pair.contexts)
                order = rng.sample(range(2), 2)
                rows.append(dict(family=family.__name__, contexts=[pair.contexts[i] for i in order],
                                 answers=[pair.answers[i] for i in order]))
        rng.shuffle(rows)
        data[split] = rows
    assert len(occupied) == 240
    content = json.dumps(data, indent=2) + '\n'
    out.mkdir(parents=True, exist_ok=True)
    path = out/'pairs.json'
    if path.exists() and path.read_text() != content:
        raise ValueError('Existing dataset differs')
    path.write_text(content)
    print(hashlib.sha256(content.encode()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    prepare(parser.parse_args().out)
