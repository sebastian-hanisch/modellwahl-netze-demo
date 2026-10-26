"""Gemeinsame Fixtures: kleine Netze, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def path4():
    """Weg 0–1–2–3. Grade 1, 2, 2, 1; keine Dreiecke; Assortativität −1/2; mittlere Weglänge 5/3; beste Zerlegung {0,1} | {2,3} mit Modularität 1/6."""
    return 4, [(0, 1), (1, 2), (2, 3)]


@pytest.fixture
def paw():
    """Dreieck 0–1–2 mit Anhängsel 3 an Knoten 2. Grade 2, 2, 3, 1; ein Dreieck, fünf verbundene Dreiergruppen: Transitivität 3/5."""
    return 4, [(0, 1), (0, 2), (1, 2), (2, 3)]


@pytest.fixture
def two_triangles():
    """Zwei getrennte Dreiecke 0–1–2 und 3–4–5 plus ein einzelner Knoten 6."""
    return 7, [(0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)]


@pytest.fixture
def barbell():
    """Zwei Dreiecke 0–1–2 und 3–4–5, verbunden durch die Kante 2–3."""
    return 6, [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (3, 5), (4, 5)]
