from __future__ import annotations

from classifier import MammalCNNClassifier


def test_slug_to_scientific_conversion() -> None:
    assert MammalCNNClassifier._slug_to_scientific("vulpes_vulpes") == "Vulpes vulpes"


def test_species_meta_falls_back_when_csv_missing() -> None:
    clf = MammalCNNClassifier.__new__(MammalCNNClassifier)
    clf._species_lookup = {}
    scientific, nl_name, en_name, tier = clf._resolve_species_meta("meles_meles")
    assert scientific == "Meles meles"
    assert nl_name == "meles meles"
    assert en_name == "Meles meles"
    assert tier == 3
