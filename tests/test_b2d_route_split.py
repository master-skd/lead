from scripts.p4.make_b2d_route_split import split_routes


def test_route_split_has_no_leakage_and_is_deterministic():
    entries = [
        {"scenario": scenario, "route": f"r{route}"}
        for scenario in ("a", "b")
        for route in range(10)
        for _ in range(3)
    ]
    train_a, heldout_a = split_routes(entries, 0.2, 17)
    train_b, heldout_b = split_routes(entries, 0.2, 17)
    assert (train_a, heldout_a) == (train_b, heldout_b)
    assert not train_a & heldout_a
    assert len(heldout_a) == 4
    for entry in entries:
        route = (entry["scenario"], entry["route"])
        assert (route in train_a) != (route in heldout_a)


def test_single_route_scenario_remains_in_training():
    entries = [{"scenario": "rare", "route": "only"}]
    train, heldout = split_routes(entries, 0.5, 1)
    assert train == {("rare", "only")}
    assert not heldout
