from adaptive_memory.data import load_stage_one, task_subset


def test_stage_one_split_is_stratified_and_deterministic() -> None:
    first = load_stage_one(seed=42)
    second = load_stage_one(seed=42)
    assert first.x_train.shape == second.x_train.shape
    assert (first.y_train == second.y_train).all()
    assert set(first.y_train) == set(range(10))


def test_continual_tasks_are_disjoint() -> None:
    bundle = load_stage_one(seed=42)
    task_a = task_subset(bundle, range(0, 5))
    task_b = task_subset(bundle, range(5, 10))
    assert set(task_a.y_train).isdisjoint(set(task_b.y_train))
