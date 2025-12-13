def test_batch_runner_import():
    from sreejita.automation.batch_runner import run_batch
    assert callable(run_batch)


def test_single_file_runner_import():
    from sreejita.automation.batch_runner import run_single_file
    assert callable(run_single_file)
