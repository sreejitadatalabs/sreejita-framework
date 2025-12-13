def test_batch_runner_import():
    from sreejita.automation.batch_runner import run_batch
    assert callable(run_batch)
