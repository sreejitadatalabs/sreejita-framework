def test_scheduler_import():
    from sreejita.automation.scheduler import start_scheduler
    assert callable(start_scheduler)
