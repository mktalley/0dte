import os
import logging
import importlib
from datetime import datetime
from zoneinfo import ZoneInfo
import pytest

# Ensure we import src.main inside tests, so datetime patching works
@pytest.fixture(autouse=True)
def patch_datetime(monkeypatch):
    """
    Patch datetime.now in src.main to a fixed date 2021-01-02
    """
    # Create a FakeDatetime class that overrides now()
    class FakeDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            # Return a fixed date: 2021-01-02 at midnight in given timezone
            return cls(2021, 1, 2, tzinfo=tz)

    # Reload src.main to ensure fresh state
    if 'src.main' in importlib.sys.modules:
        importlib.reload(importlib.import_module('src.main'))
    main_mod = importlib.import_module('src.main')
    # Patch the datetime in the module
    monkeypatch.setattr(main_mod, 'datetime', FakeDatetime)
    yield


def test_should_rollover_date_change(tmp_path):
    # Import after patch
    import src.main as main
    timezone = ZoneInfo('UTC')
    # Use tmp_path as logs directory
    handler = main.DailyRotatingFileHandler(
        orig_filename='test.log', when='midnight',
        interval=1, backupCount=0, timezone=timezone,
        log_dir=str(tmp_path)
    )
    # Simulate that the handler's current_date is stale
    handler.current_date = '2021-01-01'
    # Create a dummy log record
    record = logging.LogRecord(
        name='test', level=logging.INFO,
        pathname=__file__, lineno=1,
        msg='message', args=(), exc_info=None
    )
    assert handler.shouldRollover(record) is True


def test_should_not_rollover_same_date(tmp_path):
    import src.main as main
    timezone = ZoneInfo('UTC')
    handler = main.DailyRotatingFileHandler(
        orig_filename='test.log', when='midnight',
        interval=1, backupCount=0, timezone=timezone,
        log_dir=str(tmp_path)
    )
    # Handler.current_date should be 2021-01-02 per FakeDatetime
    record = logging.LogRecord(
        name='test', level=logging.INFO,
        pathname=__file__, lineno=1,
        msg='message', args=(), exc_info=None
    )
    assert handler.shouldRollover(record) is False


def test_do_rollover_updates_paths(tmp_path, monkeypatch):
    import src.main as main
    from logging.handlers import TimedRotatingFileHandler
    # Stub the base class doRollover to avoid file operations
    monkeypatch.setattr(TimedRotatingFileHandler, 'doRollover', lambda self: None)

    timezone = ZoneInfo('UTC')
    handler = main.DailyRotatingFileHandler(
        orig_filename='test.log', when='midnight',
        interval=1, backupCount=0, timezone=timezone,
        log_dir=str(tmp_path)
    )
    # Prepare old date folder and file
    old_date = '2021-01-01'
    old_dir = tmp_path / old_date
    os.makedirs(old_dir, exist_ok=True)
    # Force current_date to the old date
    handler.current_date = old_date
    handler.log_dir_today = str(old_dir)
    old_file_path = old_dir / 'test.log'
    old_file_path.write_text('old content')

    # Call rollover
    handler.doRollover()

    # After rollover, current_date should update to FakeDatetime.now = 2021-01-02
    assert handler.current_date == '2021-01-02'
    # New folder should be created
    new_dir = tmp_path / '2021-01-02'
    assert new_dir.exists(), "New date directory was not created"
    # baseFilename should point to new file in new directory
    expected_path = str(new_dir / 'test.log')
    assert handler.baseFilename == expected_path
    # Write a new log entry to the stream and verify file content
    log_record = logging.LogRecord(
        name='test', level=logging.INFO,
        pathname=__file__, lineno=1,
        msg='new log', args=(), exc_info=None
    )
    handler.emit(log_record)
    handler.stream.close()
    content = (new_dir / 'test.log').read_text()
    assert 'new log' in content
