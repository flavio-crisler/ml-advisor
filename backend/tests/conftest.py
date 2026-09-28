import os
import shutil
import sqlite3
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

from app.infrastructure.database.session import Database

_test_database_root = Path(tempfile.mkdtemp(prefix="ml-advisor-pytest-"))


def _migrate(database_path: Path) -> None:
    migration_directory = Path(__file__).parents[1] / "migrations"
    connection = sqlite3.connect(database_path)
    try:
        for migration in sorted(migration_directory.glob("V*__*.sql")):
            connection.executescript(migration.read_text(encoding="utf-8"))
        connection.commit()
    finally:
        connection.close()


def pytest_configure() -> None:
    """Give application startup a migrated database isolated from runtime data."""
    database_path = _test_database_root / "app.db"
    os.environ["DATABASE_URL"] = f"sqlite:///{database_path}"
    _migrate(database_path)


def pytest_unconfigure() -> None:
    shutil.rmtree(_test_database_root)


@pytest.fixture
def migrated_database(tmp_path: Path) -> Iterator[Database]:
    database_path = tmp_path / "app.db"
    _migrate(database_path)

    database = Database(f"sqlite:///{database_path}")
    yield database
    database.dispose()
