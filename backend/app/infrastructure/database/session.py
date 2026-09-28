from collections.abc import Iterator
from contextlib import contextmanager
from sqlite3 import Connection as SQLiteConnection

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


class Database:
    """Gerencia a conexão com o banco de dados e a fábrica de sessões SQLAlchemy.

    Attributes:
        engine (Engine): Instância do mecanismo SQLAlchemy conectado ao banco.
        session_factory (sessionmaker): Fábrica configurada para produção de sessões.
    """

    def __init__(self, url: str) -> None:
        """Inicializa o motor de conexão e configura suporte a chaves estrangeiras se SQLite.

        Args:
            url (str): String de conexão JDBC/ODBC do SQLAlchemy.
        """
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine: Engine = create_engine(url, connect_args=connect_args)

        if url.startswith("sqlite"):
            event.listen(self.engine, "connect", self._enable_sqlite_foreign_keys)

        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
        )

    @staticmethod
    def _enable_sqlite_foreign_keys(
        dbapi_connection: SQLiteConnection, connection_record: object
    ) -> None:
        """Ativa pragma de integridade referencial de chaves estrangeiras no SQLite.

        Args:
            dbapi_connection (SQLiteConnection): Conexão direta DBAPI do SQLite.
            connection_record (object): Registro de conexão do pool do SQLAlchemy.
        """
        del connection_record
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Fornece um context manager transacional para operações de banco de dados.

        Yields:
            Iterator[Session]: Sessão transacional aberta do SQLAlchemy.

        Raises:
            Exception: Propaga qualquer exceção após realizar rollback seguro da transação.
        """
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        """Libera e encerra todos os recursos e pools de conexões abertos pelo engine.
        """
        self.engine.dispose()

