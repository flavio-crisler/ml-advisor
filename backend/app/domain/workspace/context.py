from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class WorkspaceContext:
    """Representa o contexto de isolamento de espaço de trabalho e usuário.

    Attributes:
        workspace_id (str): Identificador do espaço de trabalho ativo.
        principal_id (str): Identificador do usuário ou entidade requisitante.
    """

    workspace_id: str
    principal_id: str


class PrincipalContextProvider(Protocol):
    """Define o protocolo para resolução do contexto de identidade e workspace."""

    def current(self) -> WorkspaceContext:
        """Obtém a instância atual do contexto do espaço de trabalho.

        Returns:
            WorkspaceContext: Instância com as informações de contexto vigentes.
        """
        ...


class LocalPrincipalContextProvider:
    """Fornece um contexto fixo para ambientes locais mono-usuário.

    Attributes:
        _context (WorkspaceContext): Instância estática local de contexto.
    """

    _context = WorkspaceContext(workspace_id="local", principal_id="local-user")

    def current(self) -> WorkspaceContext:
        """Retorna o contexto local padrão configurado.

        Returns:
            WorkspaceContext: Contexto local fixo de workspace.
        """
        return self._context
