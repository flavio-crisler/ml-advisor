from app.domain.workspace.context import LocalPrincipalContextProvider, WorkspaceContext


def test_local_context_is_stable_without_authentication() -> None:
    provider = LocalPrincipalContextProvider()

    assert provider.current() == WorkspaceContext(
        workspace_id="local",
        principal_id="local-user",
    )
    assert provider.current() is provider.current()
