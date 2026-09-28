class AdvisorError(RuntimeError):
    """Exceção base para erros ocorridos durante o fluxo do Advisor de ML.

    Attributes:
        code (str): Código identificador padronizado do erro.
    """

    code = "advisor_error"


class AdvisorRecommendationError(AdvisorError):
    """Exceção levantada quando uma recomendação gerada pelo Advisor é inválida.

    Attributes:
        issues (tuple[str, ...]): Lista de inconsistências identificadas na recomendação.
    """

    code = "invalid_advisor_recommendation"

    def __init__(self, issues: list[str]) -> None:
        """Inicializa o erro com a lista de problemas encontrados na recomendação.

        Args:
            issues: Lista de mensagens descrevendo as inconsistências da recomendação.
        """
        super().__init__("; ".join(issues))
        self.issues = tuple(issues)
