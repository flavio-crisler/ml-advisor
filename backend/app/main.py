import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.dependencies import (
    get_decision_job_executor,
    get_decision_run_repository,
    get_job_executor,
    get_semantic_job_executor,
)
from app.api.router import api_router
from app.application.jobs.executor import LocalProcessExecutor
from app.core.config.settings import get_settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Gerencia o ciclo de vida da aplicação FastAPI, inicializando e encerrando recursos.

    No startup, recupera execuções de benchmark de decisão interrompidas.
    No shutdown, encerra graciosamente os executores de jobs em processos locais.

    Args:
        _: Instância da aplicação FastAPI.
    """
    recovered = get_decision_run_repository().fail_interrupted()
    if recovered:
        logger.warning("Marked %s interrupted decision runs as failed", recovered)
    yield
    for executor in (
        get_job_executor(),
        get_decision_job_executor(),
        get_semantic_job_executor(),
    ):
        if isinstance(executor, LocalProcessExecutor):
            executor.shutdown()


def create_app() -> FastAPI:
    """Instancia e configura a aplicação FastAPI com middlewares, rotas e logging.

    Returns:
        FastAPI: Instância configurada pronta para execução.
    """
    settings = get_settings()
    configure_logging(settings.log_level)
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(api_router, prefix="/api/v1")
    return application


app = create_app()
