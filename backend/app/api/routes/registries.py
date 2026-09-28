from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_registry_catalog
from app.ml.registries.catalog import RegistryCatalog
from app.ml.registries.metadata import (
    MetricMetadata,
    ModelMetadata,
    PreprocessorMetadata,
    TunerMetadata,
    ValidationMetadata,
)

router = APIRouter(tags=["registries"])
CatalogDependency = Annotated[RegistryCatalog, Depends(get_registry_catalog)]


@router.get("/models", response_model=list[ModelMetadata])
def list_models(catalog: CatalogDependency) -> tuple[ModelMetadata, ...]:
    """Lista todos os algoritmos e modelos de Machine Learning registrados no catálogo.

    Args:
        catalog (CatalogDependency): Catálogo injetado de componentes ML.

    Returns:
        tuple[ModelMetadata, ...]: Metadados descritivos e capacidades dos modelos.
    """
    return catalog.models.all()


@router.get("/metrics", response_model=list[MetricMetadata])
def list_metrics(catalog: CatalogDependency) -> tuple[MetricMetadata, ...]:
    """Lista todas as métricas de avaliação registradas no catálogo.

    Args:
        catalog (CatalogDependency): Catálogo injetado de componentes ML.

    Returns:
        tuple[MetricMetadata, ...]: Metadados de direção, requisitos e tarefas das métricas.
    """
    return catalog.metrics.all()


@router.get("/preprocessors", response_model=list[PreprocessorMetadata])
def list_preprocessors(catalog: CatalogDependency) -> tuple[PreprocessorMetadata, ...]:
    """Lista todos os transformadores e etapas de pré-processamento disponíveis.

    Args:
        catalog (CatalogDependency): Catálogo injetado de componentes ML.

    Returns:
        tuple[PreprocessorMetadata, ...]: Metadados dos pré-processadores registrados.
    """
    return catalog.preprocessors.all()


@router.get("/validators", response_model=list[ValidationMetadata])
def list_validators(catalog: CatalogDependency) -> tuple[ValidationMetadata, ...]:
    """Lista todas as estratégias de validação e particionamento registradas.

    Args:
        catalog (CatalogDependency): Catálogo injetado de componentes ML.

    Returns:
        tuple[ValidationMetadata, ...]: Metadados das estratégias de split disponíveis.
    """
    return catalog.validators.all()


@router.get("/tuners", response_model=list[TunerMetadata])
def list_tuners(catalog: CatalogDependency) -> tuple[TunerMetadata, ...]:
    """Lista todos os algoritmos e tuners de hiperparâmetros suportados.

    Args:
        catalog (CatalogDependency): Catálogo injetado de componentes ML.

    Returns:
        tuple[TunerMetadata, ...]: Metadados descritivos dos tuners de hiperparâmetros.
    """
    return catalog.tuners.all()
