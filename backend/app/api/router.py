"""Agregador central de rotas da API v1 da aplicação."""

from fastapi import APIRouter

from app.api.routes.advisor import router as advisor_router
from app.api.routes.auto import router as auto_router
from app.api.routes.confidence_cascades import router as confidence_cascades_router
from app.api.routes.datasets import router as datasets_router
from app.api.routes.decision_providers import router as decision_providers_router
from app.api.routes.decision_runs import router as decision_runs_router
from app.api.routes.experiment_specs import router as experiment_specs_router
from app.api.routes.experiments import router as experiments_router
from app.api.routes.explanations import router as explanations_router
from app.api.routes.health import router as health_router
from app.api.routes.improvement_proposals import router as improvement_proposals_router
from app.api.routes.registries import router as registries_router
from app.api.routes.semantic_features import router as semantic_features_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(advisor_router)
api_router.include_router(auto_router)
api_router.include_router(confidence_cascades_router)
api_router.include_router(datasets_router)
api_router.include_router(decision_providers_router)
api_router.include_router(decision_runs_router)
api_router.include_router(semantic_features_router)
api_router.include_router(registries_router)
api_router.include_router(experiment_specs_router)
api_router.include_router(experiments_router)
api_router.include_router(explanations_router)
api_router.include_router(improvement_proposals_router)
