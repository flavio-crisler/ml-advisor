import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import shap
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from app.domain.experiment.spec import TaskType
from app.ml.models.base import ModelAdapter


class ResultArtifactGenerator:
    """Gerador de artefatos visuais e analíticos (gráficos Plotly e SHAP) para experimentos.

    Attributes:
        max_explanation_rows (int): Quantidade máxima de linhas amostradas para explicabilidade.
        max_background_rows (int): Quantidade máxima de linhas usadas como base neutra no SHAP.
    """

    def __init__(self, *, max_explanation_rows: int = 24, max_background_rows: int = 50):
        """Inicializa o gerador configurando os limites de amostragem para explicabilidade.

        Args:
            max_explanation_rows: Limite de linhas avaliadas pelo SHAP. Padrão 24.
            max_background_rows: Limite de linhas de referência de fundo no SHAP. Padrão 50.
        """
        self.max_explanation_rows = max_explanation_rows
        self.max_background_rows = max_background_rows

    def build_run(
        self,
        *,
        task: TaskType,
        target: Sequence[Any],
        predictions: Sequence[Any],
        probabilities: Sequence[Sequence[float]] | None,
        probability_classes: Sequence[Any] | None,
        positive_label: Any | None,
        fitted_features: pd.DataFrame,
        model: ModelAdapter,
        random_seed: int,
        learning_curve: Sequence[Mapping[str, float | int]],
    ) -> dict[str, Any]:
        """Constrói os gráficos e dados de explicabilidade para uma execução de modelo.

        Args:
            task: Tipo da tarefa de machine learning (regressão ou classificação).
            target: Valores observados reais da variável alvo.
            predictions: Valores preditos pelo modelo.
            probabilities: Matriz de probabilidades preditas (opcional para classificação).
            probability_classes: Classes correspondentes às colunas de probabilidade.
            positive_label: Rótulo da classe positiva para métricas binárias.
            fitted_features: DataFrame com as features transformadas usadas no ajuste.
            model: Adaptador do modelo treinado para predições explicativas.
            random_seed: Semente pseudoaleatória para amostragem determinística.
            learning_curve: Sequência de pontos medidos durante o aprendizado do modelo.

        Returns:
            Dicionário com schema de versão, gráficos gerados e dados SHAP de explicabilidade.
        """
        target_values = np.asarray(target)
        predicted_values = np.asarray(predictions)
        plots = (
            self._regression_plots(target_values, predicted_values)
            if task == TaskType.REGRESSION
            else self._classification_plots(
                target_values,
                predicted_values,
                np.asarray(probabilities) if probabilities is not None else None,
                tuple(probability_classes or ()),
                positive_label,
            )
        )
        if learning_curve:
            plots["learning_curve"] = self._learning_curve(learning_curve)

        explainability = self._explain(
            task=task,
            features=fitted_features,
            model=model,
            random_seed=random_seed,
        )
        if explainability["status"] == "available":
            importance = explainability["feature_importance"]
            plots["feature_importance"] = self._feature_importance(importance)
            plots["shap_summary"] = self._shap_summary(
                explainability["shap_series"],
                [str(item["feature"]) for item in importance],
            )

        return {
            "schema_version": "1.0",
            "task": task.value,
            "plots": plots,
            "explainability": explainability,
        }

    def build_comparison(
        self,
        *,
        primary_metric: str,
        runs: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        """Constrói o gráfico comparativo de barras entre múltiplos modelos pelo primary metric.

        Args:
            primary_metric: Identificador da métrica primária usada na comparação.
            runs: Sequência de dados das execuções contendo identificadores e métricas.

        Returns:
            Dicionário com especificação do gráfico serializado de comparação.
        """
        model_ids = [str(run["model_id"]) for run in runs]
        values = [float(run["metrics"][primary_metric]) for run in runs]
        figure = go.Figure(
            go.Bar(x=model_ids, y=values, customdata=[run["run_id"] for run in runs])
        )
        figure.update_layout(meta={"title_key": "results.modelComparison"})
        return {
            "schema_version": "1.0",
            "primary_metric": primary_metric,
            "plot": self._figure(figure),
        }

    def _classification_plots(
        self,
        target: np.ndarray,
        predictions: np.ndarray,
        probabilities: np.ndarray | None,
        classes: tuple[Any, ...],
        positive_label: Any | None,
    ) -> dict[str, Any]:
        """Gera os gráficos analíticos específicos para tarefas de classificação.

        Produz a matriz de confusão, curva ROC, curva de precisão/recall e distribuição
        das probabilidades por classe.

        Args:
            target: Array com os rótulos reais.
            predictions: Array com as predições de classes.
            probabilities: Matriz de probabilidades preditas ou None.
            classes: Tupla com os nomes ou rótulos das classes.
            positive_label: Rótulo da classe positiva para classificação binária.

        Returns:
            Dicionário mapeando os gráficos aos seus objetos Plotly serializados.
        """
        labels = classes or tuple(np.unique(np.concatenate([target, predictions])).tolist())
        matrix = confusion_matrix(target, predictions, labels=list(labels))
        confusion = go.Figure(
            go.Heatmap(
                z=matrix.tolist(),
                x=[str(item) for item in labels],
                y=[str(item) for item in labels],
                text=matrix.tolist(),
                texttemplate="%{text}",
                hovertemplate="%{y} → %{x}: %{z}<extra></extra>",
            )
        )
        confusion.update_layout(meta={"title_key": "results.confusionMatrix"})
        plots: dict[str, Any] = {"confusion_matrix": self._figure(confusion)}
        if probabilities is None or not classes:
            return plots

        roc = go.Figure()
        precision_recall = go.Figure()
        if len(classes) == 2:
            positive_index = self._positive_index(classes, positive_label)
            binary_target = target == classes[positive_index]
            fpr, tpr, _ = roc_curve(binary_target, probabilities[:, positive_index])
            precision, recall, _ = precision_recall_curve(
                binary_target, probabilities[:, positive_index]
            )
            average_precision = average_precision_score(
                binary_target, probabilities[:, positive_index]
            )
            roc.add_scatter(
                x=fpr.tolist(),
                y=tpr.tolist(),
                mode="lines",
                name=f"{classes[positive_index]} · AUC={auc(fpr, tpr):.3f}",
            )
            precision_recall.add_scatter(
                x=recall.tolist(),
                y=precision.tolist(),
                mode="lines",
                name=(f"{classes[positive_index]} · AP={average_precision:.3f}"),
            )
            self._add_pr_baseline(
                precision_recall,
                label=str(classes[positive_index]),
                prevalence=float(binary_target.mean()),
            )
        else:
            binary_target = label_binarize(target, classes=list(classes))
            for index, label in enumerate(classes):
                fpr, tpr, _ = roc_curve(binary_target[:, index], probabilities[:, index])
                precision, recall, _ = precision_recall_curve(
                    binary_target[:, index], probabilities[:, index]
                )
                average_precision = average_precision_score(
                    binary_target[:, index], probabilities[:, index]
                )
                roc.add_scatter(
                    x=fpr.tolist(),
                    y=tpr.tolist(),
                    mode="lines",
                    name=f"{label} · AUC={auc(fpr, tpr):.3f}",
                )
                precision_recall.add_scatter(
                    x=recall.tolist(),
                    y=precision.tolist(),
                    mode="lines",
                    name=f"{label} · AP={average_precision:.3f}",
                )
                self._add_pr_baseline(
                    precision_recall,
                    label=str(label),
                    prevalence=float(binary_target[:, index].mean()),
                )
        roc.add_scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="baseline",
            line={"dash": "dash", "color": "#64748b"},
            meta={"role": "baseline"},
        )
        roc.update_layout(meta={"title_key": "results.rocCurve"})
        precision_recall.update_layout(meta={"title_key": "results.precisionRecallCurve"})

        probability_distribution = go.Figure()
        for index, label in enumerate(classes):
            probability_distribution.add_histogram(
                x=probabilities[:, index].tolist(),
                name=str(label),
                opacity=0.65,
            )
        probability_distribution.update_layout(
            barmode="overlay",
            meta={"title_key": "results.probabilityDistribution"},
        )
        plots.update(
            {
                "roc_curve": self._figure(roc),
                "precision_recall_curve": self._figure(precision_recall),
                "probability_distribution": self._figure(probability_distribution),
            }
        )
        return plots

    def _regression_plots(self, target: np.ndarray, predictions: np.ndarray) -> dict[str, Any]:
        """Gera os gráficos analíticos específicos para tarefas de regressão.

        Produz os gráficos de dispersão Real vs Predito, dispersão de Resíduos e
        histograma da distribuição dos resíduos.

        Args:
            target: Array contendo os valores contínuos reais observados.
            predictions: Array contendo os valores contínuos previstos pelo modelo.

        Returns:
            Dicionário com os gráficos serializados de regressão.
        """
        numeric_target = target.astype(float)
        numeric_predictions = predictions.astype(float)
        residuals = numeric_target - numeric_predictions
        lower = float(min(numeric_target.min(), numeric_predictions.min()))
        upper = float(max(numeric_target.max(), numeric_predictions.max()))

        actual = go.Figure()
        actual.add_scatter(
            x=numeric_target.tolist(),
            y=numeric_predictions.tolist(),
            mode="markers",
        )
        actual.add_scatter(x=[lower, upper], y=[lower, upper], mode="lines")
        actual.update_layout(meta={"title_key": "results.actualVsPredicted"})

        residual = go.Figure(
            go.Scatter(x=numeric_predictions.tolist(), y=residuals.tolist(), mode="markers")
        )
        residual.add_hline(y=0)
        residual.update_layout(meta={"title_key": "results.residualPlot"})

        distribution = go.Figure(go.Histogram(x=residuals.tolist()))
        distribution.update_layout(meta={"title_key": "results.residualDistribution"})
        return {
            "actual_vs_predicted": self._figure(actual),
            "residual_plot": self._figure(residual),
            "residual_distribution": self._figure(distribution),
        }

    def _explain(
        self,
        *,
        task: TaskType,
        features: pd.DataFrame,
        model: ModelAdapter,
        random_seed: int,
    ) -> dict[str, Any]:
        """Computa métricas e gráficos de explicabilidade utilizando SHAP Permutation.

        Amostra uma porção dos dados para avaliação e uma base neutra de background,
        executando o cálculo de valores SHAP e ranking de importância média.

        Args:
            task: Tipo da tarefa (regressão ou classificação).
            features: DataFrame com as features de entrada pós-transformação.
            model: Adaptador do modelo treinado para gerar previsões explicativas.
            random_seed: Semente pseudoaleatória para amostragem reprodutível.

        Returns:
            Dicionário com o status ('available' ou 'unavailable') e dados SHAP.
        """
        if features.empty or not len(features.columns):
            return {"status": "unavailable", "reason": "empty_feature_matrix"}
        background = features.sample(
            n=min(len(features), self.max_background_rows),
            random_state=random_seed,
        )
        explained = features.sample(
            n=min(len(features), self.max_explanation_rows),
            random_state=random_seed,
        )
        columns = list(features.columns)
        dtypes = features.dtypes

        def predict(values: Any) -> np.ndarray:
            frame = pd.DataFrame(values, columns=columns)
            for column in columns:
                if isinstance(dtypes[column], pd.CategoricalDtype):
                    original = features[column].dtype
                    frame[column] = pd.Categorical(frame[column], categories=original.categories)
            output = (
                model.predict(features=frame)
                if task == TaskType.REGRESSION
                else model.predict_proba(features=frame)
            )
            return np.asarray(output)

        try:
            explainer = shap.Explainer(
                predict,
                background,
                algorithm="permutation",
                feature_names=columns,
                seed=random_seed,
            )
            explanation = explainer(
                explained,
                max_evals=max(2 * len(columns) + 1, 10),
                silent=True,
            )
            values = np.asarray(explanation.values, dtype=float)
            axes = tuple(index for index in range(values.ndim) if index != 1)
            importance_values = np.mean(np.abs(values), axis=axes)
            ranking = sorted(
                zip(columns, importance_values.tolist(), strict=True),
                key=lambda item: (-item[1], item[0]),
            )
            shap_series = self._shap_series(
                values=values,
                features=explained,
                model=model,
                task=task,
            )
            return {
                "status": "available",
                "method": "shap_permutation",
                "sample_rows": len(explained),
                "feature_importance": [
                    {"feature": feature, "mean_absolute_shap": float(value)}
                    for feature, value in ranking
                ],
                "shap_series": shap_series,
            }
        except Exception as error:
            return {
                "status": "unavailable",
                "reason": "shap_incompatible",
                "detail": type(error).__name__,
            }

    @staticmethod
    def _feature_importance(importance: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Constrói gráfico de barras horizontais com a importância média SHAP das features.

        Args:
            importance: Sequência contendo nome da feature e valor de mean_absolute_shap.

        Returns:
            Dicionário da figura Plotly serializada para JSON.
        """
        figure = go.Figure(
            go.Bar(
                x=[float(item["mean_absolute_shap"]) for item in importance],
                y=[str(item["feature"]) for item in importance],
                orientation="h",
            )
        )
        figure.update_layout(meta={"title_key": "results.featureImportance"})
        return ResultArtifactGenerator._figure(figure)

    @staticmethod
    def _shap_series(
        *,
        values: np.ndarray,
        features: pd.DataFrame,
        model: ModelAdapter,
        task: TaskType,
    ) -> list[dict[str, Any]]:
        """Converte matriz de valores SHAP em séries de coordenadas estruturadas para Plotly.

        Args:
            values: Matriz multidimensional de valores SHAP gerada pelo explicador.
            features: DataFrame com as features avaliadas na explicação.
            model: Adaptador do modelo treinado.
            task: Tipo da tarefa (regressão ou classificação).

        Returns:
            Lista de dicionários contendo nome da série e coordenadas x e y.

        Raises:
            ValueError: Se os valores SHAP possuírem dimensões inesperadas.
        """
        columns = list(features.columns)
        if values.ndim == 2:
            outputs = [("SHAP", values)]
        elif values.ndim == 3:
            classes = tuple(model.get_classes() or ()) if task != TaskType.REGRESSION else ()
            names = [str(item) for item in classes]
            if len(names) != values.shape[2]:
                names = [f"output_{index}" for index in range(values.shape[2])]
            outputs = [(name, values[:, :, index]) for index, name in enumerate(names)]
        else:
            raise ValueError("Unexpected SHAP value dimensions")
        return [
            {
                "name": name,
                "x": matrix.reshape(-1).tolist(),
                "y": [column for _ in range(len(matrix)) for column in columns],
            }
            for name, matrix in outputs
        ]

    @staticmethod
    def _shap_summary(
        series: Sequence[Mapping[str, Any]],
        feature_order: Sequence[str],
    ) -> dict[str, Any]:
        """Constrói gráfico de dispersão com a distribuição dos valores SHAP por feature.

        Args:
            series: Sequência de séries contendo os pontos SHAP calculados.
            feature_order: Ordem de exibição das features de acordo com o ranking.

        Returns:
            Dicionário com o gráfico Plotly do resumo SHAP serializado.
        """
        figure = go.Figure()
        for item in series:
            figure.add_scatter(
                x=list(item["x"]),
                y=list(item["y"]),
                mode="markers",
                name=str(item["name"]),
                marker={"size": 6, "opacity": 0.55},
                hovertemplate="%{y}: %{x:.4f}<extra>%{fullData.name}</extra>",
            )
        figure.add_vline(x=0, line_width=1, line_dash="dot", line_color="#64748b")
        figure.update_layout(
            meta={"title_key": "results.shapSummary"},
            yaxis={"categoryorder": "array", "categoryarray": list(reversed(feature_order))},
        )
        return ResultArtifactGenerator._figure(figure)

    @staticmethod
    def _add_pr_baseline(
        figure: go.Figure,
        *,
        label: str,
        prevalence: float,
    ) -> None:
        """Adiciona linha de referência correspondente à prevalência da classe na curva PR.

        Args:
            figure: Objeto Figure do Plotly onde a linha será adicionada.
            label: Rótulo da classe correspondente.
            prevalence: Prevalência da classe no conjunto avaliado (proporção positiva).
        """
        figure.add_scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"{label} · baseline",
            line={"dash": "dot", "width": 1},
            meta={"role": "baseline", "label": label},
        )

    @staticmethod
    def _learning_curve(points: Sequence[Mapping[str, float | int]]) -> dict[str, Any]:
        """Constrói o gráfico da curva de aprendizado com relação entre amostras e métrica.

        Args:
            points: Sequência de pontos com 'training_rows' e 'validation_value'.

        Returns:
            Dicionário com o gráfico Plotly serializado.
        """
        figure = go.Figure(
            go.Scatter(
                x=[int(point["training_rows"]) for point in points],
                y=[float(point["validation_value"]) for point in points],
                mode="lines+markers",
            )
        )
        figure.update_layout(meta={"title_key": "results.learningCurve"})
        return ResultArtifactGenerator._figure(figure)

    @staticmethod
    def _positive_index(classes: tuple[Any, ...], positive_label: Any | None) -> int:
        """Localiza o índice da classe positiva dentro da tupla de classes.

        Args:
            classes: Tupla com os rótulos de todas as classes.
            positive_label: Identificador do rótulo da classe positiva.

        Returns:
            Índice posicional inteiro da classe positiva.

        Raises:
            ValueError: Se o positive_label não for encontrado em classes.
        """
        if positive_label is None:
            return 1
        for index, label in enumerate(classes):
            if label == positive_label or str(label) == str(positive_label):
                return index
        raise ValueError("Positive label is not present in probability classes")

    @staticmethod
    def _figure(figure: go.Figure) -> dict[str, Any]:
        """Converte uma figura Plotly em um dicionário compatível com serialização JSON.

        Args:
            figure: Instância go.Figure a ser serializada.

        Returns:
            Dicionário com a árvore completa de dados e layout da figura.
        """
        return json.loads(pio.to_json(figure, validate=True, pretty=False))
