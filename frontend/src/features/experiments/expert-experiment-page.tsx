import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type FormEvent, useState } from "react";
import { useTranslation } from "react-i18next";
import { useNavigate, useParams } from "react-router-dom";

import { WorkspaceLayout } from "../../components/workspace-header";
import { Button } from "../../components/ui/button";
import { Card } from "../../components/ui/card";
import { StatePanel } from "../../components/ui/state-panel";
import { WorkflowSteps } from "../../components/workflow-steps";
import { HowItWorksButton } from "../../components/how-it-works";
import { getDataset, type DatasetDetail } from "../../services/datasets";
import { describeExperimentIssues } from "../../services/experiment-validation";
import {
  createExperiment,
  ExperimentApiError,
  type ExperimentSpec,
} from "../../services/experiments";
import {
  getRegistryCatalog,
  type RegistryCatalog,
} from "../../services/registries";

type Task = ExperimentSpec["task"];
type ParameterMap = Record<string, string>;
type PreprocessorSelection = Record<
  string,
  { enabled: boolean; strategy: string; parameters: string }
>;

const inputClass =
  "h-10 w-full rounded-md border border-border bg-surface px-3 text-sm focus:border-primary focus:outline-none focus:ring-2 focus:ring-primary/20";

/**
 * Página container do modo especialista para configuração detalhada de experimentos.
 *
 * @returns Layout do workspace com verificação de carregamento e formulário especialista.
 */
export function ExpertExperimentPage() {
  const { datasetId } = useParams();
  const { t } = useTranslation("expert");
  const dataset = useQuery({
    queryKey: ["datasets", datasetId],
    queryFn: () => getDataset(datasetId!),
    enabled: Boolean(datasetId),
  });
  const catalog = useQuery({
    queryKey: ["registry-catalog"],
    queryFn: getRegistryCatalog,
  });

  if (dataset.isPending || catalog.isPending) {
    return (
      <WorkspaceLayout>
        <StatePanel title={t("loading")} />
      </WorkspaceLayout>
    );
  }
  if (!dataset.data || !catalog.data || dataset.isError || catalog.isError) {
    return (
      <WorkspaceLayout>
        <StatePanel
          title={t("errors.load")}
          description={t("errors.loadDescription")}
          tone="error"
          actionLabel={t("errors.retry")}
          onAction={() => {
            void dataset.refetch();
            void catalog.refetch();
          }}
        />
      </WorkspaceLayout>
    );
  }
  return (
    <WorkspaceLayout>
      <ExpertExperimentForm detail={dataset.data} catalog={catalog.data} />
    </WorkspaceLayout>
  );
}

/**
 * Formulário interativo para controle total dos parâmetros de experimentos (target, features, modelos, métricas e tuners).
 *
 * @param props.detail - Metadados e perfil estatístico do dataset selecionado.
 * @param props.catalog - Catálogo de modelos, algoritmos e pré-processadores registrados.
 * @returns Formulário com seções colapsáveis para cada estágio do pipeline de Machine Learning.
 */
export function ExpertExperimentForm({
  detail,
  catalog,
}: {
  detail: DatasetDetail;
  catalog: RegistryCatalog;
}) {
  const { t } = useTranslation("expert");
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const defaultTarget =
    detail.profile.candidate_targets[0] ??
    detail.profile.column_names[detail.profile.column_names.length - 1] ??
    "";
  const defaultFeatures = detail.profile.column_profiles
    .filter(
      (column) =>
        column.name !== defaultTarget &&
        column.suggested_role === "feature" &&
        isTrainableFeature(column.inferred_type),
    )
    .map((column) => column.name);
  const initialTask = inferInitialTask(detail, defaultTarget);
  const initialModels = catalog.models
    .filter((model) => model.tasks.includes(initialTask))
    .slice(0, 1)
    .map((model) => model.id);
  const initialMetrics = catalog.metrics.filter((metric) =>
    metric.tasks.includes(initialTask),
  );
  const initialValidators = catalog.validators.filter((validator) =>
    validator.tasks.includes(initialTask),
  );

  const [name, setName] = useState(`${detail.dataset.name} expert`);
  const [objective, setObjective] = useState("");
  const [task, setTask] = useState<Task>(initialTask);
  const [target, setTarget] = useState(defaultTarget);
  const [features, setFeatures] = useState(defaultFeatures);
  const [modelIds, setModelIds] = useState(initialModels);
  const [modelParameters, setModelParameters] = useState<ParameterMap>({});
  const [preprocessors, setPreprocessors] = useState<PreprocessorSelection>(
    () =>
      Object.fromEntries(
        catalog.preprocessors.map((item) => [
          item.id,
          {
            enabled: isRelevantPreprocessor(item.category, detail),
            strategy: defaultStrategy(item.id, item.strategies),
            parameters: "{}",
          },
        ]),
      ),
  );
  const [validation, setValidation] = useState(
    initialValidators[0]?.id ?? "train_test_split",
  );
  const [folds, setFolds] = useState(5);
  const [testSize, setTestSize] = useState(0.2);
  const [randomSeed, setRandomSeed] = useState(41);
  const [primaryMetric, setPrimaryMetric] = useState(
    initialMetrics[0]?.id ?? "accuracy",
  );
  const [secondaryMetrics, setSecondaryMetrics] = useState<string[]>([]);
  const [averaging, setAveraging] = useState<"macro" | "weighted">("macro");
  const [rocStrategy, setRocStrategy] = useState<"ovr" | "ovo">("ovr");
  const [positiveLabel, setPositiveLabel] = useState(
    detail.profile.candidate_target_distributions[defaultTarget]?.[0]?.value ??
      "",
  );
  const [tuning, setTuning] = useState(false);
  const [trials, setTrials] = useState(50);
  const [formError, setFormError] = useState<string | null>(null);

  const compatibleModels = catalog.models.filter((model) =>
    model.tasks.includes(task),
  );
  const compatibleMetrics = catalog.metrics.filter((metric) =>
    metric.tasks.includes(task),
  );
  const compatibleValidators = catalog.validators.filter((item) =>
    item.tasks.includes(task),
  );
  const create = useMutation({
    mutationFn: (spec: ExperimentSpec) =>
      createExperiment(spec, { mode: "expert" }),
    onSuccess: async (experiment) => {
      await queryClient.invalidateQueries({ queryKey: ["experiments"] });
      navigate(`/experiments/${experiment.id}`);
    },
  });

  const changeTask = (nextTask: Task) => {
    const models = catalog.models.filter((model) =>
      model.tasks.includes(nextTask),
    );
    const metrics = catalog.metrics.filter((metric) =>
      metric.tasks.includes(nextTask),
    );
    const validators = catalog.validators.filter((item) =>
      item.tasks.includes(nextTask),
    );
    setTask(nextTask);
    setModelIds(models.slice(0, 1).map((model) => model.id));
    setPrimaryMetric(metrics[0]?.id ?? "");
    setSecondaryMetrics([]);
    setValidation(validators[0]?.id ?? "");
  };

  const changeTarget = (nextTarget: string) => {
    setTarget(nextTarget);
    setFeatures((current) =>
      current.filter((feature) => feature !== nextTarget),
    );
    setPositiveLabel(
      detail.profile.candidate_target_distributions[nextTarget]?.[0]?.value ??
        "",
    );
  };

  const submit = (event: FormEvent) => {
    event.preventDefault();
    try {
      if (
        !objective.trim() ||
        !target ||
        features.length === 0 ||
        modelIds.length === 0
      ) {
        throw new Error("required");
      }
      const roles = Object.fromEntries(
        detail.profile.column_names.map((column) => [
          column,
          column === target
            ? "target"
            : features.includes(column)
              ? "feature"
              : detail.profile.possible_id_columns.includes(column)
                ? "id"
                : "ignored",
        ]),
      ) as ExperimentSpec["column_roles"];
      const metricNeedsAveraging = [primaryMetric, ...secondaryMetrics].some(
        (metric) => ["precision", "recall", "f1"].includes(metric),
      );
      const usesRoc = [primaryMetric, ...secondaryMetrics].includes("roc_auc");
      const usesPositiveLabel = [primaryMetric, ...secondaryMetrics].some(
        (metric) =>
          ["precision", "recall", "f1", "roc_auc", "pr_auc"].includes(metric),
      );
      const spec: ExperimentSpec = {
        spec_version: "1.0",
        dataset_id: detail.dataset.id,
        dataset_hash: detail.dataset.hash,
        profile_version: detail.profile.profile_version,
        name: name.trim(),
        objective: objective.trim(),
        task,
        target,
        features,
        column_roles: roles,
        models: modelIds.map((id) => ({
          id,
          enabled: true,
          parameters: parseObject(modelParameters[id] ?? "{}"),
        })),
        preprocessing: catalog.preprocessors
          .filter((item) => preprocessors[item.id]?.enabled)
          .map((item) => ({
            id: item.id,
            strategy: preprocessors[item.id].strategy,
            parameters: parseObject(preprocessors[item.id].parameters),
          })),
        validation: {
          strategy: validation,
          folds,
          test_size: testSize,
          shuffle: true,
          random_seed: randomSeed,
        },
        metrics: {
          primary: primaryMetric,
          secondary: secondaryMetrics.filter(
            (metric) => metric !== primaryMetric,
          ),
          averaging:
            task === "multiclass_classification" && metricNeedsAveraging
              ? averaging
              : null,
          roc_auc_strategy:
            task === "multiclass_classification" && usesRoc
              ? rocStrategy
              : null,
          positive_label:
            task === "binary_classification" && usesPositiveLabel
              ? coerceTargetValue(positiveLabel, detail, target)
              : null,
        },
        tuning: {
          enabled: tuning,
          engine: "optuna",
          trials,
          metric: tuning ? primaryMetric : null,
        },
      };
      setFormError(null);
      create.mutate(spec);
    } catch {
      setFormError(t("errors.invalid"));
    }
  };

  return (
    <form className="space-y-6" onSubmit={submit} data-testid="expert-form">
      <WorkflowSteps current={4} />
      <section className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <button
            type="button"
            className="text-sm text-primary"
            onClick={() => navigate(`/datasets/${detail.dataset.id}`)}
          >
            {t("back")}
          </button>
          <p className="mt-4 text-sm font-semibold uppercase tracking-[0.16em] text-primary">
            {t("eyebrow")}
          </p>
          <h1 className="mt-2 text-3xl font-semibold">{t("title")}</h1>
          <p className="mt-3 max-w-3xl text-muted">{t("description")}</p>
        </div>
        <HowItWorksButton screen="expert" />
      </section>

      <Card className="grid gap-4 md:grid-cols-2">
        <TextField label={t("fields.name")} value={name} onChange={setName} />
        <SelectField
          label={t("fields.task")}
          value={task}
          onChange={(value) => changeTask(value as Task)}
          options={[
            ["binary_classification", t("tasks.binary_classification")],
            ["multiclass_classification", t("tasks.multiclass_classification")],
            ["regression", t("tasks.regression")],
          ]}
        />
        <label className="md:col-span-2 text-sm font-semibold">
          {t("fields.objective")}
          <textarea
            className={`${inputClass} mt-2 min-h-24 py-2`}
            value={objective}
            onChange={(event) => setObjective(event.target.value)}
          />
        </label>
      </Card>

      <Card>
        <SectionTitle
          title={t("columns.title")}
          description={t("columns.description")}
        />
        <div className="mt-4 grid gap-5 lg:grid-cols-[0.4fr_1fr]">
          <SelectField
            label={t("fields.target")}
            value={target}
            onChange={changeTarget}
            options={detail.profile.column_names.map((column) => [
              column,
              column,
            ])}
          />
          <fieldset>
            <legend className="text-sm font-semibold">
              {t("fields.features")}
            </legend>
            <div className="mt-2 grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {detail.profile.column_names
                .filter((column) => column !== target)
                .map((column) => {
                  const profile = detail.profile.column_profiles.find(
                    (item) => item.name === column,
                  );
                  const disabled = !isTrainableFeature(
                    profile?.inferred_type,
                    preprocessors.text_vectorization?.enabled,
                  );
                  return (
                    <Checkbox
                      key={column}
                      label={
                        disabled
                          ? `${column} (${profile?.inferred_type === "text" ? t("columns.textRequiresTfidf") : t("columns.emptyDisabled")})`
                          : `${column} (${profile?.inferred_type ?? "?"})`
                      }
                      checked={features.includes(column)}
                      disabled={disabled}
                      onChange={(checked) =>
                        setFeatures(toggle(features, column, checked))
                      }
                    />
                  );
                })}
            </div>
          </fieldset>
        </div>
        <p className="mt-3 text-xs text-muted">{t("columns.typeHelp")}</p>
      </Card>

      <Card>
        <SectionTitle
          title={t("models.title")}
          description={t("models.description")}
        />
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {compatibleModels.map((model) => (
            <div key={model.id} className="rounded-lg border border-border p-4">
              <Checkbox
                label={model.display_name}
                checked={modelIds.includes(model.id)}
                onChange={(checked) =>
                  setModelIds(toggle(modelIds, model.id, checked))
                }
              />
              {modelIds.includes(model.id) ? (
                <JsonField
                  label={t("fields.parameters")}
                  value={modelParameters[model.id] ?? "{}"}
                  onChange={(value) =>
                    setModelParameters((current) => ({
                      ...current,
                      [model.id]: value,
                    }))
                  }
                />
              ) : null}
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <SectionTitle
          title={t("preprocessing.title")}
          description={t("preprocessing.description")}
        />
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {catalog.preprocessors.map((item) => {
            const selected = preprocessors[item.id];
            return (
              <div
                key={item.id}
                className="rounded-lg border border-border p-4"
              >
                <Checkbox
                  label={t(`preprocessing.components.${item.id}`, {
                    defaultValue: item.display_name,
                  })}
                  checked={selected.enabled}
                  onChange={(enabled) =>
                    setPreprocessors((current) => ({
                      ...current,
                      [item.id]: { ...current[item.id], enabled },
                    }))
                  }
                />
                {selected.enabled ? (
                  <div className="mt-3 space-y-3">
                    <SelectField
                      label={t("fields.strategy")}
                      value={selected.strategy}
                      onChange={(strategy) =>
                        setPreprocessors((current) => ({
                          ...current,
                          [item.id]: { ...current[item.id], strategy },
                        }))
                      }
                      options={item.strategies.map((strategy) => [
                        strategy,
                        strategy,
                      ])}
                    />
                    <JsonField
                      label={t("fields.parameters")}
                      value={selected.parameters}
                      onChange={(parameters) =>
                        setPreprocessors((current) => ({
                          ...current,
                          [item.id]: { ...current[item.id], parameters },
                        }))
                      }
                    />
                  </div>
                ) : null}
              </div>
            );
          })}
        </div>
      </Card>

      <Card className="grid gap-5 md:grid-cols-2 lg:grid-cols-4">
        <div className="md:col-span-2 lg:col-span-4">
          <SectionTitle
            title={t("validation.title")}
            description={t("validation.description")}
          />
        </div>
        <SelectField
          label={t("fields.validation")}
          value={validation}
          onChange={setValidation}
          options={compatibleValidators.map((item) => [
            item.id,
            item.display_name,
          ])}
        />
        <NumberField
          label={t("fields.folds")}
          value={folds}
          onChange={setFolds}
          min={2}
          max={20}
        />
        <NumberField
          label={t("fields.testSize")}
          value={testSize}
          onChange={setTestSize}
          min={0.05}
          max={0.95}
          step={0.05}
        />
        <NumberField
          label={t("fields.seed")}
          value={randomSeed}
          onChange={setRandomSeed}
        />
      </Card>

      <Card>
        <SectionTitle
          title={t("metrics.title")}
          description={t("metrics.description")}
        />
        <div className="mt-4 grid gap-5 md:grid-cols-2">
          <SelectField
            label={t("fields.primaryMetric")}
            value={primaryMetric}
            onChange={setPrimaryMetric}
            options={compatibleMetrics.map((metric) => [
              metric.id,
              metric.display_name,
            ])}
          />
          {task === "binary_classification" ? (
            <TextField
              label={t("fields.positiveLabel")}
              value={positiveLabel}
              onChange={setPositiveLabel}
            />
          ) : null}
          {task === "multiclass_classification" ? (
            <>
              <SelectField
                label={t("fields.averaging")}
                value={averaging}
                onChange={(value) =>
                  setAveraging(value as "macro" | "weighted")
                }
                options={[
                  ["macro", "macro"],
                  ["weighted", "weighted"],
                ]}
              />
              <SelectField
                label={t("fields.rocStrategy")}
                value={rocStrategy}
                onChange={(value) => setRocStrategy(value as "ovr" | "ovo")}
                options={[
                  ["ovr", "ovr"],
                  ["ovo", "ovo"],
                ]}
              />
            </>
          ) : null}
          <fieldset className="md:col-span-2">
            <legend className="text-sm font-semibold">
              {t("fields.secondaryMetrics")}
            </legend>
            <div className="mt-2 flex flex-wrap gap-3">
              {compatibleMetrics
                .filter((metric) => metric.id !== primaryMetric)
                .map((metric) => (
                  <Checkbox
                    key={metric.id}
                    label={metric.display_name}
                    checked={secondaryMetrics.includes(metric.id)}
                    onChange={(checked) =>
                      setSecondaryMetrics(
                        toggle(secondaryMetrics, metric.id, checked),
                      )
                    }
                  />
                ))}
            </div>
          </fieldset>
        </div>
      </Card>

      <Card className="grid gap-4 md:grid-cols-3">
        <div className="md:col-span-3">
          <SectionTitle
            title={t("tuning.title")}
            description={t("tuning.description")}
          />
        </div>
        <Checkbox
          label={t("fields.enableTuning")}
          checked={tuning}
          onChange={setTuning}
        />
        {tuning ? (
          <NumberField
            label={t("fields.trials")}
            value={trials}
            onChange={setTrials}
            min={1}
            max={1000}
          />
        ) : null}
      </Card>

      {formError || create.isError ? (
        <p className="text-sm text-error" role="alert">
          {formError ??
            (create.error instanceof ExperimentApiError &&
            create.error.issues.length
              ? describeExperimentIssues(create.error.issues)
              : t("errors.create"))}
        </p>
      ) : null}
      <div className="flex justify-end">
        <Button type="submit" disabled={create.isPending}>
          {create.isPending ? t("actions.creating") : t("actions.create")}
        </Button>
      </div>
    </form>
  );
}

function SectionTitle({
  title,
  description,
}: {
  title: string;
  description: string;
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold">{title}</h2>
      <p className="mt-1 text-sm text-muted">{description}</p>
    </div>
  );
}

function TextField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="text-sm font-semibold">
      {label}
      <input
        className={`${inputClass} mt-2`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
}) {
  return (
    <label className="text-sm font-semibold">
      {label}
      <input
        type="number"
        className={`${inputClass} mt-2`}
        value={value}
        min={min}
        max={max}
        step={step}
        onChange={(event) => onChange(Number(event.target.value))}
      />
    </label>
  );
}

function SelectField({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
  options: [string, string][];
}) {
  return (
    <label className="text-sm font-semibold">
      {label}
      <select
        className={`${inputClass} mt-2`}
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {options.map(([id, name]) => (
          <option key={id} value={id}>
            {name}
          </option>
        ))}
      </select>
    </label>
  );
}

/**
 * Componente de checkbox com rótulo textual e estilo acessível.
 *
 * @param props.label - Texto de descrição do checkbox.
 * @param props.checked - Se está marcado.
 * @param props.disabled - Se está desabilitado. Padrão false.
 * @param props.onChange - Callback acionado na alteração do valor.
 * @returns Elemento label com input tipo checkbox.
 */
function Checkbox({
  label,
  checked,
  disabled = false,
  onChange,
}: {
  label: string;
  checked: boolean;
  disabled?: boolean;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      className={`flex items-center gap-2 text-sm ${disabled ? "text-muted" : ""}`}
    >
      <input
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={(event) => onChange(event.target.checked)}
      />
      <span>{label}</span>
    </label>
  );
}

/**
 * Campo de texto multilinhas para edição de JSON de parâmetros de hiperajuste ou algoritmos.
 *
 * @param props.label - Rótulo do campo JSON.
 * @param props.value - Texto JSON atual.
 * @param props.onChange - Callback para atualização da string.
 * @returns Label com textarea monoespaçada.
 */
function JsonField({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="mt-3 block text-xs text-muted">
      {label}
      <textarea
        className="mt-1 min-h-20 w-full rounded-md border border-border bg-code p-2 font-mono text-xs text-code-text"
        value={value}
        spellCheck={false}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

/**
 * Adiciona ou remove um valor de uma lista imutável com base na flag de habilitação.
 *
 * @param values - Lista original de strings.
 * @param value - Elemento a adicionar ou remover.
 * @param enabled - Se o elemento deve estar presente na lista.
 * @returns Nova lista com a alteração aplicada.
 */
function toggle(values: string[], value: string, enabled: boolean): string[] {
  return enabled
    ? [...new Set([...values, value])]
    : values.filter((item) => item !== value);
}

/**
 * Verifica se um tipo de dado inferido de coluna é elegível para uso como feature preditiva.
 *
 * @param inferredType - Tipo semântico inferido da coluna.
 * @param textEnabled - Se colunas textuais foram habilitadas para vetorização TF-IDF. Padrão false.
 * @returns Verdadeiro se a coluna puder ser usada como feature no treinamento.
 */
function isTrainableFeature(
  inferredType: string | undefined,
  textEnabled = false,
): boolean {
  return inferredType !== "empty" && (inferredType !== "text" || textEnabled);
}

/**
 * Infere a tarefa inicial padrão de Machine Learning com base na cardinalidade e tipo da coluna alvo.
 *
 * @param detail - Detalhes do dataset com perfis das colunas.
 * @param target - Nome da coluna selecionada como alvo.
 * @returns Tarefa deduzida ('binary_classification', 'multiclass_classification' ou 'regression').
 */
function inferInitialTask(detail: DatasetDetail, target: string): Task {
  const profile = detail.profile.column_profiles.find(
    (column) => column.name === target,
  );
  const cardinality =
    profile?.cardinality ??
    detail.profile.candidate_target_distributions[target]?.length ??
    0;
  if (cardinality === 2) return "binary_classification";
  if (profile?.inferred_type === "numeric" && cardinality > 20) {
    return "regression";
  }
  return cardinality >= 3
    ? "multiclass_classification"
    : "binary_classification";
}

/**
 * Realiza o parse seguro de uma string JSON garantindo que seja um objeto plano não-nulo e não-array.
 *
 * @param value - String contendo objeto JSON.
 * @returns Objeto parseado.
 * @throws {Error} Se o JSON for inválido ou não for um objeto.
 */
function parseObject(value: string): Record<string, unknown> {
  const parsed = JSON.parse(value) as unknown;
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
    throw new Error("parameters");
  return parsed as Record<string, unknown>;
}

/**
 * Seleciona a estratégia padrão recomendada para uma etapa de pré-processamento.
 *
 * @param id - Identificador do tipo de pré-processador.
 * @param strategies - Lista de estratégias suportadas disponíveis.
 * @returns Nome da estratégia preferencial ou a primeira disponível.
 */
function defaultStrategy(id: string, strategies: string[]): string {
  const preferred: Record<string, string> = {
    missing_values: "median",
    categorical_encoding: "one_hot",
    boolean_encoding: "binary",
    numeric_scaling: "standard",
    datetime: "components",
  };
  return strategies.includes(preferred[id])
    ? preferred[id]
    : (strategies[0] ?? "none");
}

/**
 * Avalia se uma categoria de pré-processador é necessária dadas as características do dataset.
 *
 * @param category - Categoria do pré-processador (missing, categorical, boolean, numeric, datetime).
 * @param detail - Detalhes do dataset com detecções de tipos e valores faltantes.
 * @returns Verdadeiro se houver colunas do tipo correspondente no dataset.
 */
function isRelevantPreprocessor(
  category: string,
  detail: DatasetDetail,
): boolean {
  if (category === "missing")
    return Object.values(detail.profile.missing_values).some(
      (value) => value > 0,
    );
  if (category === "categorical")
    return detail.profile.categorical_columns.length > 0;
  if (category === "boolean") return detail.profile.boolean_columns.length > 0;
  if (category === "numeric") return detail.profile.numeric_columns.length > 0;
  if (category === "datetime")
    return detail.profile.datetime_columns.length > 0;
  return false;
}

/**
 * Converte o valor da classe positiva em seu tipo de dado primitivo original (string, número ou booleano).
 *
 * @param value - Representação em string da classe positiva.
 * @param detail - Detalhes do dataset com perfis de distribuição.
 * @param target - Nome da coluna alvo.
 * @returns Valor convertido no tipo nativo correto.
 */
function coerceTargetValue(
  value: string,
  detail: DatasetDetail,
  target: string,
): string | number | boolean {
  const profiledValue = detail.profile.candidate_target_distributions[
    target
  ]?.find((item) => String(item.value) === value)?.value;
  if (profiledValue !== undefined && profiledValue !== null) {
    return profiledValue;
  }
  const profile = detail.profile.column_profiles.find(
    (column) => column.name === target,
  );
  if (
    profile?.inferred_type === "numeric" &&
    value.trim() !== "" &&
    Number.isFinite(Number(value))
  )
    return Number(value);
  return value;
}
