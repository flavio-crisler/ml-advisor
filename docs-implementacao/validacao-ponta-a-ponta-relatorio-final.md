# Relatório final — validação ponta a ponta do ML Advisor

**Data:** 26 de setembro de 2026

**Ambiente:** stack local Docker Compose, backend em `127.0.0.1:18000`, frontend em `127.0.0.1:18080`, SQLite e artefatos em `data/`.

**Acompanhamento detalhado:** [validacao-ponta-a-ponta-acompanhamento.md](validacao-ponta-a-ponta-acompanhamento.md).

## 1. Objetivo

Verificar cada operação publicada pelo backend, o fluxo de interface que a utiliza quando existe, os contratos de erro relevantes, a persistência e as integrações externas reais. Corrigir defeitos encontrados e repetir os cenários afetados antes de considerar a operação concluída.

## 2. Metodologia

O inventário veio do OpenAPI servido pela stack e foi confrontado com as rotas FastAPI e os serviços/componentes React. As chamadas de validação foram HTTP reais na aplicação em execução. O frontend foi operado pelo MCP Playwright com Chrome, formulários e ações de usuário, inspeção de requisições e console. Jobs foram acompanhados até estado terminal, com conferência de SQLite e arquivos quando aplicável. Respostas esperadas de erro foram exercitadas sem substituir provedores por respostas artificiais. A tabela de acompanhamento foi atualizada durante a execução.

Os testes automatizados do repositório foram executados ao final como gate adicional. Eles **não** substituem as evidências HTTP, Playwright e de provedores externos descritas acima.

## 3. Total encontrado

O OpenAPI contém **57 operações HTTP**, todas presentes no inventário. O total inclui as rotas canônicas e aliases publicados, como `decision-runs` e `decision-benchmarks`.

## 4. Backend testado

**57/57 operações** tiveram chamada real bem-sucedida e análise. Foram exercitados erros relevantes de validação, recurso inexistente, estado incompatível e processamento externo. Os jobs de treino, benchmark Jev e variáveis semânticas foram observados até conclusão ou cancelamento; registros e artefatos resultantes foram conferidos.

## 5. Frontend testado

**45/45 operações com chamada direta na interface** tiveram fluxo via MCP Playwright, confirmação da rota/status na rede e verificação do estado exibido e do console. As outras **12 operações** não têm chamada direta no frontend; foram testadas no backend e marcadas `N/A` na coluna da interface. Alguns dados dessas rotas aparecem na UI por outra resposta, por exemplo impacto semântico embutido no histórico.

## 6. Datasets e fixtures

- `backend/tests/fixtures/synthetic_sales_semantic.csv`: upload, perfil, recomendação OpenRouter, fluxo automático, treino e comparação; 30 linhas e 14 colunas no dataset persistido.
- `backend/tests/fixtures/attendance.csv`: benchmark Jev, experimento supervisionado, calibração, comparação e geração semântica; 12 linhas.
- `backend/tests/fixtures/KAGGLE_SOURCES.md`: envio inválido para conferir erro de formato e mensagem localizada.
- Datasets já persistidos foram usados para ler históricos e artefatos anteriores; a comprovação das novas execuções não dependeu apenas deles.

## 7. Serviços externos acionados

O OpenRouter foi chamado de verdade para recomendação e estratégia automática, explicações e proposta de melhoria; o modelo configurado da recomendação era `google/gemini-2.5-flash-lite`. O Decision Provider `openrouter_jev` foi consultado para teste de conexão, benchmark e geração de variáveis, com modelo solicitado `typesafe/jev-1.13`. O benchmark com `attendance.csv` completou 12 decisões externas sem cache. O conjunto semântico novo `ee55d4af-9929-43e5-8417-e7abf9853def` completou 12/12 linhas, 12 chamadas externas e zero cache hits. Não houve substituição dessas chamadas por mocks na validação ponta a ponta.

## 8. Problemas encontrados e 9. Causas relevantes

| Problema observado | Causa confirmada |
|---|---|
| `POST /auto/run` retornou 502 e depois 422 para respostas reais da IA. | Saída estruturada inválida não tinha uma repetição limitada; erro de contrato genérico não informava ao provedor o campo ou regra a corrigir. |
| Benchmark Jev antigo ficou em `running` sem worker ativo após reinício. | Executor local em memória não reconciliava registros persistidos; o shutdown encerrava apenas o executor de treino. |
| `Last-Event-ID` inválido gerava 500 nas três rotas SSE. | Conversão manual com `int()` sem validação de entrada. |
| Validação de spec aceitava hash de dataset incorreto. | A rota verificava apenas o catálogo, sem consultar o dataset e o contrato completo do serviço. |
| Geração semântica ativa não podia ser cancelada na tela. | Serviço frontend de cancelamento existia, mas não estava ligado a uma ação do histórico. |
| Primeiro `make test` falhou em cinco testes após a reconciliação no startup. | A suíte não preparava o SQLite padrão usado no ciclo de vida da aplicação; um teste antigo ainda assumia validação de spec sem dataset persistido. |

## 10. Correções implementadas

- Advisor: repetição limitada para resposta estruturada inválida e feedback de contrato específico e seguro; execução real repetida até `completed`, com 5 modelos, 250 trials e artefatos.
- Jobs Jev: registros `queued`/`running` sem worker são marcados `failed` no startup com progresso preservado; três executores são encerrados no shutdown; a UI traduz `decision_job_interrupted`. O registro órfão e a tela foram conferidos após a reconstrução.
- SSE: cursor do header tipado como inteiro não negativo nas rotas de experimentos, decisões e variáveis semânticas. HTTP real passou a devolver 422 para header inválido e preservou a retomada por ID.
- Spec: rota passou a chamar `ExperimentService.validate_specification`; hash incorreto retorna 400, dataset ausente 404. A execução real no editor e o reteste HTTP passaram.
- Variáveis semânticas: botão de cancelamento e mensagens pt-BR/en-US adicionados ao cartão ativo. Job real foi cancelado pela UI, persistiu `cancelled`, ficou em 0/12 linhas e não produziu artefato; POST 200 e console limpo.
- Testes: startup da suíte recebe SQLite temporário migrado; o teste de spec usa dataset e repositórios reais. Os cinco casos inicialmente falhos e o gate completo passaram depois da correção.

## 11. Arquivos principais modificados

Backend: `app/application/advisor/{service,validation,prompts}.py`, `app/main.py`, `app/infrastructure/repositories/sqlalchemy.py`, `app/api/routes/{experiment_specs,experiments,decision_runs,semantic_features}.py`. Frontend: `features/calibration-lab/calibration-lab-page.tsx`, `features/semantic-feature-discovery/semantic-feature-page.tsx` e as duas traduções `decisionLab.json`. Testes backend, `.gitignore` e os dois documentos de validação também foram atualizados. Os caminhos completos e o estado do checkout estão no diff do repositório.

## 12. Problemas sem correção automática

Não restou operação do inventário com falha funcional reproduzida. Permanecem dois limites de conteúdo gerado pelo modelo:

- O pedido textual por um único baseline sem tuning não foi obedecido pelo advisor real; a estratégia criada continha 5 modelos e 50 trials por modelo. O backend validou e executou o contrato recebido. Converter instruções livres em limites determinísticos de custo/escopo requer um contrato explícito de produto, não uma heurística ad hoc nesta validação.
- Uma explicação real de CatBoost citou dois números diferentes de ROC AUC no mesmo texto. A API e a UI funcionaram, mas a redação da IA não deve ser tratada como fonte primária das métricas. A correção robusta exige verificar afirmações numéricas contra o contexto estruturado antes de exibir ou confiar no texto.

## 13. Riscos e pontos de atenção

- O conjunto semântico novo sobre `attendance.csv` apresentou F1 **1,0000** com e sem variáveis geradas. Isso é ausência de variação nesta amostra de 12 linhas, não prova de ganho ou generalização. O braço TF-IDF não estava disponível porque não havia texto livre novo adequado em relação ao experimento original.
- Explicações e recomendações da IA precisam de revisão humana, sobretudo números, causalidade e escopo de treino. O valor exibido na tabela estruturada é a referência para métricas.
- Jobs locais em andamento podem ser interrompidos por reinício; a reconciliação de benchmarks Jev foi verificada, mas continuidade automática de processamento não faz parte deste executor local.
- Esta validação ocorreu na stack local com provedores externos reais. Não constitui evidência de publicação em outro ambiente.

## 14. Resultado final e gates

**Concluído no ambiente local:** 57/57 operações backend e 45/45 fluxos frontend diretos; 12 rotas sem uso direto na UI foram marcadas `N/A`. O conjunto semântico gerado gravou `questions.json`, `features.parquet`, `feature_metadata.json` e `impact.json`; a explicação de impacto retornou 200 pelo OpenRouter e foi exibida na interface. O cancelamento e os eventos SSE também passaram com estado e cursor coerentes.

`make test`: **209 testes backend e 52 testes frontend aprovados**. `make lint`: aprovado após formatação. Build frontend de produção: aprovado. `git diff --check`: aprovado. Esses gates são complementares à validação real acima.

## 15. Tabela consolidada

Cada linha corresponde a uma operação do OpenAPI. `N/A` significa ausência de chamada direta no frontend, conforme o mapeamento detalhado no [acompanhamento](validacao-ponta-a-ponta-acompanhamento.md).

| Método | Endpoint | Backend | Frontend | Status |
|---|---|---|---|---|
| GET | `/api/v1/health` | [x] | [x] | Concluído |
| POST | `/api/v1/advisor/recommend` | [x] | [x] | Concluído |
| POST | `/api/v1/auto/run` | [x] | [x] | Concluído |
| GET | `/api/v1/confidence-cascades` | [x] | [x] | Concluído |
| POST | `/api/v1/confidence-cascades` | [x] | [x] | Concluído |
| GET | `/api/v1/confidence-cascades/{cascade_id}` | [x] | N/A | Concluído |
| GET | `/api/v1/datasets` | [x] | [x] | Concluído |
| POST | `/api/v1/datasets` | [x] | [x] | Concluído |
| GET | `/api/v1/datasets/{dataset_id}` | [x] | [x] | Concluído |
| GET | `/api/v1/datasets/{dataset_id}/profile` | [x] | N/A | Concluído |
| GET | `/api/v1/decision-providers` | [x] | [x] | Concluído |
| POST | `/api/v1/decision-providers/test` | [x] | [x] | Concluído |
| GET | `/api/v1/decision-runs` | [x] | [x] | Concluído |
| POST | `/api/v1/decision-runs` | [x] | N/A | Concluído |
| POST | `/api/v1/decision-benchmarks` | [x] | [x] | Concluído |
| GET | `/api/v1/decision-benchmarks/{run_id}` | [x] | [x] | Concluído |
| GET | `/api/v1/decision-runs/{run_id}` | [x] | N/A | Concluído |
| GET | `/api/v1/decision-benchmarks/{run_id}/calibration` | [x] | [x] | Concluído |
| GET | `/api/v1/decision-benchmarks/{run_id}/compare/{experiment_id}` | [x] | [x] | Concluído |
| POST | `/api/v1/decision-runs/{run_id}/cancel` | [x] | N/A | Concluído |
| GET | `/api/v1/decision-runs/{run_id}/events` | [x] | N/A | Concluído |
| POST | `/api/v1/semantic-features/propose` | [x] | [x] | Concluído |
| POST | `/api/v1/semantic-features/generate` | [x] | [x] | Concluído |
| GET | `/api/v1/semantic-features` | [x] | [x] | Concluído |
| GET | `/api/v1/semantic-features/{feature_set_id}` | [x] | N/A | Concluído |
| GET | `/api/v1/semantic-features/{feature_set_id}/impact` | [x] | N/A | Concluído |
| POST | `/api/v1/semantic-features/{feature_set_id}/explanation` | [x] | [x] | Concluído |
| POST | `/api/v1/semantic-features/{feature_set_id}/cancel` | [x] | [x] | Concluído |
| GET | `/api/v1/semantic-features/{feature_set_id}/events` | [x] | N/A | Concluído |
| GET | `/api/v1/models` | [x] | [x] | Concluído |
| GET | `/api/v1/metrics` | [x] | [x] | Concluído |
| GET | `/api/v1/preprocessors` | [x] | [x] | Concluído |
| GET | `/api/v1/validators` | [x] | [x] | Concluído |
| GET | `/api/v1/tuners` | [x] | [x] | Concluído |
| POST | `/api/v1/experiment-specs/validate` | [x] | N/A | Concluído |
| GET | `/api/v1/experiments` | [x] | [x] | Concluído |
| POST | `/api/v1/experiments` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/{experiment_id}` | [x] | [x] | Concluído |
| PUT | `/api/v1/experiments/{experiment_id}` | [x] | [x] | Concluído |
| PATCH | `/api/v1/experiments/{experiment_id}` | [x] | [x] | Concluído |
| DELETE | `/api/v1/experiments/{experiment_id}` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/{experiment_id}/strategy` | [x] | N/A | Concluído |
| GET | `/api/v1/experiments/runs/{run_id}` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/runs/{run_id}/tuning-trials` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/runs/{run_id}/analysis` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/{experiment_id}/compare/{other_experiment_id}` | [x] | [x] | Concluído |
| POST | `/api/v1/experiments/{experiment_id}/clone` | [x] | [x] | Concluído |
| POST | `/api/v1/experiments/{experiment_id}/run` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/jobs/{job_id}` | [x] | N/A | Concluído |
| POST | `/api/v1/experiments/{experiment_id}/cancel` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/{experiment_id}/comparison` | [x] | [x] | Concluído |
| POST | `/api/v1/experiments/{experiment_id}/explanation` | [x] | [x] | Concluído |
| GET | `/api/v1/experiments/{experiment_id}/events` | [x] | [x] | Concluído |
| GET | `/api/v1/explanations/status` | [x] | [x] | Concluído |
| POST | `/api/v1/explanations` | [x] | [x] | Concluído |
| POST | `/api/v1/improvement-proposals/supervised` | [x] | [x] | Concluído |
| POST | `/api/v1/improvement-proposals/supervised/draft` | [x] | [x] | Concluído |
