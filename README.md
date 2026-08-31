# Enterprise COPOM RAG Lakehouse & Data Observability Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Pydantic v2](https://img.shields.io/badge/contracts-Pydantic%20v2-e92063.svg)](https://docs.pydantic.dev/)
[![Qdrant Vector DB](https://img.shields.io/badge/vectorstore-Qdrant-dc2626.svg)](https://qdrant.tech/)
[![Arize Phoenix](https://img.shields.io/badge/observability-Arize%20Phoenix-fbbf24.svg)](https://phoenix.arize.com/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Tests](https://img.shields.io/badge/pytest-46%20passed-brightgreen.svg)](https://docs.pytest.org/)
[![Coverage](https://img.shields.io/badge/cobertura-88%25-green.svg)](https://docs.pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Uma esteira completa de engenharia de dados e lakehouse vetorial para processamento resiliente de documentos não estruturados (**Atas e Comunicados do COPOM / Banco Central do Brasil**). Construído com arquitetura Medalhão, contratos estritos de dados (Pydantic v2), indexação vetorial idempotente no **Qdrant** e observabilidade de ponta a ponta com **Arize Phoenix (OpenTelemetry)**.

---

## 🏛️ Arquitetura do Lakehouse Medalhão

```text
┌──────────────────────────────────────────────────────────────────────────────────┐
│             ENTERPRISE COPOM RAG LAKEHOUSE & OBSERVABILITY ARCHITECTURE          │
└──────────────────────────────────────────────────────────────────────────────────┘

                       ┌───────────────────────────────┐
                       │  Banco Central do Brasil API  │
                       │   (Publicações OData / REST)  │
                       └───────────────┬───────────────┘
                                       │ HTTP Session + Tenacity Backoff
                                       │ (429 Rate Limit / 5xx Retries)
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 🥉 CAMADA BRONZE (Raw Lakehouse Storage)                            │
    │  • Ingestão Idempotente baseada em Hash SHA-256                    │
    │  • Particionamento Hive: data/bronze/year=YYYY/month=MM/            │
    │  • Arquivos Canônicos: copom_{doc_id}_{short_hash}.json             │
    │  • Contrato de Dados: Pydantic BronzeAtaRecord                      │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │ Raw Content Extraction
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 🥈 CAMADA SILVER (Cleaned, Structured & Chunked)                    │
    │  • Sanitização HTML, decodificação de entidades e normalização      │
    │  • Remoção do rodapé de presença (~14% do corpus, sem valor semântico) │
    │  • Recursive Character Splitting com overlap semântico             │
    │  • Enriquecimento com Metadados & Token Counting (TikToken)         │
    │  • Particionamento Hive: data/silver/year=YYYY/month=MM/            │
    │  • Contrato de Dados: Pydantic ChunkPayload & SilverDocument        │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │ Batch Dense Vector Embeddings
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 🥇 CAMADA GOLD VETORIAL (Vectorstore Lakehouse)                     │
    │  • Embeddings em Lote: FastEmbed multilingual (ONNX) ou OpenAI      │
    │  • Retrieval assimétrico: prefixos query/passage por modelo         │
    │  • Qdrant Vector DB: Métrica Cosseno, HNSW e Payload Indexing       │
    │  • Idempotência Garantida: UUIDv5 determinístico por Chunk ID       │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 💬 CAMADA DE GERAÇÃO (RAG — comando `ask`)                          │
    │  • Resposta fundamentada apenas nos trechos recuperados             │
    │  • Citações [1][2] rastreáveis até reunião e data                   │
    │  • Endpoint compatível com OpenAI (NVIDIA NIM por padrão)           │
    └──────────────────────────────────┬──────────────────────────────────┘
                                       │
                                       ▼
    ┌─────────────────────────────────────────────────────────────────────┐
    │ 🔍 CAMADA DE OBSERVABILIDADE & GOVERNANÇA (Arize Phoenix / OTel)    │
    │  • OpenTelemetry Spans para cada estágio do pipeline                │
    │  • Rastreamento de latência de inferência, contagem de tokens       │
    │  • Monitoramento de qualidade de recuperação RAG (Top-k Similarity) │
    └─────────────────────────────────────────────────────────────────────┘
```

---

## ⚡ Principais Recursos de Engenharia

1. **Ingestão Idempotente, Versionada & Particionamento Hive**:
   - Cada publicação tem seu hash SHA-256 calculado. Se o arquivo já existir com o mesmo conteúdo, a escrita é evitada com custo zero de I/O.
   - **O Bronze é append-only.** Quando o BCB retifica uma ata já ingerida, o novo conteúdo gera um novo hash e é gravado ao lado da versão anterior — o histórico nunca é destruído.
   - Como as camadas seguintes indexam por `doc_id`, `BronzeCollector.select_current_versions()` elege a **versão corrente** (a de ingestão mais recente, com desempate determinístico por hash) antes da promoção para Silver. Sem isso, qual versão sobreviveria dependeria da ordem alfabética do hash no nome do arquivo.
   - Organização estruturada compatível com Data Lakes modernos (`year=YYYY/month=MM/`).
2. **Contratos de Dados Rígidos (Pydantic v2)**:
   - Validação em tempo de execução com `BronzeAtaRecord`, `ChunkMetadata` e `ChunkPayload`.
   - **A validação começa na fronteira**: `RawAtaDetail` valida a resposta do endpoint de detalhes antes de qualquer uso, porque é dela que sai todo o texto do pipeline. Atas anteriores a ~2021 trazem `textoAta` nulo (só há PDF) e são rejeitadas com o motivo registrado, em vez de gerarem um documento vazio.
   - Data de publicação ilegível **levanta erro** em vez de assumir a data de hoje — o que arquivaria o documento na partição errada e o tornaria invisível ao filtro `--year`.
   - Conversão segura de tipos e garantia de integridade estrutural.
3. **Chunking Semântico com Enriquecimento**:
   - Sanitização de ruídos HTML preservando as seções do COPOM (*A) Atualização da conjuntura*, *B) Cenários e análise de riscos*, *C) Discussão sobre a condução da política monetária*, *D) Decisão de política monetária*).
   - **Remoção do rodapé de presença.** Toda ata encerra com a lista de nomes e cargos dos participantes mais uma frase padrão repetida literalmente entre publicações. Medido em 11 reuniões (240–280), isso é **14% do corpus** — texto quase idêntico entre documentos, que só dilui a recuperação. O corte usa o marcador `Presentes:` e só se aplica se ele estiver na segunda metade do documento (nas atas amostradas nunca aparece antes de 79%).
   - Divisão com `RecursiveCharacterTextSplitter` e `tiktoken`, injetando metadados como número da reunião, data, ano, mês, hash e contagem de tokens.
4. **Idempotência no Vector Store (Qdrant)**:
   - Geração de IDs de ponto vetorial baseada em `UUIDv5` determinístico a partir de `doc_id + chunk_id`.
   - Re-execuções da esteira atualizam os registros sem criar duplicações de vetores.
5. **Escolha do modelo de embedding guiada por medição**:
   - O corpus é 100% em português e a busca é assimétrica (pergunta curta contra passagem longa), então o modelo precisa ser multilingual **e** treinado para retrieval.
   - Comparação sobre 102 chunks de 16 reuniões, `hit@1` em 10 perguntas reais em português:

   | Modelo | dim | tamanho | hit@1 |
   | :--- | ---: | ---: | ---: |
   | `intfloat/multilingual-e5-large` | 1024 | 2,2 GB | **9/10** |
   | `BAAI/bge-small-en-v1.5` (só inglês) | 384 | 0,13 GB | 9/10 |
   | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 384 | 0,22 GB | 6/10 |

   - O MiniLM é multilingual, mas treinado para similaridade simétrica de paráfrase: erra 3 perguntas que os outros acertam. "Multilingual" sozinho não substitui "treinado para busca".
   - Modelos da família e5 exigem os prefixos `query:`/`passage:`. `EmbeddingGenerator` usa `passage_embed()` ao indexar e `query_embed()` ao buscar, então trocar de modelo é só configuração.
   - Trocar de modelo com largura diferente exige recriar a coleção; `ensure_collection()` levanta erro em vez de deixar o upsert falhar em silêncio.
   - Se o modelo configurado não carregar, o pipeline **para com erro** em vez de substituí-lo por outro ou gerar vetores sem significado — uma busca ruim tem muitas causas possíveis, e um embedder errado é a mais difícil de diagnosticar.

6. **Observabilidade de LLM/RAG (Arize Phoenix & OpenTelemetry)**:
   - Rastreamento completo de latência, contagem de tokens e métricas de retrieval.
   - Painel web em tempo real em `http://localhost:6006`.
7. **Infraestrutura como Código & CI/CD**:
   - `docker-compose.yml` pré-configurado para Qdrant e Arize Phoenix.
   - Workflow do GitHub Actions para validação com Ruff e Pytest.

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia | Finalidade |
| :--- | :--- | :--- |
| **Linguagem** | Python 3.11+ | Runtime principal |
| **Ingestão** | `requests`, `tenacity` | Coleta HTTP resiliente com retries e backoff |
| **Contratos** | `pydantic` v2, `pydantic-settings` | Validação de dados e configurações de ambiente |
| **Processamento** | `beautifulsoup4`, `langchain-text-splitters`, `tiktoken` | Sanitização HTML, divisão semântica e tokens |
| **Vector DB** | `qdrant-client` | Armazenamento de vetores e busca semântica |
| **Embeddings** | `fastembed` (ONNX local) / `openai` | `intfloat/multilingual-e5-large` — multilingual e treinado para retrieval |
| **Observabilidade** | `opentelemetry-sdk`, exportador OTLP/HTTP | Traces, spans, latência e monitoramento RAG (o servidor Arize Phoenix roda como container, não como dependência Python) |
| **Geração (RAG)** | `openai` (extra opcional) | Cliente compatível com OpenAI: NVIDIA NIM, OpenRouter, Ollama |
| **CLI & UI** | `typer`, `rich` | Interface de linha de comando elegante |
| **Testes & Lint** | `pytest`, `pytest-cov`, `ruff` | Qualidade de software e cobertura de código |
| **Containers** | `docker`, `docker compose` | Infraestrutura local de serviços |

---

## 📁 Estrutura do Repositório

```text
copom-rag-lakehouse/
├── .github/
│   └── workflows/
│       └── ci.yml                 # CI: Linting com Ruff e testes com Pytest
├── data/
│   ├── bronze/                    # JSONs brutos particionados year=YYYY/month=MM/
│   └── silver/                    # Documentos limpos e metadados normalizados
├── docker-compose.yml             # Serviços: Qdrant (6333) e Arize Phoenix (6006)
├── Makefile                       # Comandos de automação do ciclo de vida
├── pyproject.toml                 # Dependências e configurações de ferramentas
├── README.md                      # Documentação técnica e arquitetura
├── src/
│   ├── __init__.py
│   ├── config.py                  # Settings com Pydantic BaseSettings / .env
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── bcb_client.py          # Cliente HTTP resiliente da API BCB
│   │   ├── collector.py           # Coletor Bronze idempotente com hash SHA-256
│   │   └── schemas.py             # Schemas Pydantic da camada Bronze
│   ├── processing/
│   │   ├── __init__.py
│   │   ├── cleaner.py             # Sanitização de texto e remoção de ruídos
│   │   ├── chunker.py             # Chunking com overlap semântico e metadados
│   │   └── schemas.py             # Schema do ChunkPayload (doc_id, chunk_id, hash)
│   ├── vectorstore/
│   │   ├── __init__.py
│   │   ├── embeddings.py          # Gerador de embeddings em batch (FastEmbed / OpenAI)
│   │   └── qdrant_manager.py      # Criação de coleção, payloads e upsert idempotente
│   ├── observability/
│   │   ├── __init__.py
│   │   └── tracer.py              # Instrumentação Phoenix / OpenTelemetry
│   └── pipeline.py                # Script CLI unificado (ingest, transform, index, query)
└── tests/
    ├── __init__.py
    ├── conftest.py                # Fixtures e dados sintéticos
    ├── test_ingestion.py          # Testes unitários com mocks de requests
    ├── test_processing.py         # Testes de chunking e sanitização
    └── test_vectorstore.py        # Testes de contratos, UUIDs e busca vetorial
```

---

## 🚀 Como Executar o Projeto

### 1. Pré-requisitos
- Python 3.11 ou superior
- Docker & Docker Compose (opcional para execução com serviços em container)

### 2. Configuração do Ambiente Virtual

```bash
# Criar e ativar ambiente virtual
python -m venv .venv

# No Windows (PowerShell):
.venv\Scripts\activate

# No Linux / macOS:
source .venv/bin/activate

# Instalar o projeto em modo editável com dependências de desenvolvimento
pip install -e ".[dev]"
```

### 3. Subir a Infraestrutura (Qdrant + Phoenix)

```bash
# Iniciar Qdrant (6333) e Arize Phoenix (6006)
make up
# ou: docker compose up -d
```

- **Qdrant Dashboard**: [http://localhost:6333/dashboard](http://localhost:6333/dashboard)
- **Arize Phoenix Tracing**: [http://localhost:6006](http://localhost:6006)

---

## 🕹️ Execução do Pipeline (CLI)

O pipeline pode ser executado passo a passo ou ponta-a-ponta via CLI:

### 1. Ingestão da Camada Bronze (Raw)
Baixa as atas mais recentes da API oficial do BCB, valida os dados e grava os arquivos particionados no disco:
```bash
python -m src.pipeline ingest --limit 10
# ou: make ingest
```

### 2. Transformação da Camada Silver (Cleaned & Chunked)
Sanitiza o conteúdo HTML, divide o texto em chunks com overlap e enriquece os metadados:
```bash
python -m src.pipeline transform
# ou: make process
```

### 3. Indexação na Camada Gold (Qdrant Vectorstore)
Gera embeddings em lote e realiza o upsert idempotente no Qdrant:
```bash
python -m src.pipeline index --batch-size 32
# ou: make index
```

### 4. Execução Completa Ponta-a-Ponta
Executa Bronze ➡️ Silver ➡️ Gold de forma orquestrada com rastreamento Phoenix:
```bash
python -m src.pipeline run-all --limit 15
# ou: make run-pipeline
```

### 5. Pergunta e Resposta com Citações (RAG completo)
Recupera os trechos, gera a resposta fundamentada **apenas** neles e mostra as fontes:
```bash
python -m src.pipeline ask "por que o Copom manteve a Selic?" --limit 5
python -m src.pipeline ask "qual o balanço de riscos?" --year 2026 --meeting 280
```

Requer o extra opcional e uma chave:
```bash
pip install -e ".[openai]"
cp .env.example .env    # e preencha LLM_API_KEY
```

Sem `LLM_API_KEY` o comando **falha com mensagem explícita** em vez de responder sem fonte. O `ask` é o único comando que precisa de credencial; todo o resto do pipeline roda sem nenhuma.

O provedor é configuração, não código — o cliente fala qualquer endpoint compatível com a API da OpenAI:

| Provedor | `LLM_BASE_URL` | `LLM_MODEL` |
| :--- | :--- | :--- |
| NVIDIA NIM (padrão) | `https://integrate.api.nvidia.com/v1` | `moonshotai/kimi-k3` |
| OpenAI | `https://api.openai.com/v1` | `gpt-4o-mini` |
| OpenRouter | `https://openrouter.ai/api/v1` | `<publisher>/<model>` |
| Ollama (local, sem chave) | `http://localhost:11434/v1` | `llama3.1` |

### 6. Busca Semântica & Avaliação de Retrieval (retrieval puro)
Realize consultas em linguagem natural no Vector Store:
```bash
python -m src.pipeline query "cenário de inflação e taxa Selic" --limit 3
python -m src.pipeline query "balanço de riscos e atividade econômica" --year 2026
```

---

## 🧪 Qualidade de Código & Testes

Para rodar a suite completa de testes automatizados com cálculo de cobertura:

```bash
# Executar testes unitários e de integração
pytest --cov=src --cov-report=term-missing tests/
# ou: make test

# Executar linter e formatador Ruff
ruff check src/ tests/
ruff format src/ tests/
# ou: make lint / make format
```

---

## 📊 Observabilidade com Arize Phoenix

Durante todas as etapas do pipeline e nas consultas semânticas, o módulo `src/observability/tracer.py` emite spans compatíveis com OpenTelemetry para o coletor Arize Phoenix:

- **Traces de RAG (comando `ask`)**: spans aninhados seguindo as convenções semânticas do OpenInference — `CHAIN` (pergunta → resposta) contendo um `RETRIEVER` (documentos, ids e scores) e um `LLM` (modelo e contagem de tokens de prompt/completion). É isso que faz o Phoenix separar latência de recuperação da latência de geração, em vez de mostrar um bloco opaco.
- **Duração de cada etapa**: Ingestão, Limpeza, Tokenização, Geração de Vetores e Upsert.
- **Contabilidade de Tokens**: Volume de tokens processados por reunião e por chunk.
- **RAG Retrieval Traces**: Latência de busca no Qdrant e similaridade por Cosseno.

Acesse o painel do Phoenix em [http://localhost:6006](http://localhost:6006) para inspecionar traces e métricas detalhadas.

---

## 📜 Licença

Distribuído sob a licença MIT. Veja `LICENSE` para mais detalhes.
