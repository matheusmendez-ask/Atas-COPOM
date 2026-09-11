# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Comandos

```bash
make setup          # cria .venv e instala -e ".[dev]"
make up / make down # sobe/derruba Qdrant (6333) e Phoenix (6006) via docker compose
make test           # pytest --cov=src --cov-report=term-missing tests/
make lint           # ruff check src/ tests/
make format         # ruff format src/ tests/
```

Um teste só:
```bash
python -m pytest tests/test_ingestion.py::TestIngestionSchemas::test_raw_ata_item_validation
```

**Não reintroduza `arize-phoenix` nas dependências de runtime.** O código nunca importa `phoenix` — só o exportador OTLP/HTTP; o servidor Phoenix roda como container. Instalar o pacote traz o `arize-phoenix-client`, que registra um plugin pytest (entry point `pytest11` chamado `phoenix`) carregado automaticamente; ele importa `phoenix`, que usa `mappingproxy` como default de dataclass — aceito só a partir do 3.12. Com ele instalado, **no Python 3.11 o pytest nem chega a coletar** (o projeto declara `requires-python = ">=3.11"` e a CI testa 3.11).

Estado local medido em 2026-09-10: **110 passed, 86% de cobertura**, Python 3.11.3; CI desta alteração ainda não executada. Números fixos em documento defasam — três vezes nesta base já. A CI publica `coverage.xml` como artefato; um Codecov encerraria o problema.

Pipeline (cada etapa roda isolada; o disco é a interface entre elas):
```bash
python -m src.pipeline ingest --limit 10
python -m src.pipeline transform
python -m src.pipeline index --batch-size 32
python -m src.pipeline run-all --limit 15
python -m src.pipeline query "cenário de inflação" --limit 3 --year 2026 --meeting 280
python -m src.pipeline ask "por que o Copom manteve a Selic?" --limit 5
```

```bash
python -m src.pipeline evaluate --limit 5                      # so recuperacao, sem credencial
python -m src.pipeline evaluate --with-generation --delay 25   # inclui fatos, citacoes e recusa
```

**Reranking com cross-encoder já foi testado e REJEITADO** (2026-09-01). Não refaça sem medir antes: os três modelos do FastEmbed pioraram o hit@1 — jina-reranker-v2-multilingual 73%→55% a 5,16s/pergunta, bge-reranker-base 41%, ms-marco-MiniLM 64%, contra 0,16s/pergunta sem reranker. Reordenam por relevância de domínio geral, enquanto aqui a distinção está em números e em qual reunião. A tabela completa está no README.

**Antes de mexer em chunking, embeddings ou prompt, rode o `evaluate` e anote o número.** Estado conhecido em 2026-08-31, medido no **Qdrant servidor** com 575 chunks de 15 reuniões (266–280): hit@1 68%, hit@3 86%, hit@5 86%, MRR 0,765, proveniência 83%, fatos 83%, citações 100%, recusa 8/8. O baseline de 0,4% pertencia ao corpus em memória; não foi recalculado no registro histórico do servidor.

**Números maiores em relatos antigos vieram de Qdrant em memória sobre 403 chunks de 11 reuniões** (hit@1 73%). Servem para comparar configurações entre si, não com o servidor — o corpus real tem 172 chunks a mais de distratores. Ao medir, diga sempre qual corpus usou.

**A busca é híbrida: denso + BM25, fundidos por DBSF no servidor.** A coleção usa vetores NOMEADOS (`dense` + `bm25` esparso com `Modifier.IDF`) — coleções antigas de vetor anônimo são rejeitadas com mensagem explícita. **Não troque para RRF sem medir**: ele parece a escolha principiada (funde por posição, dispensa normalização), mas derruba a proveniência de 100% para 83%, porque promove passagens em que os dois métodos concordam e rebaixa a que só a densa achou. `FUSION_METHOD` e `PREFETCH_MULTIPLIER` estão no topo do `qdrant_manager` com a tabela da medição. Números maiores que estes em relatos antigos vieram do conjunto de 11 perguntas, que era otimista. Ancore o gabarito em frases, nunca em `chunk_id`.

**Geração medida (gemini-3.7-flash, 2026-08-31): recusa 8/8 nas armadilhas, citações 100% válidas, fatos 5/6.** O sistema não alucina; o elo fraco é a recuperação. Antes de mexer no prompt, confira se o problema não é a busca não ter entregado a passagem — o único fato não confirmado foi o modelo recusando corretamente por falta de contexto.

**`ChunkPayload.embedding_text` existe por um motivo medido.** O vetor é calculado sobre o trecho prefixado com "Ata da Nª reunião do Copom, publicada em ...", e não sobre `text` puro. Sem isso o sistema recuperava o tópico certo do **documento errado** (hit@1 27%, proveniência 40%): as atas são formulaicas e `nro_reuniao` só existia no payload, que filtra mas não embute. **Não passe `chunk.text` direto ao embedder** — `upsert_chunks` usa `embedding_text` de propósito. Mudar o prefixo exige reindexar e rerodar o `evaluate`.

**O free tier da NVIDIA estrangula rajadas.** As 30 perguntas do gabarito disparam uma chamada cada; sem `--delay` o limite corta na primeira e, uma vez estourada a cota, ela recusa até chamada única por vários minutos. O backoff do `_complete` não resolve isso sozinho — a janela do limite dura mais que qualquer retry razoável. Use `--delay 25`.

`query` é retrieval puro (sem credencial); `ask` fecha o loop de RAG e é **o único comando que exige credencial** (`LLM_API_KEY` + extra `pip install -e ".[openai]"`). Quando a resposta sair ruim, use `query` para ver o que o retrieval de fato trouxe antes de culpar o prompt.

**Antes de commitar:** a CI roda `ruff check` **e** `ruff format --check`. `make lint` só roda o `check` — passar nele não garante CI verde. Rode `make format` também.

**`make <target>` usa o `python` do PATH, não o `.venv`** (só o alvo `setup` usa `VENV_PYTHON`). Com venv ativado tudo bem; sem ativar, `make test` roda no Python global.

## Arquitetura

Lakehouse medalhão em 3 camadas, orquestrado por um CLI Typer (`src/pipeline.py`). Cada etapa lê o estado da anterior **do disco**, não da memória — por isso são idempotentes e re-executáveis isoladamente.

```
BCB API ──ingest──> data/bronze/year=YYYY/month=MM/{doc_id}_{short_hash}.json
                    └─ BronzeAtaRecord (raw_content HTML + SHA-256)
         ──transform─> data/silver/year=YYYY/month=MM/{doc_id}_silver.json
                    └─ SilverDocument { clean_text, chunks: [ChunkPayload] }
         ──index────> Qdrant collection `copom_minutes` (dense 1024d + BM25 esparso)
                    └─ point id = UUIDv5(NAMESPACE_URL, "doc_id:chunk_id")
```

- **`src/config.py`** — `settings` é um **singleton criado no import** (`Settings()` no fim do módulo). Toda config vem daqui ou de `.env`; nada de env var lida direto no meio do código.
- **`src/observability/tracer.py`** — `tracer` também é singleton de import, e o construtor chama `trace.set_tracer_provider(...)`, ou seja, **importar `src.pipeline` já inicializa o OpenTelemetry global**. Sem Phoenix rodando ele cai em `ConsoleSpanExporter`; para silenciar, `ENABLE_PHOENIX=false`.
- **`src/ingestion/`** — `bcb_client` (HTTP resiliente) → `collector` (idempotência + escrita Hive) → `schemas` (contratos).
- **`src/processing/`** — `cleaner` (BeautifulSoup + normalização) → `chunker` (split + enriquecimento) → `schemas`.
- **`src/vectorstore/`** — `embeddings` (provider abstrato) → `qdrant_manager` (coleção, upsert, busca).
- **`src/web/`** — interface local (`python -m src.web`, extra `[web]`). `create_app(answerer)` recebe o answerer por injeção, então os testes rodam sem modelo e sem Qdrant. A página é um único HTML sem build; o CSS usa OKLCH com neutros de papel quente e **cor só para estado** — recusa em azul informativo, nunca vermelho, porque recusar é acerto.
- **`src/generation/`** — `answerer` fecha o loop de RAG: `retrieve()` numera as passagens, `build_messages()` monta o prompt com citações, `generate()` chama o endpoint compatível com OpenAI. **Nada de fornecedor específico no código** — trocar NVIDIA/OpenAI/OpenRouter/Ollama é só `LLM_BASE_URL` + `LLM_MODEL`.

### Idempotência (o ponto central do projeto)

Três mecanismos distintos, cada um com uma pegadinha:

1. **Bronze — append-only, hash no nome do arquivo.** `content_hash` = SHA-256 do `raw_content`; o arquivo é `{doc_id}_{short_hash}.json`. Conteúdo alterado na origem **gera um arquivo novo em vez de substituir o antigo** — as versões acumulam na partição de propósito (histórico de auditoria), e `load_all_records()` devolve todas elas.
2. **Promoção Bronze→Silver — versão corrente.** O Silver indexa por `doc_id` (`{doc_id}_silver.json`, sem hash), então só pode receber **um** registro por documento. `BronzeCollector.select_current_versions()` elege o de `ingested_at` mais recente, com desempate por `content_hash` para nunca depender da ordem do filesystem. **Sempre passe o histórico por ele antes de promover** — chamar `load_all_records()` direto reintroduz um bug em que a versão obsoleta vence quando o hash dela ordena depois.
3. **Gold — UUIDv5 determinístico.** `generate_point_id(doc_id, chunk_id)` garante que reindexar atualiza o ponto em vez de duplicar.

`transform` reprocessa as versões correntes do Bronze. `index` sincroniza documentos completos: reaproveita payloads inalterados com a mesma assinatura de embedding e remove IDs obsoletos somente depois de concluir o documento. Uma falha pode deixar upserts parciais; repetir a sincronização os repara. Não há transação entre todos os pontos.

### Embeddings: multilingual, assimétrico e sem fallback silencioso

O default é `intfloat/multilingual-e5-large` (1024-d). A escolha foi medida, não chutada — `hit@1` sobre 102 chunks de 16 reuniões: e5-large 9/10, `bge-small-en` 9/10, `paraphrase-multilingual-MiniLM-L12-v2` 6/10. **Não troque para o MiniLM**: ele é multilingual, mas treinado para paráfrase simétrica, e erra 3 perguntas que os outros acertam.

A busca é assimétrica: `embed_texts()` usa `passage_embed()` e `embed_query()` usa `query_embed()`. Para modelos e5 é o FastEmbed que injeta os prefixos `query:`/`passage:` aí dentro; para modelos simétricos os dois viram um `embed` comum. **Não unifique os dois caminhos** — foi assim que o retrieval ficava errado antes.

O e5-large baixa 2,2 GB no primeiro uso, então a CI sobrescreve `EMBEDDING_MODEL_NAME`/`EMBEDDING_DIMENSION` para o MiniLM pequeno no passo de testes (os testes exercitam encanamento, não qualidade de recuperação). Trocar de modelo com largura diferente exige recriar a coleção do Qdrant — `ensure_collection()` levanta `ValueError` se a dimensão não bater.

### Embeddings falham alto — não degradam

`EmbeddingGenerator` levanta `EmbeddingUnavailableError` em vez de substituir o modelo ou inventar vetores. Os quatro caminhos: modelo configurado não carrega, pacote `fastembed` ausente, `provider=openai` sem `OPENAI_API_KEY`, cliente OpenAI falha ao iniciar.

**Não reintroduza fallback aqui.** A versão anterior trocava em silêncio pelo modelo default (inglês, 384-d — dimensão que às vezes batia, então nada denunciava) e, no limite, gerava vetores pseudo-aleatórios de SHA-256. O único sintoma era busca ruim, que aponta para o chunking em vez da causa real. Os testes não precisam disso: eles injetam um modelo falso direto em `_model`.

`ensure_collection()` valida o vetor denso nomeado e sua dimensão. A sincronização registra a identidade dos modelos e a versão do FastEmbed para evitar reaproveitar embeddings de outra configuração.

### Detalhes que não são óbvios pelo código

- **O rodapé de presença é removido no Silver.** `AtaCleaner.strip_attendance_roster()` corta a partir do marcador `Presentes:` — são ~14% do corpus em nomes/cargos de participantes e uma frase de encerramento idêntica entre atas. Só corta se o marcador estiver após 50% do texto (nas atas amostradas ele nunca aparece antes de 79%); caso contrário loga um aviso e mantém tudo. O Bronze segue com o texto íntegro.
- **A geração falha alta, de propósito.** Sem `LLM_API_KEY`, `generate()` levanta `GenerationUnavailableError` com instrução do que setar. Não adicione fallback — responder sem fonte é pior que não responder, e é a mesma armadilha do fallback de embeddings acima.
- **Os spans do `ask` usam convenções do OpenInference** (`openinference.span.kind`, `retrieval.documents.N.document.*`, `llm.token_count.*`) para o Phoenix renderizar trace de RAG. São só atributos de span: **não é preciso a dependência `openinference-instrumentation`**, que foi removida de propósito.
- `QdrantManager.search()` devolve `[]` quando a coleção não existe (com aviso no log), em vez de estourar traceback — `query` e `ask` já tratam resultado vazio com mensagem útil.
- **`CHUNK_SIZE=150` foi medido, não arbitrado.** Com 800 tokens o hit@1 era 36%; com 150, 73%. Um número que distingue uma ata das outras quase não move um vetor de 800 tokens de prosa formulaica. **O `.env` sobrescreve o default do código** — se `evaluate` der números de 800, confira o `.env` antes de suspeitar do código. Mudar exige reindexar e recalcular os baselines do gabarito (que dependem do tamanho do corpus).
- `CHUNK_SIZE` e `CHUNK_OVERLAP` são **tokens, não caracteres**: o `RecursiveCharacterTextSplitter` recebe `length_function=TokenCounter.count` (tiktoken `cl100k_base`, com fallback heurístico de ~4 chars/token).
- A API do BCB é camelCase (`nroReuniao`, `textoAta`, `dataPublicacao`); os schemas Pydantic usam snake_case com `alias=`. Ao mexer em campos novos, adicione o alias.
- **`RawAtaDetail` valida o payload de detalhes na fronteira** — é dele que sai todo o texto do pipeline. Atas anteriores a ~2021 trazem `textoAta` nulo (só publicaram PDF) e são rejeitadas com motivo; `run()` as conta em `failed` e segue. Verificado ao vivo: reuniões 240–280 passam, 220 e 230 não.
- **Data de publicação ilegível levanta erro**, não vira "hoje". O fallback antigo arquivava uma ata de 2019 em `year=2026/month=08` e o filtro `--year 2019` nunca mais a encontrava.
- `BCBClient` converte **429 e 5xx em `BCBClientError`** justamente para que o Tenacity os capture e faça backoff — `raise_for_status()` sozinho não daria retry.
- `BCB_ODATA_URL` / `fetch_odata_publications()` existem na config e no cliente mas **não são usados** pelo pipeline; o fluxo real usa `sitebcb/copom/atas` e `sitebcb/copom/atas_detalhes`.
- `src/pipeline.py` reconfigura `stdout`/`stderr` para UTF-8 antes dos imports do projeto (Windows). Mantenha os imports do `src.*` depois desse bloco.
- `run-all` chama `ingest()`, `transform()` e `index()` como funções Python normais, não como subprocessos.
- `data/bronze` e `data/silver` são gitignored exceto pelos `.gitkeep`.

## Testes

`tests/conftest.py` traz HTML sintético de ata (com as seções A/B/C reais do COPOM), respostas mockadas dos dois endpoints e um `in_memory_qdrant` usando `QdrantClient(location=":memory:")`. **A suíte não precisa de Docker nem de rede** — o `requests` é mockado e o Qdrant é em memória. Se um teste novo exigir serviço externo, é sinal de que o desenho está errado.


## Avaliação e sincronização (2026-09-10)

- As 30 perguntas existentes são desenvolvimento; `evaluation/test_set.json` é teste prospectivo. Não ajuste parâmetros com seu placar.
- `evaluate --retrieval-only --output caminho.json` salva snapshot real do índice, configuração sem segredos, versões, hashes e resultados por pergunta. Nunca publique chaves ou conteúdo de `.env` em relatórios.
- `python -m src.experiments --chunk-sizes 150` compara inferência de reunião com os mesmos vetores em memória; não altera o servidor.
- `sync_documents` aceita documentos completos; `upsert_chunks` continua disponível para lotes parciais sem remoção. Limpeza só após sucesso por documento; não é transacional e exige um único escritor durante indexação.
- A verificação de citações separa IDs, presença e números nos trechos citados; `semantic_support=not_verified`. Não chame isso de prova de ausência de alucinação.
- `python scripts/demo.py` inicia a demonstração; `--prepare` também executa coleta, transformação e indexação.


Medição em memória de 2026-09-10 (15 atas, 575 chunks, FastEmbed 0.8.0): inferência de reunião mudou hit@1 15/22 → 17/22, hit@5 19/22 → 20/22, MRR 0,765 → 0,833 e proveniência 5/6 → 6/6. `focus-278` agora rank 1. Relatórios em `evaluation/results/`; método em `evaluation/VALIDATION.md`. Não apresentar isso como medição nova do servidor ou de geração.
