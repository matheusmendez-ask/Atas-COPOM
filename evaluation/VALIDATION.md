# Validação local — 2026-09-10

- Python 3.11.3, Windows: **110 testes passaram**, cobertura de `src` **86%**.
- `ruff check --no-cache src tests scripts` e `ruff format --check src tests scripts`: aprovados.
- `git diff --check`: aprovado.
- Navegador: campos de reunião/ano, exibição dos filtros aplicados, link oficial, alerta de número ausente na fonte citada e navegação da citação até o trecho verificados com resposta fictícia local. O fixture não chama LLM.
- Layout verificado em 390×844 e 320×740, sem overflow horizontal; em 320 px também foi verificada a resposta com fontes. Foram verificações manuais no navegador, não testes automatizados de regressão visual.
- A página real abriu com Qdrant indisponível e exibiu esse estado sem carregar embeddings para a consulta de status.
- Os oito casos respondíveis do novo conjunto de teste tiveram suas âncoras verificadas nas respectivas atas 266–269. Nenhum placar do conjunto de teste foi usado para ajustar parâmetros.

Comando da suíte:

```bash
python -m pytest -p no:phoenix --cov=src --cov-report=term tests/
```

A opção `-p no:phoenix` foi necessária por um plugin residual no ambiente local que falha no Python 3.11 antes de coletar testes. O pacote Phoenix não foi reinserido nas dependências do projeto; o servidor continua no Docker. O teste não precisou dele.

As verificações de conteúdo são heurísticas numéricas; não comprovam suporte semântico de cada afirmação. Respostas inválidas são preservadas nos relatórios de avaliação e contam como falha de validação, evitando inflar métricas ao excluir essas saídas.

Não foi executada uma nova avaliação de geração com API nem uma nova avaliação no Qdrant servidor. Os containers estavam indisponíveis; os experimentos de recuperação usam Qdrant em memória.


## Comparação controlada de recuperação — 2026-09-10

Qdrant **em memória**, mesmas 15 atas (266–280), 575 chunks de 150 tokens com overlap 25, `intfloat/multilingual-e5-large`, FastEmbed 0.8.0, BM25 em português e DBSF. As 30 perguntas são de desenvolvimento: 22 respondíveis e 8 de recusa. Não houve geração nesta execução. Os hashes dos snapshots do corpus são iguais nos dois relatórios.

| Métrica | Sem filtro automático | Com filtro automático |
| :--- | ---: | ---: |
| hit@1 | 15/22 (68,2%) | 17/22 (77,3%) |
| hit@3 | 19/22 (86,4%) | 20/22 (90,9%) |
| hit@5 | 19/22 (86,4%) | 20/22 (90,9%) |
| MRR | 0,765 | 0,833 |
| Reunião esperada recuperada | 5/6 | 6/6 |

`focus-278` passou de não recuperado para rank 1; `focus-270`, de rank 2 para rank 1. Permanecem sem recuperação no top 5: `manutencao-nivel-corrente` e `elevacao-025-271`.

O baseline aleatório recalculado no corpus completo, sem filtros, foi 0,324%. Ele não é o baseline de uma busca restrita a uma reunião. A amostra é pequena e foi usada no desenvolvimento; não representa uma avaliação independente de generalização.

Relatórios completos, com configuração, versões, hashes, snapshots e resultados por pergunta:

- [Sem filtro automático](results/2026-09-10-without_meeting_filter.json)
- [Com filtro automático](results/2026-09-10-with_meeting_filter.json)

Reproduzir sobre as atas já presentes no Bronze:

```bash
python -m src.experiments --chunk-sizes 150
```

A execução anterior foi usada para diagnosticar um filtro excessivamente conservador que confundia os anos 2026 e 2027 com números de reunião. Os relatórios acima correspondem à correção validada por testes. O conjunto de teste separado não foi usado nessa decisão.
