# Avaliação

`golden_set.json` contém as 30 perguntas de **desenvolvimento** usadas nos ajustes históricos.
`test_set.json` contém 10 perguntas novas, reservadas para avaliação final. Não use seu placar para escolher parâmetros. Separar as 30 antigas retroativamente não produziria um teste independente.

```bash
python -m src.pipeline evaluate --retrieval-only --output evaluation/runs/dev.json
python -m src.pipeline evaluate --retrieval-only --golden-set evaluation/test_set.json --output evaluation/runs/test.json
python -m src.pipeline evaluate --with-generation --delay 25 --output evaluation/runs/generation.json
python -m src.experiments --chunk-sizes 150 250 400 800
```

O experimento recria o índice em memória a partir das versões correntes do Bronze e compara a inferência de reunião desligada/ligada sobre os mesmos vetores. Não altera o índice do servidor. Os scripts antigos de reranking não foram recuperados: suas tabelas são registros históricos, não experimentos reproduzidos por este comando.

Cada JSON inclui perguntas, respostas, passagens recuperadas, tempos, configurações sem segredos, versões instaladas, commit, hash do código, hash do prompt e snapshot dos payloads efetivamente indexados. O baseline aleatório é recalculado no corpus completo, sem filtros, e não deve ser interpretado como a dificuldade equivalente de uma busca filtrada. Configurações registradas não provam com qual versão um índice antigo foi criado; reindexe para uma comparação controlada.

Os relatórios não sobrescrevem arquivos existentes. `runs/` fica ignorado por padrão; publique apenas resultados revisados com `git add -f evaluation/runs/arquivo.json`. Não há resultados novos do servidor até executar o comando com Qdrant disponível.

Citações dentro da faixa verificam somente IDs. Presença de citação e números ausentes nos trechos citados são checagens separadas; nenhuma comprova suporte semântico, causalidade ou correção de cada afirmação. Recusa é uma heurística textual. O conjunto de teste precisa de segundo anotador antes de sustentar conclusões de generalização.


Os resultados revisados da comparação de filtros estão em [VALIDATION.md](VALIDATION.md) e `results/`; os arquivos em `runs/` permanecem como registros locais de experimentação.
