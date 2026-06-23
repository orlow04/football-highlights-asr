# Experimento 2b — Generalização do mecanismo de detecção (SoccerNet-Echoes)

**Projeto:** Detecção de highlights em narrações de futebol via fusão ASR + SER
**Escopo deste documento:** justificar metodologicamente o uso do SoccerNet-Echoes,
delimitar o que ele valida (e o que não valida) e documentar o pipeline `src/soccernet_echoes.py`, em nível de relatório técnico.

---

## 1. Papel no desenho experimental

O trabalho sustenta três afirmações distintas (ver `estrategia_validacao.md` §3). O SoccerNet-Echoes sustenta **apenas a terceira**, que é **lateral**:

> *"Para verificar se a arquitetura de detecção generaliza além do português e de n pequeno, o módulo de detecção foi avaliado sobre M jogos do SoccerNet com ground-truth de ação humano, usando as transcrições Echoes. Esta avaliação isola o mecanismo de detecção da etapa de ASR e do domínio linguístico."*

Ele **não** é a validação central — esta é o Exp. 2a (fusão ASR+SER sobre os jogos PT anotados). O 2b existe para responder a uma única pergunta de robustez: *o detector léxico-temporal, que ajustamos em poucos jogos PT, continua localizando eventos quando aplicado, sem retreino, a muitos jogos de outra liga e outro idioma, contra ground-truth independente?*

## 2. O que o 2b valida — e o que NÃO valida

| Valida | Não valida |
|---|---|
| O **mecanismo de detecção**: léxico ponderado → curva por 1 s → pico (`mean + k·σ`) → casamento por tolerância ±τ | O **ASR** — a transcrição é do Whisper, não humana; WER contra ela mediria concordância entre ASRs, não acurácia |
| **Robustez a n grande** (dezenas/centenas de jogos) e a **ground-truth humano** (Labels-v2 de ação) | O **português** — o dataset é EN/ES/etc., sem PT |
| A **transferência do desenho léxico** para outro idioma (léxico EN análogo) | A **fusão multimodal** — não há áudio no Echoes, logo **não há SER** |

Três fatos tornam essas exclusões inegociáveis:

1. **Não há áudio no SoccerNet-Echoes.** O dataset distribui só transcrições (`{half}_asr.json`). O Parakeet nunca processa nada dele; o componente acústico (SER) é, por construção, inaplicável.
2. **Não há português** entre os idiomas. Nada se conclui sobre adaptação ao PT-BR.
3. **As transcrições são geradas por Whisper.** Carregam erro de ASR; a detecção léxica sobre elas herda esse ruído. Por isso o 2b mede **robustez do mecanismo**, não desempenho absoluto.

## 3. Consequência de desenho: 2b é detecção SÓ-LÉXICA

Como não há áudio, o 2b roda **apenas o ramo lexical** do pipeline (`α=1, β=0`), não a fusão. Isso é coerente e honesto: o 2b testa a parte do sistema que **independe do áudio** — a transformação de pistas textuais de domínio em uma curva temporal e a detecção de picos sobre ela. O ramo SER e a fusão permanecem avaliados exclusivamente no Exp. 2a (jogos PT), onde há áudio real.

Implicação para o relatório: **não comparar diretamente F1(2a) com F1(2b)**. São regimes diferentes (multimodal em PT vs. léxico-só em EN). O 2b sustenta "o mecanismo generaliza", não "a fusão atinge F1 X em geral".

## 4. Dados — o que baixar (e o que NÃO baixar)

| Item | Fonte | Tamanho | Papel |
|---|---|---|---|
| `Labels-v2.json` | cliente pip `SoccerNet` | leve | ground-truth de evento (humano) |
| Transcrições Echoes (`whisper_v3/...`) | repo git `SoccerNet/sn-echoes` | leve (texto) | entrada lexical |
| ~~Vídeo/áudio LQ~~ | ~~NDA~~ | ~~centenas de GB~~ | **NÃO baixar — não há ASR a rodar** |

Formato Echoes: `whisper_version/liga/temporada/jogo/{half}_asr.json`, com
`{"segments": {idx: [start, end, text], ...}}` (tempos em segundos). Os Labels ficam na mesma árvore `liga/temporada/jogo/Labels-v2.json` (`gameTime "h - mm:ss"`, `position` em ms desde o início da metade).

**Premissa de alinhamento** (declarada): os tempos de `{half}_asr.json` e o `position/1000` dos labels da **mesma metade** compartilham a origem (início daquela metade). Por isso processa-se **uma metade por vez** (`--half`), nunca misturando as duas.

## 5. Pipeline (`src/soccernet_echoes.py`)

1. **Emparelhamento** — para cada `{half}_asr.json` sob `--echoes-root`, localiza o `Labels-v2.json` do mesmo jogo sob `--labels-root` (mesmo caminho relativo `liga/temporada/jogo`).
2. **Score lexical EN** — normaliza o texto de cada segmento (minúsculas, sem acento/pontuação) e pontua o léxico de domínio em inglês por janela de 1 s, no instante de início do segmento. Casamento por limite de palavra (`\b`) evita falsos (`goal` em `goalkeeper`); chaves multi-palavra (`red card`) casam por substring. Negador no mesmo segmento (`almost`, `offside`, `wide`…) atenua o peso.
3. **Detecção** — picos por `mean + k_sigma·σ` com distância mínima (mesma `detect.py` do 2a), varrendo `k_sigma`.
4. **Avaliação** — P/R/F1 por tolerância τ ∈ {5, 10, 15}s contra os labels, casamento guloso 1-para-1 (mesma `evaluate.py`).
5. **Agregação** — uma linha por **jogo** + linha **AGREGADO** (micro: soma TP/FP/FN antes de P/R/F1). Saída em `out/e2b/e2b_results.csv` e wandb opcional.

Todos os parâmetros (léxico EN, pesos, negadores, `relevant_labels`, `k_sigma`) vivem em `configs/params.yaml::soccernet` / `experiments.e2`, versionados e sujeitos a ablação.

## 6. Como reportar

- Tabela **por jogo + AGREGADO**, F1×τ, com a varredura de `k_sigma`.
- Frase-alvo: *"sobre M jogos do SoccerNet com ground-truth humano, o detector léxico-temporal mantém F1 de X (±τ=10s) sem retreino — evidência de que o mecanismo generaliza para além do português e de n pequeno."*
- Declarar que é **léxico-só, em inglês, desacoplado do ASR e do SER**, e que o ruído de Whisper é um piso de erro herdado.

## 7. Limitações declaradas

- **Erro de Whisper embutido.** A entrada não é texto perfeito; parte dos FN/FP vem da transcrição, não do detector. Mede-se robustez, não teto de desempenho.
- **Léxico EN como hiperparâmetro.** Definido por conhecimento de domínio e ablável; não aprendido de dados rotulados.
- **Sem SER.** O 2b não exercita o componente acústico nem a fusão — esses ficam só no 2a (PT, com áudio).
- **Cobertura de labels.** Só os `relevant_labels` (gol, pênalti, cartões, finalizações) contam como evento; a escolha é explícita e ablável.
