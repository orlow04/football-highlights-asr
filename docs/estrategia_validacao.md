# Estratégia de Validação e Argumentação Científica

**Projeto:** Detecção de highlights em narrações de futebol via fusão ASR + SER
**Propósito deste documento:** consolidar os argumentos de defesa, o papel de cada fonte de dados e as limitações assumidas, de modo que cada afirmação do trabalho seja sustentável sob escrutínio.

---

## 1. Tese central (o que o trabalho realmente defende)

> **Adaptação de domínio para narração esportiva em português brasileiro.**
> Um modelo de ASR pode reconhecer PT-BR genérico (e até fala espontânea, como podcasts) e ainda assim degradar diante do vocabulário e da prosódia específicos da narração de futebol — jargões ("gooool", "matou no peito", "pintura"), nomes próprios e a entonação exagerada do narrador. O trabalho quantifica esse gap e mostra que pistas acústicas de excitação (SER) complementam as pistas lexicais (ASR) na localização de momentos-chave.

Tudo que segue existe para sustentar essa tese — ou para delimitar honestamente onde ela não alcança.

---

## 2. O papel de cada fonte de dados (e por que não se confundem)

Esta separação é o eixo da defesa. As duas fontes provam coisas **diferentes e não-sobrepostas**; apresentá-las como se provassem a mesma coisa enfraqueceria o trabalho.

| Dimensão | Jogos em português (anotados por nós) | SoccerNet-Echoes |
|---|---|---|
| Idioma | PT-BR | Inglês, espanhol, etc. (**sem PT**) |
| Contém áudio? | **Sim** (baixado do YouTube) | **Não** — só texto |
| Roda o ASR Parakeet? | **Sim** | Não (não há áudio) |
| Transcrição é verdade-fundamental? | **Sim** (humana) | **Não** (gerada por Whisper) |
| Ground-truth de eventos | Anotação manual nossa | Labels de ação anotados por humanos (SoccerNet original) |
| O que valida | A **tese central** (ASR + fusão em PT) | A **generalização do mecanismo** de detecção |

### 2.1 Por que o SoccerNet NÃO valida o ASR

Três fatos se acumulam e tornam isso inegociável:

1. **Não há áudio no SoccerNet-Echoes** — ele distribui apenas transcrições (índice do segmento, tempos, texto, caminho do jogo). Logo, o Parakeet nunca processa nada dele.
2. **Não há português** entre os 10 idiomas do dataset. Não diz nada sobre adaptação ao PT.
3. **As transcrições são geradas por Whisper, não por humanos.** Calcular WER contra elas mediria *concordância entre dois ASRs*, não acurácia. Apresentar isso como validação de ASR seria um erro metodológico que um avaliador atento derrubaria imediatamente.

**Conclusão:** o SoccerNet contribui **apenas** para a etapa de detecção de highlight (Experimento 2), e ainda assim em inglês, isolado do ASR.

### 2.2 O que o SoccerNet legitimamente agrega

Os **labels de ação anotados por humanos** (gol, cartão, substituição, com timestamp) são ground-truth genuíno. Com eles, sobre muitos jogos, podemos avaliar:

- a detecção léxica (léxico adaptado ao inglês: "goal", "penalty", "save"…) sobre as transcrições Echoes;
- a lógica de fusão e de detecção de picos;
- P/R/F1 com **n grande**.

Isso sustenta a frase: *"o mecanismo de detecção generaliza para N jogos com ground-truth humano"* — uma afirmação **lateral** e honesta, não a tese central.

---

## 3. As três afirmações defensáveis do trabalho

O desenho foi montado para produzir exatamente três frases sustentáveis, sem que nenhuma finja ser a outra:

1. **Adaptação de domínio (tese central, ASR).**
   *"O fine-tune PT-BR melhora o WER global frente ao base multilíngue, mas ambos mantêm erro elevado no léxico de domínio futebolístico — evidenciando que adaptação de idioma ≠ adaptação de domínio."*
   Sustentada por: WER/CER e WER-no-léxico sobre os jogos PT anotados.

2. **Ganho da fusão multimodal (tese central, detecção em PT).**
   *"Em N jogos de narração brasileira, a fusão ASR+SER supera cada modalidade isolada na detecção de highlights (F1 com tolerância ±τ)."*
   Sustentada por: ablação SÓ-ASR / SÓ-SER / FUSÃO sobre os jogos PT.

3. **Generalização do mecanismo (afirmação lateral, SoccerNet).**
   *"Para verificar se a arquitetura de fusão generaliza além do português e de n pequeno, o módulo de detecção foi avaliado sobre M jogos do SoccerNet com ground-truth de ação humano, usando as transcrições Echoes. Esta avaliação isola o mecanismo de detecção da etapa de ASR e do domínio linguístico."*
   Sustentada por: P/R/F1 sobre M jogos SoccerNet.

---

## 4. Limitações assumidas (declaradas por nós, antes da banca)

Antecipar o contra-argumento é o que distingue um projeto que **se defende** de um que precisa **ser defendido**. Estas limitações vão explícitas no relatório:

- **n e anotador no lado PT.** Os resultados de WER e detecção em português vêm de poucos jogos (alvo: ~6) e, na ausência de um segundo anotador, de anotação única. São apresentados como evidência consistente entre jogos, não como significância estatística forte.
- **SoccerNet não valida ASR nem PT.** Reafirmado: serve só ao mecanismo de detecção, em inglês, desacoplado do áudio.
- **Erro de Whisper embutido no Echoes.** As transcrições do SoccerNet carregam erros de ASR; a detecção léxica sobre elas herda esse ruído. Por isso o SoccerNet mede robustez do mecanismo, não desempenho absoluto.
- **Léxico como hiperparâmetro.** Os pesos do léxico foram definidos com conhecimento de domínio e sujeitos a ablação; não são aprendidos de dados rotulados em larga escala.
- **SER ≈ arousal acústico, não emoção categórica.** O componente mede excitação/intensidade vocal (energia, F0, taxa de fala), não categorias emocionais — escolha deliberada, pois é o arousal que prediz highlight e é robusto ao ruído de estádio.

---

## 5. O ponto mais frágil — e como mitigá-lo

A maior vulnerabilidade dos jogos PT **não é o n**, é ser **anotador único**. Mitigações, em ordem de impacto:

1. **Segundo anotador (alvo em andamento).** Se um colega anotar 1–2 dos jogos PT de forma independente, calcula-se **Cohen's κ**. Concordância alta vale mais para a credibilidade da tese do que dezenas de jogos do SoccerNet, porque ataca diretamente a crítica mais provável.
2. **Dupla anotação própria.** Na ausência de segunda pessoa, anotar cada jogo em duas passagens separadas (com intervalo) e reportar a consistência intra-anotador.
3. **Protocolo de anotação escrito.** Definição clara do que conta como highlight (gol, defesa difícil, trave, pênalti, cartão vermelho, grande chance perdida) e do que **não** conta (falta comum, lateral, posse de meio-campo). Marcar sempre o **instante do lance**, não o grito do narrador.

---

## 6. Plano de dados (estado e metas)

| Item | Estado | Meta |
|---|---|---|
| Jogos PT anotados | 1 concluído | ~6 (anotando +5) |
| Segundo anotador | a confirmar | 1 pessoa, 1–2 jogos → Cohen's κ |
| SoccerNet — transcrições Echoes | a baixar | subset (whisper_v3) |
| SoccerNet — labels de ação | a baixar | `Labels-v2.json` (train/valid/test) |

> **Nota sobre o que baixar do SoccerNet:** apenas os `Labels-v2.json` (leves, ground-truth de evento) e as transcrições Echoes (texto). **Não** baixar vídeo/áudio bruto (centenas de GB, exige NDA, e é inútil aqui — não há ASR a rodar sobre o SoccerNet).

---

## 7. Desenho experimental consolidado

### Experimento 1 — Adaptação de idioma vs. domínio (ASR) — **só jogos PT**
Comparar, sobre as mesmas narrações PT anotadas:
- **C1:** Parakeet v3 base (multilíngue) — PT genérico
- **C2:** fine-tune ptBR-plus — ganho de adaptação ao PT-BR
- **C3:** erro residual no subconjunto de palavras de domínio

Métricas: WER, CER, WER-no-léxico.
Hipótese: C2 < C1 no WER global, mas ambos com erro alto no léxico (C3).

### Experimento 2 — Ablação da fusão (detecção de highlight)
**(2a) Validação primária — jogos PT:** SÓ-ASR / SÓ-SER / FUSÃO sobre os ~6 jogos PT.
**(2b) Validação de generalização — SoccerNet:** mesmo mecanismo, léxico em inglês, sobre M jogos com labels humanos.

Métrica: P/R/F1 com tolerância ±τ (τ = 5, 10, 15 s) + varredura de α/β e do limiar.
Hipótese: F1(FUSÃO) > max(F1(SÓ-ASR), F1(SÓ-SER)) em (2a); mecanismo mantém F1 competitivo em (2b).

### Controle obrigatório
Baseline ingênuo: detector só-RMS (energia, sem ASR). Se a fusão não superar isso, o resultado não se sustenta.

---

## 8. Resumo de uma linha para a defesa

> *"Provamos adaptação de domínio e ganho de fusão em narração brasileira (jogos PT, ASR no circuito), e mostramos que o mecanismo de detecção generaliza sob ground-truth humano (SoccerNet, em inglês, desacoplado do ASR) — declarando explicitamente que cada fonte valida uma afirmação distinta e que o lado PT é consistente entre jogos, não estatisticamente forte."*
