# Detecção de Highlights em Narrações de Futebol via Fusão ASR + SER

**Documentação técnica e protocolo experimental**
Processamento de Áudio e Voz — Atividade Final

---

## 1. Visão geral

O sistema converte o áudio de transmissões de futebol em transcrições sincronizadas no tempo (ASR), estima a intensidade emocional/excitação da narração a partir de pistas acústicas (SER), e funde as duas modalidades para localizar **momentos-chave (highlights)** que servem como pontos de corte automáticos.

```
                    ┌─────────────────────────┐
   vídeo/áudio  →   │  0. Pré-processamento    │  → WAV 16 kHz mono
   bruto            └─────────────────────────┘
                                │
              ┌─────────────────┴──────────────────┐
              ▼                                     ▼
   ┌────────────────────┐               ┌──────────────────────┐
   │ 1. ASR (Parakeet)  │               │ 2. SER / arousal      │
   │  texto + timestamps│               │  curva de excitação   │
   └────────────────────┘               └──────────────────────┘
              │ léxico ponderado                    │
              ▼                                     ▼
   ┌────────────────────┐               ┌──────────────────────┐
   │ score lexical s_kw │               │ score acústico s_ser  │
   └────────────────────┘               └──────────────────────┘
              └─────────────────┬──────────────────┘
                                ▼
                    ┌─────────────────────────┐
                    │ 3. Fusão temporal        │  s = α·s_kw + β·s_ser
                    └─────────────────────────┘
                                │
                    ┌─────────────────────────┐
                    │ 4. Detecção de picos     │  → highlights (t_ini, t_fim)
                    │    e segmentação         │
                    └─────────────────────────┘
                                │
                    ┌─────────────────────────┐
                    │ 5. Avaliação             │  WER/CER, P/R/F1 ±tol
                    └─────────────────────────┘
```

### Estratégia de dados (decisão de projeto)

| Papel | Fonte | Idioma | Uso |
|---|---|---|---|
| **Caso principal** | Vídeo bruto com narração real | PT-BR | Demonstra os experimentos ponta a ponta; base para avaliar ASR e fusão |
| **Fallback / prototipagem** | SoccerNet-Echoes | EN | Valida que o pipeline funciona; fornece *ground-truth* de eventos pronto |

Essa separação é deliberada: o **SoccerNet-Echoes é em inglês**, então não serve para avaliar o ASR PT-BR diretamente, mas oferece anotações de eventos (gols, cartões etc.) que funcionam como verdade-fundamental barata para validar a etapa de detecção de highlight e depurar o código antes de aplicar ao vídeo PT.

---

## 2. Modelos e justificativa

### 2.1 ASR — `alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus`

Fine-tune do **Parakeet-TDT-0.6B-v3** (NVIDIA) para português brasileiro. Decisão importante de enquadramento:

> O Parakeet v3 base já é **multilíngue e suporta português oficialmente**. Logo, o experimento **não** demonstra "falha de transferência inglês→português". O eixo correto é **adaptação de domínio**: o modelo entende PT-BR genérico, mas não necessariamente o jargão e a prosódia da narração esportiva ("gooool", "matou no peito", "pintura", nomes próprios).

A NVIDIA documenta que parte das diferenças de desempenho do v3 em português vem de o treino usar **português europeu** enquanto benchmarks usam **brasileiro** — o que justifica a existência de um fine-tune PT-BR e dá fundamento teórico ao seu trabalho.

**Verificação da model card (concluída — listagem de arquivos):**

- [x] **Formato `.nemo`** — checkpoint `parakeet-tdt-0.6b-v3-datasets-ptbr-e-podcasts.nemo` (2.51 GB). **Não há** `model.safetensors`/`config.json`; a inferência é **obrigatoriamente via NVIDIA NeMo** (NeMo + PyTorch + GPU). A alternativa `transformers` está descartada.
- [x] **Variante PT-BR** confirmada (`...ptbr...` no nome).
- [x] **Dados de treino: PT-BR genérico + podcasts** (nome do checkpoint: `...ptbr-e-podcasts`). Ou seja, fala espontânea/conversacional, **sem nenhum dado de narração esportiva** — ver implicação metodológica abaixo. O `hparams.yaml` (195 kB) vale baixar para confirmar arquitetura TDT, tokenizer e lista exata de datasets (proveniência para a metodologia).
- [ ] **Timestamps por palavra** — *ainda a confirmar em runtime.* O TDT (Token-and-Duration Transducer) prediz durações de token, então em tese suporta timestamps de palavra, mas confirme rodando `transcribe(..., timestamps=True)` e checando se `hyp[0].timestamp["word"]` vem preenchido. Se vier só `segment`, os cortes ficam com granularidade de segmento (funciona, mas a janela de corte fica menos precisa).

> **Implicação metodológica (reforça o Experimento 1):** o fine-tune adaptou o modelo ao **idioma (PT-BR) e até à fala espontânea (podcasts)**, mas **não ao domínio futebol**. Logo, o erro residual em jargão/prosódia de narração esportiva é genuinamente *out-of-distribution* do treino — exatamente o gap que a condição C3 do Experimento 1 quer isolar. Não há risco de vazamento de domínio com seus jogos de teste.

### 2.2 SER — detecção de *arousal*, não de emoção categórica

**Decisão metodológica central:** classificadores de emoção discreta (raiva/alegria/tristeza), treinados em bases atuadas (IEMOCAP, RAVDESS), generalizam mal para narração esportiva (gritos, ruído de estádio, prosódia exagerada) — e a categoria emocional não é o que interessa. O que prediz um highlight é o **arousal** (nível de excitação/tensão vocal).

Duas vias, da mais simples à mais robusta:

- **(A) Features acústicas diretas** — energia (RMS), F0 (pitch), taxa de fala, centroide espectral. Interpretável, sem GPU, fácil de defender no relatório. **Recomendada como baseline.**
- **(B) Modelo wav2vec2 de dimensões contínuas** (arousal/valence), p. ex. `audeering/wav2vec2-large-robust-...-msp-dim`. Usar como variante avançada se houver tempo.

Em ambos os casos, descreva o componente como **estimador de excitação acústica**, não como "reconhecimento de emoção" — é mais defensável e alinhado ao problema.

---

## 3. Pré-processamento (etapa 0)

```bash
# Extrair áudio do vídeo bruto e normalizar para 16 kHz mono
ffmpeg -i jogo_bruto.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 jogo.wav
```

Pontos de rigor:

- **16 kHz mono** é o formato esperado pelo Parakeet. Documente a taxa original e a conversão.
- Se a narração estiver misturada com som de estádio, **não** aplique redução de ruído agressiva: o ruído de torcida é sinal útil para o SER. Registre qualquer filtro aplicado.
- Para áudios longos (>24 min), use *chunking* com janelas sobrepostas (o Parakeet v3 suporta áudio longo via *local attention*, mas confirme os limites de memória da sua GPU).

---

## 4. Etapa 1 — ASR e extração de palavras-chave

### 4.1 Transcrição com timestamps (via NeMo)

```python
import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(
    model_name="alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus"
)

# timestamps=True devolve offsets de palavra e segmento
hyp = asr_model.transcribe(["jogo.wav"], timestamps=True)
word_ts = hyp[0].timestamp["word"]      # [{'word','start','end'}, ...]
segment_ts = hyp[0].timestamp["segment"]
```

> **Caminho único: NeMo.** Verificado que o checkpoint é distribuído apenas em `.nemo` (sem variante `transformers`). Setup (a 4090 é **Ada Lovelace, compute 8.9** → exige PyTorch com **CUDA 12.x**; CUDA 11.x pode cair em fallback de CPU ou erro de kernel):
>
> ```bash
> conda create -n highlights python=3.10 -y && conda activate highlights
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
> pip install -U "nemo_toolkit[asr]"
> python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
> # esperado: True NVIDIA GeForce RTX 4090
> ```
>
> **Ambiente de execução: RTX 4090 (24 GB) via SSH.** Folga de sobra para o 0.6B em inferência e para áudio longo com *local attention* — não há necessidade de Colab nem de *chunking* por limite de VRAM (só por conveniência). A instalação do NeMo é pesada e sensível a versões de PyTorch/CUDA, então **valide o setup cedo** (carga do modelo + smoke-test) antes de investir na anotação. Se o setup travar, o plano B (Whisper-PT, com timestamps de palavra nativos e instalação trivial) garante ASR, mas **perde o eixo do Experimento 1** (a comparação base vs. fine-tune PT-BR).

### 4.2 Léxico ponderado de domínio

Não use lista plana — pondere por importância do evento e trate variações:

```python
LEXICO = {
    # evento forte
    "gol": 3.0, "gooo": 3.0, "golaço": 3.0, "golaco": 3.0,
    # eventos médios
    "pênalti": 2.0, "penalti": 2.0, "expuls": 2.0, "vermelho": 2.0,
    "defendeu": 1.6, "defesa": 1.5, "trave": 1.6, "travessão": 1.6,
    # tensão / quase-evento
    "perigo": 1.3, "chance": 1.2, "incrível": 1.0, "que jogada": 1.4,
    "na rede": 2.5, "balançou": 2.0,
}

# Mitigação de falsos positivos (negação / quase-evento)
NEGADORES = ["quase", "perdeu", "isolou", "pra fora", "por cima",
             "anulado", "impedimento", "se fosse"]
```

Regra de pontuação por janela de 1 s, com penalização por negador próximo:

```python
import numpy as np, unicodedata

def normaliza(t):
    t = t.lower()
    return "".join(c for c in unicodedata.normalize("NFD", t)
                   if unicodedata.category(c) != "Mn")

def score_lexical(word_ts, duracao_s, janela=1.0, decai_negador=0.5):
    n = int(np.ceil(duracao_s / janela))
    s = np.zeros(n)
    palavras = [(normaliza(w["word"]), w["start"]) for w in word_ts]
    for i, (w, t) in enumerate(palavras):
        peso = 0.0
        for chave, p in LEXICO.items():
            if normaliza(chave) in w:
                peso = max(peso, p)
        if peso == 0:
            continue
        # checa negador na janela de ±3 palavras
        ctx = " ".join(pw for pw, _ in palavras[max(0, i-3):i+4])
        if any(normaliza(ng) in ctx for ng in NEGADORES):
            peso *= decai_negador
        idx = min(int(t // janela), n - 1)
        s[idx] += peso
    return s   # vetor por janela de 1 s
```

> **Correção em relação ao snippet acima (implementada em `src/asr.py`).** O trecho
> ilustrativo casa cada chave com `normaliza(chave) in w`, ou seja, **substring dentro de
> um único token**. Isso torna as chaves **multi-palavra** (`"na rede"`, `"que jogada"`)
> *inalcançáveis*: nenhum token isolado contém um espaço, então elas nunca pontuam. A
> implementação real separa o léxico em dois conjuntos: chaves de **uma palavra** seguem
> casando por substring; chaves **multi-palavra** casam por **tokens consecutivos** — para
> a chave `("na", "rede")`, compara-se a janela `palavras[i:i+2]` com a tupla da chave.
> Pesos são acumulados por `max`, e o léxico continua normalizado sem acento (logo
> `golaço`→`golaco`, `pênalti`→`penalti`). Mantenha os pesos versionados em
> `configs/params.yaml`.

> **Documente o léxico como hiperparâmetro.** Idealmente derive os pesos de uma pequena amostra anotada, não "no olho", e relate a sensibilidade do resultado a eles.

---

## 5. Etapa 2 — Estimador de excitação acústica (SER)

```python
import librosa, numpy as np

def curva_arousal(wav_path, janela=1.0):
    y, sr = librosa.load(wav_path, sr=16000)
    hop = int(sr * 0.1)                      # quadros de 100 ms
    rms  = librosa.feature.rms(y=y, hop_length=hop)[0]
    f0   = librosa.yin(y, fmin=80, fmax=400, sr=sr, hop_length=hop)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop)[0]

    def z(x):
        x = np.nan_to_num(x)
        return (x - x.mean()) / (x.std() + 1e-8)

    arousal = 0.5*z(rms) + 0.3*z(f0) + 0.2*z(cent)   # pesos = hiperparâmetro

    # reamostra de 100 ms para a grade de 1 s (média)
    fator = int(janela / 0.1)
    n = len(arousal) // fator
    arousal_1s = arousal[:n*fator].reshape(n, fator).mean(axis=1)
    return arousal_1s
```

**Suavização temporal** (média móvel) reduz picos espúrios de ruído isolado:

```python
def suaviza(x, k=3):
    return np.convolve(x, np.ones(k)/k, mode="same")
```

Registre os pesos (0.5/0.3/0.2) e o tamanho de janela como hiperparâmetros sujeitos a ablação.

---

## 6. Etapa 3 — Fusão temporal (experimento principal)

Coloque ambas as curvas na **mesma grade de 1 s**, normalize e combine por **fusão tardia (late fusion)**:

```python
def fusao(s_kw, s_ser, alpha=0.6, beta=0.4):
    def z(x):
        return (x - x.mean()) / (x.std() + 1e-8)
    n = min(len(s_kw), len(s_ser))
    return alpha * z(s_kw[:n]) + beta * z(s_ser[:n])
```

Justificativas a registrar no relatório:

- **Late fusion** é interpretável e permite isolar a contribuição de cada modalidade (necessário para o experimento comparativo).
- **α/β são hiperparâmetros** — faça varredura (ver §9.3).
- **Alinhamento temporal:** as duas curvas compartilham a grade de 1 s ancorada no mesmo `t=0` do áudio.

---

## 7. Etapa 4 — Detecção de picos e segmentação

```python
from scipy.signal import find_peaks

def detecta_highlights(score, janela=1.0, k_sigma=1.5, dist_min_s=8,
                       pre=5.0, pos=10.0):
    limiar = score.mean() + k_sigma * score.std()
    picos, _ = find_peaks(score, height=limiar,
                          distance=int(dist_min_s / janela))
    segmentos = []
    for p in picos:
        t = p * janela
        segmentos.append([max(0, t - pre), t + pos])
    return funde_sobrepostos(segmentos)

def funde_sobrepostos(segs):
    if not segs:
        return []
    segs = sorted(segs)
    out = [segs[0]]
    for ini, fim in segs[1:]:
        if ini <= out[-1][1]:
            out[-1][1] = max(out[-1][1], fim)
        else:
            out.append([ini, fim])
    return out
```

**Assimetria da janela de corte (`pre=5s`, `pos=10s`):** o lance *causa* a reação — o "gooool" vem **depois** do gol. Capturar mais tempo após o pico preserva a jogada que o antecedeu e a comemoração. Documente isso como decisão fundamentada, não arbitrária.

---

## 8. Etapa 5 — Métricas

### 8.1 Qualidade do ASR

- **WER** (Word Error Rate) e **CER** (Character Error Rate) contra transcrição de referência.
- Reporte também um **WER restrito ao léxico de domínio** (taxa de acerto só nas palavras-chave) — é o que de fato impacta a detecção e revela o gap de domínio mesmo quando o WER global parece bom.

```python
from jiwer import wer, cer
WER = wer(referencia, hipotese)
CER = cer(referencia, hipotese)
```

### 8.2 Qualidade da detecção de highlight

Um highlight previsto é **acerto (TP)** se seu pico cair dentro de uma tolerância ±τ de um evento real anotado.

```python
def avalia_deteccao(picos_s, eventos_s, tol=10.0):
    eventos = list(eventos_s)
    tp = 0
    for p in picos_s:
        for e in list(eventos):
            if abs(p - e) <= tol:
                tp += 1
                eventos.remove(e)
                break
    fp = len(picos_s) - tp
    fn = len(eventos)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    f1   = 2*prec*rec / (prec + rec) if (prec + rec) else 0.0
    return dict(precision=prec, recall=rec, f1=f1, tp=tp, fp=fp, fn=fn)
```

Relate **precisão, recall e F1** para várias tolerâncias (τ = 5, 10, 15 s) — a curva F1×τ mostra a robustez temporal do sistema.

---

## 9. Desenho experimental (rigor científico)

### 9.1 Perguntas de pesquisa

- **PP1 (adaptação de domínio):** quanto o fine-tune PT-BR melhora a transcrição da narração esportiva frente ao base multilíngue, e qual o **erro residual atribuível ao domínio** (jargão/prosódia)?
- **PP2 (fusão multimodal):** a fusão ASR+SER melhora a detecção de highlights frente a cada modalidade isolada?

### 9.2 Experimento 1 — Adaptação de idioma vs. de domínio (ASR)

Comparação de três níveis de adaptação sobre **as mesmas narrações PT-BR anotadas**:

| Condição | Modelo | Mede |
|---|---|---|
| C1 | `parakeet-tdt-0.6b-v3` (base multilíngue) | PT genérico |
| C2 | `...ptBR-plus` (fine-tune) | ganho de adaptação ao PT-BR |
| C3 | erro residual em palavras de domínio (subconjunto do léxico) | gap de domínio |

Métricas: WER, CER e WER-no-léxico. **Hipótese:** C2 < C1 em WER global, mas ambos mantêm erro elevado no léxico de domínio (C3), evidenciando que adaptação de idioma ≠ adaptação de domínio.

> **Fundamento confirmado pela proveniência do modelo:** o fine-tune foi treinado em **PT-BR + podcasts** (fala espontânea), sem narração esportiva. Isso ancora a hipótese empiricamente — o ganho de C2 vem de idioma e registro conversacional, mas o jargão/prosódia de futebol permanece *out-of-distribution*, então C3 deve persistir alto.

### 9.3 Experimento 2 — Ablação da fusão (highlight)

Três condições sobre o mesmo conjunto, com o mesmo detector de picos:

| Condição | Score usado |
|---|---|
| **SÓ-ASR** | s_kw apenas (β = 0) |
| **SÓ-SER** | s_ser apenas (α = 0) |
| **FUSÃO** | α·s_kw + β·s_ser |

Métrica: P/R/F1 com tolerância ±10 s (e curva F1×τ). **Hipótese:** F1(FUSÃO) > max(F1(SÓ-ASR), F1(SÓ-SER)).

**Varredura de hiperparâmetros** (relatar em tabela ou heatmap):

- α/β ∈ {(1,0), (0.7,0.3), (0.5,0.5), (0.3,0.7), (0,1)}
- k_sigma do limiar ∈ {1.0, 1.5, 2.0}
- janela de corte (pre, pos)

### 9.4 *Ground-truth* (a parte mais crítica do projeto)

- **PT (vídeo bruto):** anote manualmente os timestamps dos eventos reais (gols, defesas, faltas decisivas, expulsões) e transcreva ~3–5 min para o cálculo de WER. Defina um **protocolo de anotação** (o que conta como highlight) e, se possível, **2 anotadores** com medida de concordância (Cohen's κ). Isso é o que dá legitimidade científica.
- **EN (SoccerNet-Echoes):** use as anotações de evento já fornecidas como GT pronto, para validar o pipeline e gerar uma segunda leva de resultados.

### 9.5 Controles de validade

- **Vazamento:** garanta que os jogos avaliados não estavam no treino do fine-tune.
- **Reprodutibilidade:** fixe `seed`, registre versões (`nemo_toolkit`, `librosa`, `torch`, driver CUDA), e versione o léxico e os hiperparâmetros. Transcrições podem variar sutilmente entre versões de NeMo.
- **Execução longa via SSH:** rode a transcrição do jogo inteiro dentro de `tmux`/`nohup` para a sessão não morrer no meio (`tmux new -s asr` … `Ctrl+B D` … `tmux attach -t asr`).
- **Baseline ingênuo:** compare contra um detector trivial (só energia RMS, sem ASR) — se a fusão não superar isso, o resultado não se sustenta.
- **Honestidade sobre n:** com 1 vídeo PT, os números são **ilustrativos**, não estatísticos. Declare isso explicitamente e use o SoccerNet (mais jogos) para qualquer afirmação quantitativa mais forte.

---

## 10. Estrutura de repositório sugerida

```
projeto/
├── data/
│   ├── pt/                 # vídeo bruto + anotações manuais (GT)
│   └── soccernet/          # subset EN + anotações de evento
├── src/
│   ├── common.py           # utilitários: config, normalização, grade de 1 s, seed
│   ├── preprocess.py       # etapa 0
│   ├── asr.py              # etapa 1 (transcrição + léxico)
│   ├── ser.py              # etapa 2 (arousal)
│   ├── fusion.py           # etapa 3
│   ├── detect.py           # etapa 4
│   ├── evaluate.py         # etapa 5 (WER/CER, P/R/F1)
│   └── pipeline.py         # orquestra 1→4 + métricas (--mode fusion|asr|ser)
├── configs/
│   └── params.yaml         # léxico, pesos, α/β, k_sigma, janelas
├── notebooks/
│   └── experimentos.ipynb  # E1 e E2 com tabelas/figuras
├── smoke_test.py           # checklist 1: valida GPU + carga .nemo + timestamps
├── requirements.txt
└── README.md
```

---

## 11. Checklist de execução

1. [ ] Confirmar model card (timestamps, formato, variante PT-BR, licença)
2. [ ] Extrair e normalizar áudio (ffmpeg → 16 kHz mono)
3. [ ] Rodar pipeline ponta a ponta no **SoccerNet** (fallback) — depurar
4. [ ] Anotar GT do vídeo PT (eventos + transcrição parcial)
5. [ ] Experimento 1 (WER/CER: base vs fine-tune vs léxico)
6. [ ] Experimento 2 (ablação SÓ-ASR / SÓ-SER / FUSÃO + varredura α/β)
7. [ ] Gerar tabelas, curva F1×τ e heatmap de hiperparâmetros
8. [ ] Discutir limitações (n pequeno, viés do léxico, ruído)

---

## 12. Riscos e mitigações

| Risco | Mitigação |
|---|---|
| Checkpoint confirmado só em `.nemo`, setup NeMo pesado/sensível a CUDA | Validar carga do modelo cedo em Colab/GPU (smoke-test); Whisper-PT como plano B para o ASR |
| Timestamps de palavra não confirmados (TDT) | Checar `hyp[0].timestamp["word"]` no smoke-test; fallback para granularidade de segmento |
| GT em PT custoso / subjetivo | Protocolo de anotação claro; 2 anotadores + κ; começar com 1 jogo |
| SER confunde ruído de torcida com excitação | Suavização; comparar com baseline só-RMS; documentar |
| n=1 vídeo PT enfraquece estatística | Resultados PT como ilustração; quantitativo forte no SoccerNet |
| Falsos positivos do léxico ("quase gol") | Regra de negadores; reportar antes/depois da mitigação |

---

## 13. Runbook de execução na 4090 (via SSH)

Sequência completa de comandos, de máquina recém-acessada até resultados. **Tudo o que
toca o modelo (download de 2,5 GB, transcrição do jogo inteiro) deve rodar dentro de
`tmux`** para sobreviver a quedas de conexão SSH.

> **Estado atual da validação.** Sintaxe de todos os módulos, validade do YAML e a *lógica
> pura* (score lexical com negadores → fusão → detecção → métricas P/R/F1) já foram
> testadas em ambiente sem GPU. **Falta validar na 4090** o que depende de NeMo/librosa/GPU:
> `asr.py`, `ser.py` e o pipeline real. O portão de entrada é o `smoke_test.py` — se ele
> passar, o resto do pipeline roda.

### 13.0 Sessão persistente (faça isto primeiro)

```bash
ssh usuario@maquina-4090
tmux new -s highlights          # reconectar depois: tmux attach -t highlights
# (Ctrl+B depois D para destacar sem matar a sessão)
```

### 13.1 Setup do ambiente (uma vez)

A 4090 é **Ada Lovelace (compute 8.9)** → o PyTorch precisa ser do índice **CUDA 12.x
(cu121)**; cu11x cai em fallback de CPU ou erro de kernel. Instale o torch **antes** do
`requirements.txt` para fixar o índice certo.

```bash
git clone <repo> highlights && cd highlights      # ou cd para o projeto já clonado
conda create -n highlights python=3.10 -y && conda activate highlights
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# Verifique a GPU antes de seguir:
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# esperado: True NVIDIA GeForce RTX 4090
```

### 13.2 Smoke-test (portão obrigatório — checklist §11.1)

Valida GPU + carga do `.nemo` + presença de timestamps de **palavra**. O download do
checkpoint (~2,5 GB) acontece aqui, na primeira execução.

```bash
python smoke_test.py                       # tom sintético: só exercita setup + timestamps
python smoke_test.py --audio amostra.wav   # opcional: com áudio real curto
```

Critério de aprovação: imprime `GPU: OK | ASR+timestamps: OK` (exit 0). Se aparecer
`sem 'word'`, os cortes caem para granularidade de **segmento** (plano B da §4.1 — o
pipeline ainda funciona, com janela de corte menos precisa).

### 13.3 Pré-processamento (etapa 0)

Só é necessário se você parte de vídeo; se já tem um WAV 16 kHz mono, pule.

```bash
# via módulo (lê sample_rate do config):
python -m src.preprocess --input jogo_bruto.mp4 --output data/jogo.wav
# equivalente em ffmpeg puro:
ffmpeg -i jogo_bruto.mp4 -vn -ac 1 -ar 16000 -sample_fmt s16 data/jogo.wav
```

### 13.4 Pipeline ponta a ponta

**Depure no SoccerNet (EN, GT pronto) antes do vídeo PT.** O `--events` é opcional: com
ele, o pipeline já imprime P/R/F1 por tolerância.

```bash
# SoccerNet (fallback / depuração)
python -m src.pipeline --audio data/soccernet/jogo.wav \
    --events data/soccernet/eventos.json --out out/soccernet/

# Vídeo PT (caso principal) — pré-processa o vídeo automaticamente com --video
python -m src.pipeline --video data/pt/jogo_bruto.mp4 \
    --events data/pt/eventos.json --out out/pt/
```

Transcrição do jogo inteiro é a parte longa — mantenha-a no `tmux`. Saídas:
`out/.../asr.json` (texto + word_ts + s_kw) e `out/.../highlights.json` (score, picos,
segmentos, métricas).

### 13.5 Estágios isolados (debug)

```bash
python -m src.asr      --audio data/jogo.wav --out out/asr.json
python -m src.ser      --audio data/jogo.wav --out out/ser.json
```

### 13.6 Experimento 2 — ablação da fusão (§9.3)

Mesmo áudio e mesmo detector; só muda `--mode`:

```bash
python -m src.pipeline --audio data/jogo.wav --events data/eventos.json \
    --mode asr    --out out/abla_asr/      # β=0  → só léxico
python -m src.pipeline --audio data/jogo.wav --events data/eventos.json \
    --mode ser    --out out/abla_ser/      # α=0  → só arousal
python -m src.pipeline --audio data/jogo.wav --events data/eventos.json \
    --mode fusion --out out/abla_fusion/   # α,β do params.yaml
```

Para a **varredura α/β · k_sigma**, edite `configs/params.yaml` (ou copie para
`configs/params_<exp>.yaml` e passe `--config`) e re-rode `--mode fusion`. Versione cada
config — é variável de ablação.

### 13.7 Métricas avulsas

```bash
# Detecção: highlights previstos vs. eventos GT (curva F1×τ)
python -m src.evaluate --pred out/pt/highlights.json --events data/pt/eventos.json

# ASR (Experimento 1): WER, CER e recall do léxico de domínio
python -m src.evaluate --ref data/pt/ref.txt --hyp out/pt/hyp.txt
```

> **Experimento 1 (base vs. fine-tune).** Para a condição C1, troque `asr.model_name` em
> `configs/params.yaml` para o checkpoint base (`parakeet-tdt-0.6b-v3`), re-rode a
> transcrição e compare WER/CER/recall-de-léxico contra o fine-tune `...ptBR-plus` (C2)
> sobre o **mesmo** áudio PT anotado.

### 13.8 Formato dos arquivos de GT

`_load_times` (em `evaluate.py`) aceita o JSON de eventos em qualquer destes formatos:

```jsonc
[12.0, 45.5, 88.0]                          // lista de segundos
{"events_s": [12.0, 45.5]}                  // ou "peaks_s" / "events"
[{"t": 12.0}, {"time": 45.5}, {"start": 88.0}]  // lista de objetos
```

A referência de ASR (`--ref`) e a hipótese (`--hyp`) são `.txt` simples.
