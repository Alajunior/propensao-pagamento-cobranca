# Propensão a pagamento em cobrança

Modelo que ordena os contratos de uma carteira de cobrança pela chance de pagar nos próximos 30 dias, e o **painel em que a operação usa esse modelo todo dia**:
- funil por faixa;
- fila de trabalho por **valor esperado**;
- simulador de capacidade;
- a aba que vigia o modelo em produção: calibração, backtest de produção e escada de candidatos com gate.

**▶ Abrir o painel:** https://Alajunior.github.io/propensao-pagamento-cobranca/ (abre no navegador, sem instalar nada; funciona no celular)
· [direto na Qualidade do Modelo](https://Alajunior.github.io/propensao-pagamento-cobranca/?aba=qualidade)

> **Todos os dados são sintéticos**, gerados por [`dados/gerar_carteira.py`](dados/gerar_carteira.py) (100 mil clientes, seed 42). Nenhum dado real de cliente ou de empresa. O painel reproduz o funcionamento de um painel que construí e mantenho em produção; o código foi escrito do zero.

![Painel: operação](docs/painel.png)

## Como usar (2 minutos)

1. **Fase (atraso), no topo:** filtra tudo por faixa de dias de atraso. A pirâmide Diamante 10% · Ouro 20% · Prata 30% · Bronze 40% é cortada **dentro de cada fase**.
2. **Operação & Acompanhamento:**
   - **Período em análise:** compare dia × dia, semana × semana, mês × mês ou veja um período único. A seleção comanda os 7 indicadores e todos os cards.
   - **Funil por faixa:** contratos → boletos → pagos, com conversão, pagos por 1.000 contratos e lift.
   - **Tendência diária:** boletos empilhados por faixa e linhas de pagos, em quantidade ou R$, pelo caixa do dia ou pela coorte do boleto.
   - **Concentração & simulador:** arraste a capacidade e veja quanto da recuperação esperada a fila captura. A linha verde mostra o que de fato foi pago.
   - **Perfil da faixa** e **Detalhamento de contratos** (a fila nominal, com busca).
3. **Qualidade do Modelo:**
   - backtest de produção por fase (troque entre **retreino mensal** e **congelado**);
   - previu × pagou (calibração);
   - lift por fase;
   - estabilidade semanal e inversões;
   - a escada de candidatos com o gate;
   - AUC e PSI no tempo.

Cada card tem um botão **ⓘ ajuda** com "o que é / como ler / atenção". Três temas (Midnight, Dark e Light); o fundo de partículas reage ao mouse.

![Painel: qualidade do modelo](docs/painel_qualidade.png)

## Problema de negócio

Uma assessoria recebe contratos inadimplentes e trabalha cada um por cerca de 180 dias. Tem mais contratos do que operadores, linhas e disparos.

**Pergunta:** em cada data de referência D, quais contratos da carteira têm mais chance de pagar algo até D + 30 dias, e por onde começar para trazer mais R$?

## O que tem por trás

| Parte | Como funciona | Onde |
|---|---|---|
| **Carteira sintética** | 100 mil clientes e 138 mil contratos, de jan/2025 a dez/2026, com acionamentos, acordos e parcelas. Sujeiras de propósito: UF bagunçada, nulos, duplicatas, data em texto. **Duas mudanças de cenário:** em fev/2026 entram clientes de renda menor (muda a população); em jun/2026 muda a relação entre renda, atraso e pagamento | [`dados/gerar_carteira.py`](dados/gerar_carteira.py) |
| **Base sem olhar o futuro** | As variáveis usam só eventos anteriores a D; o alvo fica em (D, D+30]. Há versão em SQL com teste de equivalência | [`src/base_analitica.py`](src/base_analitica.py), [`sql/base_analitica.sql`](sql/base_analitica.sql) |
| **Escoragem diária como em produção** | Cada dia útil é escorado pelo modelo **vigente naquele dia**: logística treinada nas 3 safras fechadas anteriores à de calibração. É o retreino mensal. Dia sem 30 dias de futuro fica "não maduro", e o alvo fica vazio, nunca zero | [`src/escoragem.py`](src/escoragem.py) |
| **Calibração** | Isotônica **por fase**, ajustada numa safra que o modelo não viu no treino. Medida por Brier e ECE | [`src/calibracao.py`](src/calibracao.py) |
| **Valor esperado** | Chance calibrada × ticket típico da fase (mediana do que os pagadores pagaram ÷ dívida). **A faixa vem da chance; a fila vem do valor** | [`src/valor.py`](src/valor.py) |
| **Backtest de produção** | A faixa publicada em cada dia contra o pagamento real: ordenação perfeita / fraca / INVERTIDA / sem dados, inversões semanais, estabilidade e lift por fase | [`src/qualidade.py`](src/qualidade.py) |
| **Escada com gate** | Régua (sem modelo) → logística → LightGBM. O desafiante só substitui o campeão se ganhar em R$ no top 10% com IC bootstrap acima de zero, sem piorar nenhuma fase e sem inverter faixa | [`src/candidatos.py`](src/candidatos.py) |
| **Backtest fora do tempo e PSI** | Congelado × retreino mensal, safra a safra, com PSI do score e das variáveis | [`src/backtest.py`](src/backtest.py), [`src/avaliar.py`](src/avaliar.py) |
| **Painel** | O Python calcula tudo e grava **um HTML** com os dados embutidos. Front em HTML, CSS e JS puro, com Chart.js. Chave ausente no payload = card escondido, nunca card vazio | [`src/montar_painel.py`](src/montar_painel.py), [`src/painel.html`](src/painel.html) |

## Resultados

### 1. O score ordena, mas o número dele estava errado (e a calibração conserta)

A logística usa `class_weight="balanced"`, que ajuda a ordenar numa base desbalanceada (pagam ~4%), mas infla a probabilidade.

| | Score bruto | Calibrado |
|---|---|---|
| Chance média prevista (a real é 4,15%) | 34,4% | **4,14%** |
| Brier | 0,149 | **0,035** |
| ECE (distância média entre previsto e pago) | 30,2% | **0,18%** |

- **O decil de cima mostra o tamanho do erro:** o score bruto previa 86% e pagaram 23%; o calibrado prevê 24%.
- **A calibração não muda a ordem dentro da fase:** só coloca o número no lugar. Isso importa porque o valor esperado multiplica a chance pelo ticket, e uma chance 8 vezes maior distorce a fila.

### 2. Fila por valor esperado traz mais R$; fila por chance traz mais pagadores

Nos dias maduros, com capacidade para acionar parte do estoque:

| Capacidade | Fila por chance (pagadores · R$ pago) | Fila por valor esperado (pagadores · R$ pago) |
|---|---|---|
| 10% | 56,2% · 38,0% | 38,8% · **58,8%** |
| 20% | 70,6% · 56,5% | 59,1% · **75,2%** |
| 30% | 78,8% · 67,8% | 70,7% · **83,9%** |

- **Com 10% de capacidade, a fila por valor captura 21 pontos a mais do R$ pago** e alcança 17 pontos a menos de pagadores.
- **A escolha é de negócio:** se a meta é R$ recuperado, fila por valor; se é quantidade de acordos, fila por chance.
- **Por isso o painel separa as duas coisas:** a faixa (Diamante..Bronze) continua sendo cortada pela chance, porque é o que a ordenação promete; a fila de trabalho usa o valor.

### 3. Backtest de produção: as faixas publicadas se comportaram

Dias maduros de 01/12/2025 a 30/10/2026, com retreino mensal:

| Fase | AUC | Taxa Diamante | Taxa Bronze | Ordenação |
|---|---|---|---|---|
| 0001-0360 | 0,766 | 35,1% | 3,21% | perfeita |
| 0361-0720 | 0,800 | 24,8% | 1,40% | perfeita |
| 0721-1080 | 0,809 | 13,6% | 0,67% | perfeita |
| 1081+ | 0,817 | 12,5% | 0,50% | perfeita |
| **Todas** | **0,833** | **21,3%** | **1,39%** | **perfeita** |

- **Inversões existem, e o painel as mostra:** 16 semanas com a ordem quebrada, todas nas fases antigas (0721-1080 e 1081+).
- **Em 15 delas, a Prata passou o Ouro:** nessas fases as duas pagam perto de 1%, e uma semana basta para a ordem trocar. É ruído entre faixas quase iguais, não o modelo invertido: o Diamante nunca foi passado.
- **A regra de leitura:** inversão isolada entre faixas vizinhas é ruído; inversão persistente, ou envolvendo o Diamante, pede retreino.
- **"Diamante" não vale o mesmo entre fases:** o lift do Diamante vai de 4,2× a média da fase (atraso curto) a 6,3× (atraso antigo). O card de lift por fase existe para ninguém ler o Diamante como uma coisa só.

### 4. Escada de candidatos: o gate aprovou o desafiante, e ele vai para modo sombra

Safras de teste de dez/2025 a out/2026, cada candidato retreinado todo mês. **Captura** = % do R$ pago que fica nos 10% de maior score.

| Candidato | Captura no top 10% | AUC |
|---|---|---|
| Régua (sem modelo: contato recente, depois menor atraso) | 31,9% | 0,733 |
| Regressão logística | 38,0% | 0,835 |
| LightGBM | **39,5%** | **0,839** |

- **Logística contra régua:** +6,1 pp de captura (IC 95% de +4,5 a +7,6). Passou nos 4 critérios.
- **LightGBM contra logística:** +1,5 pp (IC 95% de +0,8 a +2,2), sem piorar nenhuma fase e sem inverter faixa. **Passou no gate.**
- **Decisão:** o LightGBM vira o novo campeão, mas **entra em modo sombra** (escora em paralelo, sem ir para a operação) antes de substituir a logística, que segue em produção no painel.
  - O ganho é real mas pequeno: 1,5 pp do R$ no topo. Ele precisa se confirmar na escoragem diária, nas faixas publicadas, e não só no backtest mensal.
- **Ressalva honesta:** o gerador sintético usa uma relação logística por trás, o que favorece a logística. Mesmo assim o LightGBM ganha no topo da fila. Em dado real, com interações que ninguém programou, a vantagem do GBM costuma ser maior.
  - **A versão anterior deste repositório, com uma base 5× menor, concluiu o contrário:** o LightGBM não ganhava. Com mais dados, a diferença saiu do ruído. É exatamente para isso que existe o intervalo de confiança no gate.

### 5. A validação não é a produção, e o PSI vê a mudança errada

| | Até mai/2026 | De jun/2026 em diante |
|---|---|---|
| LightGBM congelado (AUC fora do tempo) | 0,821 | 0,835 |
| LightGBM com retreino mensal | 0,823 | 0,847 |

- **Depois da mudança de relação, o congelado fica de 0,5 a 1,6 ponto abaixo do retreinado.** A AUC de validação dele continua dizendo 0,815 o tempo todo.
- **O PSI da renda passa de 0,25 em jun/2026 e chega a 0,56:** é a mudança de **população** de fev/2026 se acumulando na carteira. Mesmo assim o modelo continuou ordenando bem, então esse alarme foi falso para o desempenho.
- **A mudança de relação, a que importa, não aparece em PSI nenhum:** o do score fica abaixo de 0,05 e o dos dias de atraso perto de zero.
- **Lição:** PSI alto manda investigar; quem manda retreinar é a queda de desempenho fora do tempo, que chega com 30 dias de atraso. O retreino mensal com janela curta é a defesa.

### 6. Três modelos ordenam parecido

Média das safras antes de jun/2026, com retreino mensal ([`relatorios/comparacao_modelos.csv`](relatorios/comparacao_modelos.csv)):

| Modelo | AUC | KS | PR-AUC | Captura de pagadores no top 20% | Treino (s) |
|---|---|---|---|---|---|
| LightGBM | 0,823 | 0,520 | **0,264** | 69,1% | 0,47 |
| Regressão logística | 0,822 | **0,524** | 0,240 | **69,5%** | **0,19** |
| MLP (rede neural) | 0,819 | 0,519 | 0,249 | 68,7% | 2,89 |

- **A rede neural não entra:** não ganha em nada e custa 15 vezes mais para treinar que a logística.
- **O que separa o LightGBM é o topo da fila:** a maior PR-AUC. É o que o gate mede em R$.

## Monitoramento e quando retreinar

| Sinal | Onde aparece | Ação |
|---|---|---|
| PSI de variável ou do score > 0,25 | AUC e PSI no tempo | **Investigar** o que mudou na carteira (política, mix, canal) |
| Faixa INVERTIDA persistente na mesma fase, ou envolvendo o Diamante | Backtest de produção e inversões | **Retreinar** e revisar as variáveis da fase |
| AUC fora do tempo com queda > 3 pontos por 2 safras | AUC no tempo | **Retreinar** com a janela mais recente |
| ECE subindo (previsto se afastando do pago) | Previu × pagou | **Recalibrar** na safra fechada mais recente |
| Desafiante passa no gate | Escada de candidatos | **Modo sombra**, depois troca |

## Como rodar

```bash
pip install -r requirements.txt
python dados/gerar_carteira.py      # carteira sintética de 100 mil clientes (~12 s)
python -m src.treinar               # backtest, comparação, escada com gate e relatorios/ (~1,5 min)
python -m src.montar_painel         # escoragem diária, agregados e docs/index.html (~2,5 min; cache em dados/escoragem/)
python -m pytest                    # 62 testes
```

Para ver o painel, sirva a pasta `docs/` (por exemplo `python -m http.server --directory docs`) ou abra `docs/index.html`. A cada push na `main`, a CI roda o pipeline do zero numa base pequena, roda os testes e publica a pasta `docs/` no GitHub Pages.

Opcional: com `pip install mlflow`, o `src.treinar` registra cada modelo e modo no MLflow.

## Estrutura

```
dados/gerar_carteira.py      gerador sintético (24 meses, duas mudanças de cenário)
sql/base_analitica.sql       a base em SQL (estilo PostgreSQL, roda em SQLite)
src/base_analitica.py        base as-of por safra (nada do futuro)
src/modelos.py               logística, LightGBM e MLP no mesmo Pipeline
src/backtest.py              congelado × retreino mensal fora do tempo, PSI
src/escoragem.py             escoragem diária com o modelo vigente; boletos por parcela
src/calibracao.py            isotônica por fase, Brier, ECE, previu × pagou
src/valor.py                 ticket por fase, valor esperado, concentração, fila por valor
src/qualidade.py             backtest de produção: ordenação, inversões, estabilidade, lift
src/candidatos.py            escada régua → logística → LightGBM com gate e IC bootstrap
src/faixas.py                pirâmide Diamante/Ouro/Prata/Bronze dentro da fase
src/montar_painel.py         monta o payload e grava o HTML
src/painel.html              o painel (HTML, CSS e JS puro; Chart.js; fundo em canvas)
src/treinar.py               orquestra o backtest, a comparação e a escada
dags/escoragem_diaria.py     esboço da escoragem diária no Airflow
docs/                        o painel publicado e as imagens deste README
relatorios/                  tabelas e gráficos estáticos do backtest
tests/                       62 testes: grão, futuro, SQL = Python, calibração, faixas, gate, painel, guarda de publicação
```

## Próximos passos

- **Validação mais dura:** separação treino × validação por cliente (hoje o mesmo contrato aparece em safras empilhadas dos dois lados), alvo embaralhado e curva de aprendizado.
- **Explicabilidade:** "medir o sinal" de cada variável antes de modelar, os principais motivos do score por contrato e o perfil típico por faixa.
- **Linha do tempo para a diretoria:** sem modelo × congelado × retreino mensal, mês a mês.
- **Travas de produção:** foto ausente, queda de volume, manifest com hash do modelo e publicação atômica na escoragem diária.
