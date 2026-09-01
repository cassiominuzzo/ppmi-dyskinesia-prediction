---
title: Calculadora Discinesia PPMI (Total-6)
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
license: mit
---

# Calculadora de discinesia problemática induzida por levodopa — modelo Total-6

Estima a probabilidade individual de **discinesia problemática** (MDS-UPDRS 4.1≥2 OU 4.2≥2) em **3, 5, 7 e 10 anos** após o início da levodopa, a partir de **6 variáveis clínicas** obtidas em uma única consulta:

1. **MDS-UPDRS total** (Partes I+II+III)
2. **Idade de início** da DP
3. **Sexo**
4. **IMC**
5. **Razão TD/PIGD** (subtipo motor)
6. **Congelamento da marcha** (item 2.13)

**Desempenho (coorte PPMI, n = 813, 165 eventos):** C-index aparente 0,712 · corrigido por otimismo **0,701** (IC 95% 0,656 a 0,750) · fora da amostra 0,696 · observado/esperado 1,02 em 3, 5 e 7 anos · benefício líquido positivo de 6% a 50% de limiar (curva de decisão interna).

Deixa-um-sítio-de-fora, nos 23 dos 50 sítios com pelo menos dez participantes e dois eventos: C mediano 0,727 (IIQ 0,576 a 0,790) e média ponderada por eventos 0,666. A amplitude reflete uma mediana de cinco eventos por sítio, não heterogeneidade real de desempenho.

**Validação temporal:** treinando na onda de recrutamento de 2010 a 2019 e testando na de 2020 a 2025 (421 participantes, 20 eventos, 15 sítios ausentes da onda anterior), o C foi 0,740 (IC 95% 0,642 a 0,825) com inclinação de calibração 1,004. O risco absoluto foi superestimado em cerca de um quarto nessa onda de menor risco.

Modelo de Cox de riscos proporcionais. Predição por fórmula fechada (numpy), idêntica ao lifelines. Genética, neuroimagem e biomarcadores foram testados em todo o banco (89 variáveis, 10 domínios) e **não acrescentaram discriminação**; para a imagem esse nulo é bem-poderado, com 794 dos 813 participantes.

> ⚠️ **Ferramenta de pesquisa.** Não substitui julgamento clínico. O estudo é de desenvolvimento com validação interna e temporal: **não há validação externa**, e o modelo não foi testado fora do PPMI. O risco absoluto pode exigir recalibração local. Dados: PPMI, cut 29-Abr-2026. Relato segundo TRIPOD+AI.

## Rodar localmente
```bash
pip install -r requirements.txt
streamlit run app.py
```
