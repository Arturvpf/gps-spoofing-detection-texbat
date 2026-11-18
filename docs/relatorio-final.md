# Relatório Técnico Final — Detecção de Spoofing em Sinais GPS Civis no Dataset TEXBAT

**Disciplina:** Sinais e Sistemas (ES413) — Centro de Informática (CIn), Universidade Federal de Pernambuco (UFPE)  
**Equipe:**
- Artur Vinicius Pereira Fernandes
- Isabela Maria de Moura Nascimento
- Karina Lima de Oliveira

---

## 1. Introdução

### 1.1 Funcionamento da Recepção GPS Civil (L1 C/A)
O Sistema de Posicionamento Global (GPS) baseia-se na medição do tempo de propagação de sinais de radiofrequência transmitidos por uma constelação de satélites em órbita média (MEO). Na banda civil **L1** ($f_{L1} = 1575{,}42\text{ MHz}$), cada satélite utiliza acesso múltiplo por divisão de código (**CDMA**) modulando em fase binária (**BPSK**) uma sequência pseudoaleatória exclusiva denominada código **C/A (*Coarse/Acquisition*)**, pertencente à família de códigos de Gold de comprimento $L = 1023\text{ chips}$ e taxa $f_{\text{chip}} = 1{,}023\text{ Mchips/s}$ (período exato de $T_{\text{coh}} = 1\text{ ms}$).

Devido à atenuação de espaço livre ao longo de $\approx 20.200\text{ km}$, o sinal GPS atinge a antena receptora com densidade espectral abaixo do ruído térmico de fundo ($\text{SNR} \approx -19\text{ dB}$ na largura de banda de recepção). A detecção só se torna possível graças ao **ganho de processamento por correlação cruzada** ($\approx 43\text{--}44\text{ dB}$ em $1\text{ ms}$ a $25\text{ MSPS}$), concentrando a energia do código PRN alinhado em um pico triangular estreito de largura $\pm 1\text{ chip}$.

### 1.2 Ameaça de Spoofing e Taxonomia de Ataques no TEXBAT
Como a estrutura dos códigos C/A civis e das mensagens de navegação é publicamente documentada (IS-GPS-200) e desprovida de autenticação criptográfica na portadora L1 C/A legada, um atacante equipado com um rádio definido por software (SDR) pode sintetizar e irradiar réplicas falsas coerentes. Na bateria padrão **TEXBAT (*Texas Spoofing Test Battery*)**, desenvolvida pelo *Radionavigation Laboratory* da Universidade do Texas em Austin (Humphreys et al., 2012), os ataques dividem-se em três mecanismos fundamentais:

1. **Switch-Over (`DS1`):** substituição abrupta dos sinais autênticos pelas portadoras do *spoofer*, provocando descontinuidades instantâneas de fase, código e amplitude nos correladores.
2. **Overpowered Time/Position Push (`DS2`, `DS5`):** o *spoofer* injeta réplicas alinhadas com vantagem elevada de potência ($\approx +10\text{ dB}$), captura as malhas de código/portadora (DLL/PLL) pela força bruta e desloca gradualmente a solução de tempo ou posição, elevando o piso de ruído e a relação $C/N_0$.
3. **Matched-Power Time/Position Push (`DS3`, `DS4`, `DS6`):** ataque furtivo com vantagem mínima de potência ($+0{,}4\text{ dB}$ a $+1{,}3\text{ dB}$) e fase controlada. Durante o *pull-off* (arraste das malhas), a superposição construtiva/destrutiva entre o triângulo de correlação autêntico e o falsificado deforma a simetria e a largura do pico sem gerar saltos grosseiros de potência total.

---

## 2. Metodologia

A solução foi estruturada como um pacote Python modular (`src/texbat_spoofing/`) articulado em dois estágios complementares, reproduzidos integralmente no notebook `notebooks/texbat_analysis.ipynb` e via linha de comando (`scripts/run_pipeline.py`).

### 2.1 Estágio 1 — Processamento de Banda-Base Bruta ($25\text{ MSPS}$) e Forma do Pico
1. **Ingestão I/Q com Mapeamento em Memória (`load_iq_from_bin`):** leitura sob demanda (`np.memmap`) de pares intercalados `int16` convertidos para `complex64` ($I + jQ$), remoção de nível DC e suporte explícito a deslocamento temporal `start_time_s` para inspecionar o regime de ataque ativo ($t > 125\text{ s}$).
2. **Análise Espectral (`compute_psd_and_flatness`):** estimativa da Densidade Espectral de Potência (PSD) bilateral pelo método de Welch, cálculo da potência média de banda larga $10\log_{10}(\mathbb{E}[|x[n]|^2])$ e da **Planicidade Espectral de Wiener (*Spectral Flatness*)**:
   $$\text{SFM} = \frac{\exp\!\left(\frac{1}{K}\sum_{k=0}^{K-1}\ln P_{xx}[k]\right)}{\frac{1}{K}\sum_{k=0}^{K-1} P_{xx}[k]}$$
3. **Aquisição Paralela 2D via FFT (`acquire_prn_pcps`):** geração exata dos códigos Gold C/A via registradores LFSR `G1` e `G2` com amostragem retangular por retenção de ordem zero (*ZOH*), seguida de *wipe-off* da portadora Doppler e correlação circular por FFT em malha fina de $100\text{ Hz}$, acumulação não-coerente e calibração estatística de $C/N_0$ ($\text{dB-Hz}$).
4. **Métricas SQM de Forma de Onda (`extract_waveform_sqm_features`):**
   - **RCP (*Ratio of the Correlation Peak*):** razão entre o pico principal e o maior pico secundário fora da vizinhança circular de $\pm 2{,}0\text{ chips}$.
   - **Largura de Meio Pico (`Largura HPW` em chips):** largura interpolada linearmente em sub-amostra a $50\%$ do excesso de amplitude acima do piso de ruído (referência ideal $= 1{,}000\text{ chip}$).
   - **Assimetria (`Skew`):** $(w_{\text{lead}} - w_{\text{trail}})/(w_{\text{lead}} + w_{\text{trail}})$.
   - **Razão e Diferença Simétrica Early–Prompt–Late:** $M_{\text{ratio}} = \frac{R_E + R_L}{2R_P}$ e $\Delta_{\text{SQM}} = \frac{R_E - R_L}{2R_P}$.

### 2.2 Estágio 2 — Monitoramento Multicanal Contínuo nos 8 Cenários do TEXBAT
Para treinar e validar classificadores estatisticamente robustos ao longo dos $\approx 420\text{--}480\text{ s}$ completos de cada cenário (`cleanStatic`, `cleanDynamic`, `ds1`–`ds6`), extraímos janelas não-sobrepostas de **$1\text{ s}$** consolidando todos os canais de satélites rastreados simultaneamente ($\ge 4$ satélites visíveis por época), totalizando **3.303 épocas rotuladas** e **15 atributos**:
- **Potência e Relação Portadora-Ruído:** `cn0_mean_dbhz`, `cn0_std_dbhz` (variação entre canais exigida na proposta), `cn0_temporal_jitter_dbhz`, `corr_peak_norm_mean` (normalizado pelo tempo de integração coerente) e `corr_peak_cv`.
- **Integridade do Pico de Correlação (SQM):** `sqm_ratio_mean`, `sqm_ratio_max`, `sqm_ratio_std`, `sqm_delta_abs_mean` e `sqm_delta_std`.
- **Estabilidade de Portadora, quadratura e Código (PLL/DLL):** `iq_quadrature_ratio` ($|Q_P|/|I_P|$), `doppler_residual_hz`, `phase_slip_rate`, `pll_lock_mean` e `dll_code_error_rms_chips`.

### 2.3 Modelos de Classificação e Protocolos de Validação
Avaliamos os três algoritmos propostos na especificação da disciplina — **SVM** (kernels Linear e RBF), **Random Forest** e **MLP (*Multi-Layer Perceptron*)** — encapsulados em `sklearn.pipeline.Pipeline` com `StandardScaler` ajustado estritamente nos dados de treino e busca de hiperparâmetros por `GridSearchCV` (5 folds). Adotamos três protocolos de complexidade crescente:
1. **Stratified Hold-Out (70% treino / 30% teste) + 5-Fold CV.**
2. **Divisão por Blocos Temporais Disjuntos (Gap de Guarda $= 10\text{ s}$):** os primeiros 70% cronológicos de cada regime contínuo compõem o treino, $10\text{ s}$ intermediários são descartados e os 30% finais compõem o teste, eliminando qualquer vazamento por autocorrelação entre janelas vizinhas.
3. **Leave-One-Attack-Out (LOAO) em `DS1`–`DS6`:** em cada rodada, um cenário inteiro de ataque é omitido do treinamento para medir a capacidade de detecção *zero-shot* de modalidades inéditas de spoofing.

---

## 3. Processo de Desenvolvimento, Tentativas, Erros e Resultados

### 3.1 Análise Crítica das Tentativas Preliminares e Bugs Corrigidos
Durante o desenvolvimento iterativo do projeto, cinco problemas técnicos significativos surgiram nos protótipos iniciais do notebook e foram diagnosticados e resolvidos:

1. **Problema do Dataset Inicial de 4 Amostras ($X \in \mathbb{R}^{4 \times 5}$):** na primeira versão do classificador, apenas 1 segmento de $1\text{ ms}$ era extraído por arquivo (`CleanStatic`, `CleanDynamic`, `DS1`, `DS2`), gerando 2 amostras de treino e 2 de teste. Nessa condição degenerada, a Random Forest obteve $0\%$ de acurácia e a SVM linear obteve $100\%$ por puro acaso geométrico em $\mathbb{R}^5$, sem validade estatística.
2. **Erro de Janela Pré-Ataque ($t = 0\text{ s}$) nos Binários do TEXBAT:** a leitura inicial de `num_samples = 5_000_000` a partir de `start_sample = 0` cobria apenas os primeiros $0{,}2\text{ s}$ da gravação. Contudo, no protocolo TEXBAT, os ataques só iniciam entre $t = 105\text{ s}$ e $t = 125\text{ s}$! Rotular $t \in [0, 0{,}2\text{ s}]$ de `ds1.bin` e `ds2.bin` como *Spoofed* treinava o modelo sobre ruído de fundo pré-ataque. Corrigimos tanto o deslocamento temporal (`start_time_s >= 130 s`) quanto a rotulagem temporal exata por regime (`pre_attack_clean` vs. `active_spoofing`, com banda de guarda de $\pm 8\text{ s}$ na transição).
3. **Degradação por `scipy.signal.resample` em Janelas $> 1\text{ ms}$:** ao tentar ampliar segmentos para $10\text{ ms}$ (`250.000` amostras), o uso de `resample(ca_code_chips, N_SAMPLES_SEGMENT)` interpolava um único período de $1023\text{ chips}$ por $10\text{ ms}$ (reduzindo a taxa de código em $10\times$!) e introduzia oscilações de Gibbs nas transições BPSK, derrubando a relação sinal-ruído de $11{,}88$ para $4{,}43$ e inflando artificialmente a largura HPW para $4{,}05\text{ chips}$. A substituição por amostragem retangular periódica $\lfloor n \cdot f_{\text{chip}} / f_s \rfloor \bmod 1023$ restaurou a periodicidade física exata de $1\text{ ms}$.
4. **Travamento do `GridSearchCV` da SVM sem Escalonamento:** nas tentativas de busca em grade com `C = 100` e `kernel = 'linear'` sobre variáveis brutas (`Doppler` na casa de $\pm 4000\text{ Hz}$ e `Assimetria` na casa de $0{,}01$), o número de condicionamento extremo ($\approx 4 \times 10^5$) fazia o otimizador SMO da LIBSVM não convergir, resultando em `KeyboardInterrupt`. A introdução de `StandardScaler` dentro do `Pipeline` reduziu o tempo de ajuste completo para menos de $1\text{ s}$.
5. **Quantização Inteira na Largura HPW e Assimetria:** a estimativa sem interpolação sub-amostra produzia `Assimetria = 0.000` idêntica para cenários limpos e atacados. A interpolação linear fracionária nas bordas ascendente e descendente revelou claramente o enviesamento negativo/positivo sob *time-push*.

### 3.2 Resultados Quantitativos nos 8 Cenários Reais do TEXBAT

#### Protocolo A — Divisão Estratificada (70/30) + Validação Cruzada 5-Fold ($N_{\text{teste}} = 991$)

| Modelo | Melhores Hiperparâmetros | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | `n_estimators=100, max_depth=8, min_samples_leaf=1` | **99,90%** | **99,83%** | **100,00%** | **0,9991** | **1,0000** |
| **SVM** | `kernel='linear', C=5.0` | 99,80% | 99,66% | 100,00% | 0,9983 | 0,9997 |
| **MLP (Neural Net)** | `hidden=(64, 32), alpha=1e-4` | 99,39% | 99,49% | 99,49% | 0,9949 | 0,9998 |

#### Protocolo B — Divisão Cronológica em Blocos Temporais Sem Vazamento (Gap $= 10\text{ s}$, $N_{\text{teste}} = 952$)

| Modelo | Matriz de Confusão `[[TN, FP], [FN, TP]]` | Acurácia | Precisão | Recall | F1-Score | ROC-AUC |
|---|---|---:|---:|---:|---:|---:|
| **Random Forest** | `[[385, 0], [0, 567]]` | **100,00%** | **100,00%** | **100,00%** | **1,0000** | **1,0000** |
| **SVM** | `[[385, 0], [2, 565]]` | 99,79% | 100,00% | 99,65% | 0,9982 | 1,0000 |
| **MLP (Neural Net)** | `[[384, 1], [10, 557]]` | 98,84% | 99,82% | 98,24% | 0,9902 | 0,9994 |

A análise de importância por permutação confirmou que as métricas derivadas diretamente da geometria de correlação — **`sqm_ratio_mean`**, **`sqm_ratio_max`**, **`corr_peak_norm_mean`** e **`cn0_mean_dbhz`** — respondem pela maior contribuição preditiva, validando a hipótese central de monitoramento de integridade do sinal.

#### Protocolo C — Generalização *Leave-One-Attack-Out* (LOAO) em Ataques Inéditos

| Cenário Retido (Inédito no Treino) | Tipo de Ataque | Recall SVM (RBF) | Recall Random Forest | Recall MLP | Taxa de Falso Alarme (`Clean`) |
|---|---|---:|---:|---:|---:|
| `ds1` | *Static Switch-Over* ($0\text{ dB}$) | 100,00% | 100,00% | 100,00% | 0,00% |
| `ds2` | *Static Overpowered Time-Push* ($+10\text{ dB}$) | 100,00% | 100,00% | 100,00% | 0,00% |
| `ds3` | *Static Matched-Power Time-Push* ($+1{,}3\text{ dB}$) | 99,70% | 98,18% | 100,00% | 0,00% |
| `ds4` | *Static Matched-Power Pos-Push* ($+0{,}4\text{ dB}$) | **86,90%** | **83,63%** | **81,25%** | 0,00% |
| `ds5` | *Dynamic Overpowered Time-Push* ($+9{,}9\text{ dB}$) | 99,02% | 98,37% | 97,06% | 0,00% |
| `ds6` | *Dynamic Matched-Power Pos-Push* ($+0{,}8\text{ dB}$) | 93,46% | 99,67% | 95,10% | 0,00% |
| **Média Global LOAO** | — | **96,51%** | **96,64%** | **95,57%** | **0,00%** |

### 3.3 Discussão sobre Falhas e Limitações Físicas do Modelo
1. **Janela de Alinhamento Inicial em Ataques *Matched-Power* Ultra-Furtivos (`DS4`):** conforme revela a tabela LOAO acima, o único cenário com queda mensurável de *recall* em regime *zero-shot* ($\approx 84\text{--}87\%$, embora com $\text{ROC-AUC} = 0{,}9998$) é o `DS4` (*Matched-Power Position-Push* com apenas $+0{,}4\text{ dB}$ de vantagem de potência). Entre $t = 115\text{ s}$ e $t \approx 170\text{ s}$, o *spoofer* mantém sua réplica quase perfeitamente sincronizada em fase de código ($\Delta\tau \approx 0$) antes de iniciar o descolamento espacial. Enquanto $\Delta\tau \approx 0$ e $\Delta P \approx 0{,}4\text{ dB}$, o triângulo resultante é fisicamente indistinguível do sinal autêntico por observáveis puramente baseados na envoltória de correlação; assim que o arraste ultrapassa $\approx 0{,}15\text{ chip}$ ($t > 172\text{ s}$), `sqm_ratio` sobe de $0{,}76$ para $> 0{,}86$ e a detecção atinge $100\%$.
2. **Multipercurso Severo em Ambientes Urbanos Densos:** em `CleanDynamic`, reflexões de multipercurso elevam moderadamente `sqm_delta_abs_mean` ($0{,}051$ vs. $0{,}028$ no estático) e reduzem `RCP` ($2{,}26\text{--}3{,}49$). Embora a agregação multicanal (8 satélites simultâneos) tenha zerado os falsos alarmes no TEXBAT (pois multipercurso afeta satélites isolados de baixa elevação, enquanto um *spoofer* de antena única distorce todos os canais coerentemente), cânions urbanos extremos com poucos satélites em visada direta podem exigir filtragem temporal sequencial adicional.

---

## 4. Conexão com a Disciplina de Sinais e Sistemas (ES413)

Todos os blocos do pipeline traduzem diretamente teoremas centrais de **Sinais e Sistemas**:

1. **Propriedade de Modulação / Deslocamento em Frequência da Transformada de Fourier:** a compensação do efeito Doppler na aquisição multiplica o sinal discreto em banda-base por um fasor complexo $e^{-j2\pi f_d n / f_s}$, deslocando o espectro $X(e^{j\omega})$ de volta à origem antes da filtragem casada com o código PRN.
2. **Correlação Cruzada e Teorema da Convolução Circular via DFT/FFT:** calcular a correlação direta no tempo para $N = 25.000$ amostras exigiria $\mathcal{O}(N^2) \approx 6{,}25 \times 10^8$ operações por bin Doppler. Pelo teorema da correlação na DFT, $R_{xc}[m] = \text{IDFT}\{\text{DFT}\{x[n]\} \cdot \text{DFT}\{c[n]\}^*\}$, reduzimos o custo para $\mathcal{O}(N \log_2 N)$, viabilizando a busca 2D nos 32 satélites.
3. **Amostragem, Reconstrução ZOH vs. Interpolação Sinc:** demonstramos na prática como a interpolação ideal limitada em faixa (`scipy.signal.resample`, equivalente a convolução com kernel Dirichlet/sinc) provoca o **fenômeno de Gibbs** em sinais retangulares descontínuos (chips BPSK $\pm 1$), ao passo que o reconstrutor de ordem zero (*ZOH*) preserva exatamente a energia e a periodicidade da forma de onda retangular do transmissor.
4. **Teorema de Wiener–Khinchin e Método de Welch:** a Densidade Espectral de Potência (PSD) equivale à Transformada de Fourier da função de autocorrelação estatística; o periodograma modificado de Welch reduz a variância espectral por janelamento e média de segmentos sobrepostos, permitindo quantificar o piso de ruído térmico e a entropia espectral.
5. **Sistemas Realimentados Lineares na Vizinhança do Ponto de Operação (PLL e DLL):** os erros residuais de fase e código e a métrica Early–Late refletem a resposta dinâmica de malhas de controle de 2ª/3ª ordem submetidas a entradas em degrau (`DS1` *switch-over*) ou em rampa (`DS2`–`DS6` *time/position push*).

---

## 5. Conclusão e Trabalhos Futuros

Este projeto desenvolveu, depurou e validou um sistema completo de detecção de ataques de *spoofing* em sinais GPS civis sobre o dataset **TEXBAT**, unindo processamento clássico de sinais em banda-base e aprendizado de máquina supervisionado. A combinação de métricas SQM de deformação do pico de correlação com estatísticas multicanal atingiu **$100\%$ de acurácia ($F_1 = 1{,}0000$) para Random Forest** e **$99{,}79\%$ ($F_1 = 0{,}9982$) para SVM** em blocos temporais livres de vazamento, além de **$96{,}6\%$ de *recall* médio com $0{,}00\%$ de falso alarme** em ataques completamente inéditos (*Leave-One-Attack-Out*).

**Trabalhos Futuros:**
- Incorporar correladores *Multi-Correlator* densos ($\pm 0{,}05$ a $\pm 1{,}0\text{ chip}$) diretamente em rastreamento contínuo de gravações `.bin` completas via aceleração GPU/vetorizada.
- Explorar modelos sequenciais temporais (HMM, filtros CUSUM/SPRT ou redes recorrentes leves) para reduzir a latência de detecção nos primeiros segundos de ataques *matched-power* (`DS4`).
- Integrar verificação cruzada de consistência Doppler vs. derivada de pseudodistância e resíduos RAIM na camada de navegação PVT.

---

## 6. Referências

1. **Humphreys, T. E., Ledvina, B. M., Psiaki, M. L., O'Hanlon, B. W., & Kintner, P. M.** (2008). *Assessing the Spoofing Threat: Development of a Portable GPS Civilian Spoofer.* Proceedings of ION GNSS 2008, pp. 2314–2325.
2. **Humphreys, T. E., Bhatti, J. A., Shepard, D., & Wesson, K.** (2012). *The Texas Spoofing Test Battery: Toward a Standard for Evaluating GPS Signal Authentication Techniques.* Proceedings of ION GNSS 2012, pp. 3569–3583.
3. **Psiaki, M. L., & Humphreys, T. E.** (2016). *GNSS Spoofing and Detection.* Proceedings of the IEEE, 104(6), 1258–1270.
4. **Phelts, R. E.** (2001). *Multicorrelator Techniques for Robust Mitigation of Threats to GPS Signal Quality.* Ph.D. Thesis, Stanford University.
5. **Shafique, A., Mehmood, A., & Elhadef, M.** (2021). *Detecting Signal Spoofing Attack in UAVs Using Machine Learning Models.* IEEE Access, 9, 93803–93815.
6. **Nature Scientific Reports** (2025). *GNSS Spoofing Detection via Signal Quality Monitoring and Machine Learning.* https://www.nature.com/articles/s41598-025-16853-1
7. **University of Texas Radionavigation Laboratory — TEXBAT Dataset:** https://radionavlab.ae.utexas.edu/texbat/
