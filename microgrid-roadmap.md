# Gestione energetica di una microgrid — Roadmap del progetto

Pipeline "predict-then-optimize": un modello di ML prevede carico e produzione fotovoltaica del giorno dopo, un modello MILP decide ora per ora come usare batteria e rete per minimizzare costo ed emissioni.

Ogni fase termina con qualcosa di pubblicabile: fai un commit (o più) alla fine di ciascuna.

---

## Fase 0 — Definizione del problema

1. Scegli lo scenario: un edificio o piccolo campus con fotovoltaico, una batteria e connessione alla rete.
2. Fissa i parametri dell'impianto (valori di partenza ragionevoli, poi li varierai):
   - potenza di picco del fotovoltaico (es. 100 kWp)
   - capacità della batteria (es. 200 kWh) e potenza massima di carica/scarica (es. 50 kW)
   - rendimenti di carica e scarica (es. 0,95 ciascuno)
   - stato di carica minimo e massimo (es. 10%–90%)
3. Fissa risoluzione e orizzonte: passo orario, pianificazione sulle 24 ore successive.
4. Scrivi in una frase la domanda a cui il progetto risponde, ad esempio: "Quanto si risparmia in costo e CO₂ gestendo la batteria con previsione + ottimizzazione rispetto a una regola semplice?"

**Output:** sezione "Problema" del README.

---

## Fase 1 — Setup del repository

1. Crea il repository su GitHub con licenza (MIT) e `.gitignore` per Python.
2. Crea la struttura:

   ```
   microgrid-ems/
   ├── README.md
   ├── pyproject.toml          # dipendenze e configurazione
   ├── config/
   │   └── default.yaml        # parametri impianto, percorsi, seed
   ├── data/
   │   ├── raw/                # dati scaricati (non versionati)
   │   └── processed/          # dataset pulito (non versionato)
   ├── notebooks/              # solo analisi esplorativa e grafici
   ├── src/microgrid/
   │   ├── data.py             # download e pulizia
   │   ├── features.py         # costruzione delle feature
   │   ├── forecast.py         # modelli di previsione
   │   ├── optimize.py         # modello MILP
   │   ├── simulate.py         # simulazione a orizzonte mobile
   │   └── evaluate.py         # metriche e confronti
   ├── scripts/
   │   └── run_all.py          # esegue l'intero pipeline
   ├── tests/
   └── docs/
       └── formulation.md      # formulazione matematica
   ```

3. Crea un ambiente virtuale e installa: `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `pyomo` (o `pulp`), `highspy`, `matplotlib`, `pyyaml`, `pytest`.
4. Aggiungi `data/` al `.gitignore`: i dati si riscaricano con uno script, non si caricano sul repo.

**Output:** repo vuoto ma ordinato, primo commit.

---

## Fase 2 — Raccolta dei dati

Servono cinque serie orarie sullo stesso periodo (almeno 1 anno, meglio 2) e sulla stessa zona:

| Serie | Fonti possibili |
|---|---|
| Carico elettrico | UCI ML Repository, Open Power System Data |
| Produzione fotovoltaica | PVGIS (simulata per una località), Open Power System Data |
| Meteo (temperatura, irraggiamento, nuvolosità) | Open-Meteo (archivio storico) |
| Prezzo dell'energia | ENTSO-E Transparency Platform (prezzi day-ahead) |
| Intensità di carbonio della rete (gCO₂/kWh) | Electricity Maps, oppure calcolata dal mix di generazione ENTSO-E |

1. Verifica licenza e copertura temporale di ogni fonte prima di sceglierla.
2. Scrivi in `data.py` una funzione di download per ciascuna serie, che salva in `data/raw/`.
3. Se il carico è di scala diversa dal tuo scenario (es. nazionale), riscalalo al picco desiderato e dichiaralo nel README.
4. Annota nel README fonte, periodo e licenza di ogni dataset.

**Output:** `python -m microgrid.data` scarica tutto da zero.

---

## Fase 3 — Pulizia ed esplorazione

1. Porta tutte le serie su un unico indice orario con fuso orario esplicito (attenzione ai cambi d'ora legale).
2. Gestisci i valori mancanti: interpolazione per buchi brevi, esclusione dei giorni con buchi lunghi.
3. Controlla valori anomali (carico negativo, fotovoltaico di notte).
4. Unisci tutto in un solo file `data/processed/dataset.parquet`.
5. In un notebook, esplora:
   - profilo medio giornaliero e settimanale del carico
   - stagionalità di carico e fotovoltaico
   - correlazione carico–temperatura
   - andamento di prezzi e intensità di carbonio nelle 24 ore
6. Dividi i dati **in ordine temporale**: train (es. primi 70%), validation (15%), test (ultimi 15%). Mai mescolare a caso.

**Output:** dataset pulito + notebook `01_exploration.ipynb` con 4–5 grafici.

---

## Fase 4 — Previsione (Machine Learning)

L'obiettivo è prevedere, a una certa ora del giorno D, le 24 ore di carico e di fotovoltaico del giorno D+1.

1. **Baseline:** previsione ingenua (stessa ora di 7 giorni prima per il carico, stessa ora del giorno prima per il fotovoltaico). Ogni modello va confrontato con questa.
2. **Feature:**
   - calendario: ora, giorno della settimana, mese, festivo (codificate anche con seno/coseno)
   - ritardi: valori di 24, 48 e 168 ore prima
   - statistiche mobili: media delle ultime 24 ore e 7 giorni
   - meteo previsto per il giorno D+1
3. **Evita il leakage:** usa solo informazioni disponibili al momento della previsione. Niente ritardi inferiori a 24 ore.
4. **Nota sul meteo:** usare il meteo reale come se fosse previsione è una semplificazione ottimistica. Dichiarala nel README, oppure usa previsioni meteo archiviate se disponibili.
5. **Modello principale:** gradient boosting (LightGBM), uno per il carico e uno per il fotovoltaico.
6. **Modello opzionale:** una rete LSTM o un MLP, solo per confronto.
7. **Validazione:** time series cross-validation (finestre espandibili) per scegliere gli iperparametri.
8. **Metriche** sul test set: MAE, RMSE e MAE normalizzato; riportale in una tabella baseline vs modelli.
9. Grafici: previsto vs reale su una settimana, errore per ora del giorno.
10. Salva i modelli addestrati e le previsioni sul test set.

**Output:** `forecast.py`, tabella delle metriche, notebook `02_forecasting.ipynb`.

---

## Fase 5 — Formulazione del modello di ottimizzazione

Scrivila in `docs/formulation.md` prima di programmarla.

**Insiemi e parametri** (per ogni ora t = 1…24, con Δt = 1 h):

- `L_t` carico previsto, `PV_t` fotovoltaico previsto [kW]
- `c_t` prezzo di acquisto, `s_t` prezzo di vendita [€/kWh]
- `e_t` intensità di carbonio della rete [kgCO₂/kWh]
- `E_max` capacità batteria, `P_max` potenza massima, `η_c`, `η_d` rendimenti
- `SOC_min`, `SOC_max`, `SOC_0` stato di carica iniziale
- `G_max` potenza massima scambiabile con la rete
- `λ` peso delle emissioni [€/kgCO₂]

**Variabili:**

- `p_ch_t`, `p_dis_t` ≥ 0 — potenza di carica e scarica
- `g_imp_t`, `g_exp_t` ≥ 0 — potenza acquistata e venduta
- `curt_t` ≥ 0 — fotovoltaico non utilizzato
- `soc_t` — energia in batteria
- `u_t` ∈ {0,1} — 1 se la batteria carica, 0 se scarica
- `v_t` ∈ {0,1} — 1 se si acquista, 0 se si vende

**Funzione obiettivo:**

```
min  Σ_t [ c_t · g_imp_t − s_t · g_exp_t + λ · e_t · g_imp_t ] · Δt
```

**Vincoli:**

```
Bilancio:      PV_t − curt_t + p_dis_t + g_imp_t = L_t + p_ch_t + g_exp_t
Dinamica SOC:  soc_t = soc_{t−1} + η_c · p_ch_t · Δt − p_dis_t · Δt / η_d
Limiti SOC:    SOC_min ≤ soc_t ≤ SOC_max
Carica:        p_ch_t  ≤ P_max · u_t
Scarica:       p_dis_t ≤ P_max · (1 − u_t)
Acquisto:      g_imp_t ≤ G_max · v_t
Vendita:       g_exp_t ≤ G_max · (1 − v_t)
Fine giornata: soc_24 ≥ SOC_0
Taglio PV:     curt_t ≤ PV_t
```

Note:

1. Senza `u_t` e `v_t` il problema è un LP puro. Le binarie lo rendono un MILP, risolto dal solver con Branch and Cut.
2. Il vincolo di fine giornata impedisce al modello di svuotare la batteria all'ultima ora.
3. Estensioni facoltative: costo di degradazione della batteria (termine proporzionale all'energia ciclata), tariffa sulla potenza di picco, generatore diesel con accensione binaria.

**Output:** `docs/formulation.md` completo.

---

## Fase 6 — Implementazione dell'ottimizzatore

1. Implementa in `optimize.py` una funzione che riceve previsioni, prezzi, intensità di carbonio, parametri e SOC iniziale, e restituisce il piano orario.
2. Usa Pyomo (o PuLP) con solver HiGHS.
3. Prova prima su un singolo giorno con dati inventati di cui conosci la soluzione attesa (es. prezzo basso di notte e alto di sera: deve caricare di notte e scaricare di sera).
4. Controlla lo stato del solver (ottimo, infeasible) e gestisci gli errori.
5. Scrivi i test in `tests/`:
   - il bilancio energetico è rispettato in ogni ora
   - il SOC resta nei limiti
   - mai carica e scarica nella stessa ora
   - con batteria di capacità zero, il risultato coincide col caso senza batteria
6. Disegna il piano di un giorno: carico, fotovoltaico, scambio con la rete e SOC.

**Output:** `optimize.py` testato, grafico di un giorno tipo.

---

## Fase 7 — Simulazione a orizzonte mobile

Qui i due blocchi si uniscono. Per ogni giorno del test set:

1. Genera le previsioni di carico e fotovoltaico per il giorno dopo.
2. Risolvi il MILP con le previsioni e ottieni il piano della batteria.
3. Applica il piano della batteria ai **dati reali**: la differenza fra previsto e reale viene assorbita dalla rete (compri di più o vendi di meno).
4. Se il piano non è eseguibile con i dati reali (es. batteria piena prima del previsto), applica una regola di correzione semplice e documentala.
5. Calcola costo ed emissioni reali del giorno.
6. Passa il SOC finale come SOC iniziale del giorno successivo.

**Output:** `simulate.py`, serie giornaliera di costi ed emissioni sul test set.

---

## Fase 8 — Confronti e analisi

Esegui la stessa simulazione con quattro strategie:

| Strategia | Descrizione |
|---|---|
| Nessuna batteria | Il fotovoltaico copre il carico, il resto va/viene dalla rete |
| Regola ingenua | Carica quando c'è surplus solare, scarica quando c'è deficit |
| Previsione + MILP | Il tuo sistema |
| Previsione perfetta + MILP | Limite superiore: ottimizzazione con i dati reali |

1. Tabella riassuntiva: costo totale, emissioni totali, risparmio percentuale rispetto a "nessuna batteria".
2. Calcola quanta parte del risparmio teorico (previsione perfetta) il tuo sistema cattura: è la misura del valore della previsione.
3. **Fronte di Pareto:** ripeti la simulazione variando `λ` e disegna costo vs emissioni.
4. **Analisi di sensitività:** varia capacità della batteria e potenza del fotovoltaico e osserva come cambia il risparmio.
5. Guarda i giorni peggiori: quando la previsione sbaglia di più, quanto costa?

**Output:** `evaluate.py`, notebook `03_results.ipynb`, grafici finali.

---

## Fase 9 — Qualità e riproducibilità

1. `scripts/run_all.py` esegue tutto: download, pulizia, addestramento, simulazione, grafici.
2. Tutti i parametri stanno in `config/default.yaml`, niente numeri sparsi nel codice.
3. Fissa i seed casuali.
4. Aggiungi `ruff` per formattazione e lint.
5. Configura GitHub Actions: a ogni push esegue lint e test.
6. Aggiungi docstring alle funzioni pubbliche e type hint.
7. Verifica la riproducibilità: clona il repo in una cartella nuova e segui il README da zero.

**Output:** badge verde della CI nel README.

---

## Fase 10 — README e presentazione

Struttura del README:

1. Titolo e una frase di descrizione.
2. **Risultato principale in cima:** un grafico e 2–3 numeri (es. "−X% costo, −Y% CO₂ rispetto a nessuna batteria").
3. Schema del pipeline (dati → previsione → ottimizzazione → simulazione).
4. Come eseguire: 3–4 comandi.
5. Dati: fonti, periodo, licenze.
6. Metodo: sintesi di previsione e formulazione, con link a `docs/formulation.md`.
7. Risultati: tabella dei confronti, fronte di Pareto, sensitività.
8. Limiti e semplificazioni dichiarate.
9. Possibili sviluppi.

Poi:

1. Aggiungi descrizione e topic al repository (`energy`, `optimization`, `milp`, `forecasting`, `machine-learning`).
2. Fissa il repository in evidenza sul profilo GitHub.

**Output:** repository presentabile.

---

## Fase 11 — Estensioni facoltative

1. Dashboard Streamlit: si scelgono taglia della batteria e peso `λ`, si vede il piano risultante.
2. Previsione probabilistica (quantili) e ottimizzazione stocastica o robusta su più scenari.
3. Costo di degradazione della batteria nell'obiettivo.
4. Ricarica di veicoli elettrici come carico spostabile.
5. Ri-ottimizzazione infragiornaliera ogni ora (controllo predittivo vero e proprio).

---

## Checklist riassuntiva

- [x] Fase 0 — Problema e parametri definiti
- [x] Fase 1 — Repository impostato
- [x] Fase 2 — Dati scaricabili con uno script
- [x] Fase 3 — Dataset pulito ed esplorato
- [x] Fase 4 — Previsione migliore della baseline (fotovoltaico sì, MAE −39%; carico no sul test: MAE +4%, RMSE −10%)
- [x] Fase 5 — Formulazione scritta
- [x] Fase 6 — Ottimizzatore implementato e testato
- [x] Fase 7 — Simulazione a orizzonte mobile
- [x] Fase 8 — Confronti, Pareto, sensitività
- [ ] Fase 9 — CI, test, riproducibilità
- [ ] Fase 10 — README completo
- [ ] Fase 11 — Estensioni (facoltative)
