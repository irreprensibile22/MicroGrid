# Microgrid EMS

Gestione energetica di una microgrid con approccio *predict-then-optimize*: un modello di machine learning prevede carico e produzione fotovoltaica del giorno dopo, un modello MILP decide ora per ora come usare batteria e rete per minimizzare costo ed emissioni.

> Progetto in costruzione. Il piano di lavoro è in [microgrid-roadmap.md](microgrid-roadmap.md).

## Problema

### Domanda

**Quanto si risparmia, in costo e in CO₂, gestendo la batteria con previsione + ottimizzazione rispetto a una regola semplice?**

### Scenario

Due abitazioni di Costanza (Germania), gestite come un'unica microgrid connessa alla rete elettrica, dotata di:

- un impianto fotovoltaico,
- una batteria di accumulo condivisa,
- un punto di connessione alla rete, da cui può acquistare e a cui può vendere energia.

Il gestore decide ogni giorno il piano di carica e scarica della batteria per il giorno successivo. Il carico e la produzione fotovoltaica non sono controllabili: si possono solo prevedere.

Perché questo scenario:

- **Dati reali a scala di edificio.** Il carico viene dalle misure del dataset *household data* di Open Power System Data, non da un carico nazionale riscalato.
- **Profilo adatto all'accumulo.** Il fotovoltaico supera il carico solo in un terzo delle ore: c'è surplus da spostare verso la sera e la notte.
- **Coerenza geografica.** Meteo, prezzi day-ahead e intensità di carbonio si riferiscono alla stessa località, alla stessa zona di mercato e allo stesso periodo.

Utenze, periodo e riscalatura del carico sono descritti nella sezione [Dati](#dati).

### Parametri dell'impianto

Valori di partenza; capacità della batteria e potenza del fotovoltaico verranno variate nell'analisi di sensitività.

| Parametro | Simbolo | Valore |
|---|---|---|
| Potenza di picco del fotovoltaico | — | 100 kWp |
| Capacità della batteria | `E_max` | 200 kWh |
| Potenza massima di carica/scarica | `P_max` | 50 kW |
| Rendimento di carica | `η_c` | 0,95 |
| Rendimento di scarica | `η_d` | 0,95 |
| Stato di carica minimo | `SOC_min` | 10% (20 kWh) |
| Stato di carica massimo | `SOC_max` | 90% (180 kWh) |

### Risoluzione e orizzonte

- Passo temporale: 1 ora (`Δt = 1 h`).
- Orizzonte di pianificazione: le 24 ore del giorno successivo (D+1), decise al giorno D.
- La simulazione procede a orizzonte mobile: lo stato di carica a fine giornata è quello iniziale del giorno dopo.

### Obiettivo

Minimizzare il costo netto dell'energia scambiata con la rete più le emissioni associate all'energia acquistata, pesate da un coefficiente `λ` [€/kgCO₂]. Variando `λ` si ottiene il compromesso costo–emissioni (fronte di Pareto).

### Termini di confronto

Il sistema viene valutato contro tre riferimenti:

1. **Nessuna batteria** — il fotovoltaico copre il carico, il resto va alla rete o viene dalla rete.
2. **Regola ingenua** — carica quando c'è surplus solare, scarica quando c'è deficit.
3. **Previsione perfetta + MILP** — limite superiore teorico, ottimizzando sui dati reali.

## Installazione

Serve Python 3.10 o successivo. Dalla cartella del repository:

```
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Dati

Cinque serie orarie sul periodo **1 novembre 2015 – 3 febbraio 2018** (826 giorni, 19 824 ore), tutte riferite a Costanza o alla Germania.

| Serie | Fonte | Contenuto | Licenza |
|---|---|---|---|
| Carico | [Open Power System Data, *Household Data*](https://data.open-power-system-data.org/household_data/2020-04-15/) (versione 2020-04-15, progetto CoSSMic) | Letture di contatore di utenze di Costanza | CC BY 4.0 |
| Fotovoltaico | [PVGIS 5.3](https://re.jrc.ec.europa.eu/pvg_tools/en/) (Commissione Europea, JRC), radiazione SARAH-3 | Produzione simulata per 1 kWp a Costanza, moduli a sud inclinati di 35°, perdite 14% | Riuso libero con citazione della fonte |
| Meteo | [Open-Meteo](https://open-meteo.com/), archivio storico | Temperatura, irraggiamento, nuvolosità a Costanza | CC BY 4.0 |
| Prezzo | [Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE), dati Bundesnetzagentur / SMARD.de | Prezzo day-ahead della zona DE-AT-LU | CC BY 4.0 |
| Intensità di carbonio | [Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE), produzione elettrica tedesca per fonte ogni 15 minuti | Calcolata dal mix di produzione | CC BY 4.0 |

I dati non sono nel repository. Si scaricano e si puliscono con:

```
python -m microgrid.data       # scarica le serie grezze in data/raw/
python -m microgrid.dataset    # costruisce data/processed/dataset.parquet
```

L'esplorazione del dataset è in [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb).

### Pulizia

- **Indice unico.** Tutte le serie sono su un indice orario in UTC, riferito all'inizio dell'ora. I giorni sono quelli dell'ora locale di Costanza: i giorni del cambio dell'ora hanno 23 o 25 ore.
- **Giorni non validi.** Nessuna serie ha ore mancanti (per il fotovoltaico vedi però *Ore senza dati satellitari* più sotto), ma nel dataset del carico molte letture di contatore sono state interpolate dal fornitore. Un'interpolazione ripartisce in parti uguali l'energia delle ore coinvolte: il totale è corretto, l'andamento orario no. I buchi fino a 6 ore sono accettati; i giorni che contengono buchi più lunghi sono marcati come non validi (50 su 826, tutti nei dati di addestramento).
- **Valori anomali.** Un consumo negativo è trattato come valore mancante. La produzione fotovoltaica con irraggiamento nullo è portata a zero.
- **Divisione.** In ordine temporale e per giorni interi: addestramento fino al 31 maggio 2017 (70%), validazione dal 1° giugno al 1° ottobre 2017 (15%), test dal 2 ottobre 2017 al 3 febbraio 2018 (15%).

### Scelte e semplificazioni

- **Utenze.** Il carico è la somma di due abitazioni (`residential4` e `residential5`), le sole con misure complete e poco interpolate per più di due anni. Con più abitazioni il profilo sarebbe più regolare, ma il periodo utile scenderebbe a 15 mesi e i giorni da escludere salirebbero a oltre il 20%.
- **Consumo dietro il contatore.** Per `residential4`, che ha il fotovoltaico a valle del contatore, il consumo è ricostruito come prelievo + produzione − immissione.
- **Riscalatura del carico.** Il carico aggregato reale ha media di 1,1 kW e picco di 6,3 kW. Viene moltiplicato per 7,95 per portare il picco a 50 kW, coerente con un impianto da 100 kWp. La forma del profilo è reale, la scala no: equivale a studiare le due abitazioni con un impianto otto volte più piccolo.
- **Fotovoltaico simulato.** La produzione non è misurata ma simulata da PVGIS a partire dalla radiazione satellitare delle stesse ore. Il valore di ogni ora è quello dell'istante osservato dal satellite, circa 10 minuti dopo l'inizio dell'ora.
- **Ore senza dati satellitari.** In 69 ore su 19 824 la produzione simulata è zero mentre Open-Meteo indica un irraggiamento superiore a 50 W/m²; in 57 di queste PVGIS riporta irraggiamento nullo, per esempio per tutto il 16 ottobre 2016. Sono con ogni probabilità ore in cui manca il dato del satellite. Non vengono corrette: la produzione di quelle ore è sottostimata. Solo 2 cadono nel test set.
- **Test set invernale.** Il test set copre solo autunno e inverno, la stagione con più carico e meno fotovoltaico. I risultati non rappresentano un anno intero.
- **Zona di mercato.** Nel periodo la Germania era nella zona DE-AT-LU, divisa in DE-LU e AT dal 1° ottobre 2018.
- **Intensità di carbonio.** È la media dei fattori di emissione sul ciclo di vita (IPCC AR5, Annex III) pesata sulla produzione nazionale di ogni fonte. Non considera gli scambi con l'estero. Il rapporto IPCC non distingue la lignite dal carbone: si usa lo stesso fattore, quindi le emissioni sono sottostimate. I fattori sono in `config/default.yaml`.

## Previsione

Alla mezzanotte fra il giorno D e il giorno D+1 si prevedono carico e fotovoltaico di ogni ora di D+1, con un modello LightGBM per ciascuna grandezza.

```
python -m microgrid.forecast    # addestra, prevede validazione e test, scrive results/
```

- **Feature.** Calendario (ora, giorno della settimana, festività del Baden-Württemberg, stagione, con codifica seno/coseno), meteo del giorno D+1, valori di 24, 48 e 168 ore prima, media del giorno precedente e dei 7 giorni precedenti.
- **Nessuna informazione futura.** Le feature usano solo misure fino alla fine del giorno D; i giorni non validi non vengono usati. Un test verifica che modificare i dati del giorno D+1 non cambi le sue feature.
- **Iperparametri.** Scelti con cross-validation a 4 finestre espandibili sui soli dati di addestramento.
- **Riaddestramento.** Durante validazione e test il modello è riaddestrato ogni 7 giorni con tutte le misure disponibili fino a quel momento.
- **Confronto.** Previsione ingenua: stessa ora di 7 giorni prima per il carico, stessa ora del giorno prima per il fotovoltaico.

Risultati sul test set (2 ottobre 2017 – 3 febbraio 2018, 3001 ore), in kW:

| Grandezza | Previsione | MAE | RMSE | MAE normalizzato |
|---|---|---|---|---|
| Carico | ingenua | 3,34 | 5,63 | 0,275 |
| Carico | LightGBM statico | 3,84 | 5,72 | 0,316 |
| Carico | LightGBM riaddestrato ogni 7 giorni | 3,46 | 5,05 | 0,285 |
| Fotovoltaico | ingenua | 5,79 | 14,09 | 0,804 |
| Fotovoltaico | LightGBM statico | 3,57 | 8,25 | 0,496 |
| Fotovoltaico | LightGBM riaddestrato ogni 7 giorni | 3,53 | 8,16 | 0,490 |

Sul fotovoltaico il modello riduce il MAE del 39%. Sul carico, nel test set, **non batte la previsione ingenua** sull'errore medio (+4%), pur commettendo meno errori grandi (RMSE −10%); in validazione la batteva del 25%. La causa è un cambio di abitudini in una delle due abitazioni nell'inverno 2017–18, che rende il carico molto regolare da una settimana all'altra. Le metriche complete sono in [results/forecast_metrics.csv](results/forecast_metrics.csv), l'analisi in [notebooks/02_forecasting.ipynb](notebooks/02_forecasting.ipynb).

Limiti:

- **Meteo reale al posto delle previsioni.** Il meteo del giorno D+1 è quello effettivamente osservato: per il periodo di studio non esistono previsioni meteo archiviate. L'errore sul fotovoltaico è quindi ottimistico.
- **Riaddestramento introdotto dopo aver visto il test.** Il modello statico era stato valutato per primo; il riaddestramento periodico è stato aggiunto dopo. Non usa dati futuri e gli iperparametri non sono cambiati, ma è una scelta fatta conoscendo il risultato sul test.
- **Emissione a mezzanotte.** Un vero mercato del giorno prima richiede la previsione entro mezzogiorno del giorno D, con mezza giornata di misure in meno.
