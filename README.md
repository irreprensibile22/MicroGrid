# Microgrid EMS

Gestione energetica di una microgrid con approccio *predict-then-optimize*: un modello di machine learning prevede carico e produzione fotovoltaica del giorno dopo, un modello MILP decide ora per ora come usare batteria e rete per minimizzare costo ed emissioni.

> Progetto in costruzione. Il piano di lavoro è in [microgrid-roadmap.md](microgrid-roadmap.md).

## Problema

### Domanda

**Quanto si risparmia, in costo e in CO₂, gestendo la batteria con previsione + ottimizzazione rispetto a una regola semplice?**

### Scenario

Quattro abitazioni di Costanza (Germania), gestite come un'unica microgrid connessa alla rete elettrica, dotata di:

- un impianto fotovoltaico,
- una batteria di accumulo condivisa,
- un punto di connessione alla rete, da cui può acquistare e a cui può vendere energia.

Il gestore decide ogni giorno il piano di carica e scarica della batteria per il giorno successivo. Il carico e la produzione fotovoltaica non sono controllabili: si possono solo prevedere.

Perché questo scenario:

- **Dati reali a scala di edificio.** Il carico viene dalle misure del dataset *household data* di Open Power System Data, non da un carico nazionale riscalato.
- **Profilo adatto all'accumulo.** Il carico è distribuito su tutta la giornata, con consumi serali e notturni che il fotovoltaico non copre: c'è surplus da spostare.
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

## Dati

Cinque serie orarie sul periodo **1 novembre 2015 – 31 gennaio 2017** (15 mesi, 10 992 ore), tutte riferite a Costanza o alla Germania. Nessuna ha ore mancanti nel periodo.

| Serie | Fonte | Contenuto | Licenza |
|---|---|---|---|
| Carico | [Open Power System Data, *Household Data*](https://data.open-power-system-data.org/household_data/2020-04-15/) (versione 2020-04-15, progetto CoSSMic) | Letture di contatore di utenze di Costanza | CC BY 4.0 |
| Fotovoltaico | [PVGIS 5.3](https://re.jrc.ec.europa.eu/pvg_tools/en/) (Commissione Europea, JRC), radiazione SARAH-3 | Produzione simulata per 1 kWp a Costanza, moduli a sud inclinati di 35°, perdite 14% | Riuso libero con citazione della fonte |
| Meteo | [Open-Meteo](https://open-meteo.com/), archivio storico | Temperatura, irraggiamento, nuvolosità a Costanza | CC BY 4.0 |
| Prezzo | [Energy-Charts](https://api.energy-charts.info/) (Fraunhofer ISE), dati Bundesnetzagentur / SMARD.de | Prezzo day-ahead della zona DE-AT-LU | CC BY 4.0 |
| Intensità di carbonio | [Energy-Charts](https://api.energy-charts.info/), produzione elettrica tedesca per fonte ogni 15 minuti | Calcolata dal mix di produzione | Non indicata nella risposta dell'API: da verificare |

I dati non sono nel repository. Si scaricano con:

```
python -m microgrid.data
```

Scelte e semplificazioni:

- **Utenze.** Il carico è la somma di quattro abitazioni (`residential1`, `2`, `4`, `5`). Il periodo è la finestra in cui tutte e quattro hanno misure complete. Le altre utenze residenziali del dataset sono escluse: `residential3` non copre il periodo, per `residential6` il consumo ricostruito non è attendibile (segue l'andamento del sole).
- **Consumo dietro il contatore.** Per `residential4`, che ha il fotovoltaico a valle del contatore, il consumo è ricostruito come prelievo + produzione − immissione.
- **Riscalatura del carico.** Il carico aggregato reale ha media di circa 2 kW e picco di circa 10 kW. Viene riscalato a un picco di 50 kW per essere coerente con un impianto da 100 kWp. La forma del profilo è reale, la scala no: equivale a studiare le quattro abitazioni con un impianto cinque volte più piccolo.
- **Fotovoltaico simulato.** La produzione non è misurata ma simulata da PVGIS a partire dalla radiazione satellitare delle stesse ore.
- **Zona di mercato.** Nel periodo la Germania era nella zona DE-AT-LU, divisa in DE-LU e AT dal 1° ottobre 2018.
- **Intensità di carbonio.** È la media dei fattori di emissione sul ciclo di vita (IPCC AR5, Annex III) pesata sulla produzione nazionale di ogni fonte. Non considera gli scambi con l'estero. Il rapporto IPCC non distingue la lignite dal carbone: si usa lo stesso fattore, quindi le emissioni sono sottostimate. I fattori sono in `config/default.yaml`.
