# Microgrid EMS

[![CI](https://github.com/irreprensibile22/MicroGrid/actions/workflows/ci.yml/badge.svg)](https://github.com/irreprensibile22/MicroGrid/actions/workflows/ci.yml)

Gestione energetica di una microgrid con approccio *predict-then-optimize*: un modello di machine learning prevede carico e produzione fotovoltaica del giorno dopo, un modello MILP decide ora per ora come usare batteria e rete per minimizzare costo ed emissioni.

## Risultato principale

**Domanda: quanto si risparmia, in costo e in CO₂, gestendo la batteria con previsione + ottimizzazione rispetto a una regola semplice?**

**Risposta: con la tariffa di riferimento, niente.** Su 125 giorni di dati reali la batteria gestita con previsione + ottimizzazione riduce il costo del 20% e le emissioni del 22% rispetto a non avere la batteria. Ma una regola che carica quando c'è sole in più e scarica quando manca fa meglio: −23% di costo, −25% di emissioni.

![Costo totale per strategia](results/comparison.png)

- **−20,2% di costo** con previsione + MILP + correzione in tempo reale; **−22,7%** con la regola ingenua; **−24,2%** conoscendo il futuro.
- **Eseguire il piano alla lettera costa caro**: solo −8,5%. Far assorbire l'errore di previsione alla batteria invece che alla rete vale 711 €, tre quarti di ciò che darebbe una previsione perfetta.
- **Decide la tariffa.** Con oneri sull'acquisto fino a 0,10 €/kWh l'ottimizzazione è davanti o alla pari; con oneri nulli è l'unica strategia che guadagna. Da 0,20 €/kWh in su vince la regola ingenua.

L'analisi completa è in [notebooks/03_results.ipynb](notebooks/03_results.ipynb).

## Schema

```mermaid
flowchart LR
    A["Dati<br/>carico, fotovoltaico, meteo,<br/>prezzi, intensità di carbonio"] --> B["Pulizia<br/>dataset orario unico"]
    B --> C["Previsione<br/>LightGBM, giorno D+1"]
    C --> D["Ottimizzazione<br/>MILP, piano di 24 ore"]
    D --> E["Simulazione<br/>piano eseguito sui dati reali"]
    E --> F["Confronto<br/>5 strategie, Pareto, sensitività"]
    E -- "stato di carica reale" --> D
```

| Passo | Modulo | Comando |
|---|---|---|
| Download | [data.py](src/microgrid/data.py) | `python -m microgrid.data` |
| Pulizia | [dataset.py](src/microgrid/dataset.py) | `python -m microgrid.dataset` |
| Previsione | [features.py](src/microgrid/features.py), [forecast.py](src/microgrid/forecast.py) | `python -m microgrid.forecast` |
| Ottimizzazione | [optimize.py](src/microgrid/optimize.py) | `python -m microgrid.optimize` |
| Simulazione | [simulate.py](src/microgrid/simulate.py) | `python -m microgrid.simulate` |
| Confronto | [evaluate.py](src/microgrid/evaluate.py) | `python -m microgrid.evaluate` |

## Come eseguire

Serve Python 3.12. Dalla cartella del repository:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt && pip install -e . --no-deps
python scripts/run_all.py
```

`run_all.py` esegue l'intero pipeline in circa 10 minuti: scarica i dati, li pulisce, addestra i modelli, simula, confronta le strategie e riesegue i notebook. I dati non sono nel repository: vengono scaricati dalle fonti.

- **Test.** `pytest` esegue 90 test, senza bisogno dei dati.
- **Parametri.** Sono tutti in [config/default.yaml](config/default.yaml).
- **Riproducibilità.** Il seme casuale è fissato e LightGBM è usato in modalità deterministica. `requirements-lock.txt` contiene le versioni esatte usate per i risultati; per le più recenti: `pip install -e ".[dev]"`.
- **Controlli automatici.** A ogni push GitHub Actions esegue `ruff` (lint e formattazione) e i test.

## Problema

Due abitazioni di Costanza (Germania), gestite come un'unica microgrid connessa alla rete elettrica, con un impianto fotovoltaico e una batteria condivisa. Il gestore decide ogni giorno il piano di carica e scarica per il giorno successivo. Carico e produzione fotovoltaica non sono controllabili: si possono solo prevedere.

| Parametro | Simbolo | Valore |
|---|---|---|
| Potenza di picco del fotovoltaico | — | 100 kWp |
| Capacità della batteria | `E_max` | 200 kWh |
| Potenza massima di carica/scarica | `P_max` | 50 kW |
| Rendimenti di carica e scarica | `η_c`, `η_d` | 0,95 |
| Stato di carica minimo e massimo | `SOC_min`, `SOC_max` | 10% e 90% (20 e 180 kWh) |
| Potenza massima scambiabile con la rete | `G_max` | 100 kW |
| Oneri sull'energia acquistata | `τ` | 0,20 €/kWh (assunto) |
| Peso delle emissioni | `λ` | 0 €/kgCO₂ di default |

- **Risoluzione e orizzonte.** Passo di 1 ora; si pianificano le 24 ore del giorno D+1.
- **Tariffa.** Si acquista al prezzo day-ahead più `τ` e si vende al prezzo day-ahead.
- **Obiettivo.** Minimizzare il costo netto dell'energia scambiata con la rete più le emissioni dell'energia acquistata, pesate da `λ`.

## Dati

Cinque serie orarie sul periodo **1 novembre 2015 – 3 febbraio 2018** (826 giorni, 19 824 ore), tutte riferite a Costanza o alla Germania.

| Serie | Fonte | Contenuto | Licenza |
|---|---|---|---|
| Carico | [Open Power System Data, *Household Data*](https://data.open-power-system-data.org/household_data/2020-04-15/) (versione 2020-04-15, progetto CoSSMic) | Letture di contatore di utenze di Costanza | CC BY 4.0 |
| Fotovoltaico | [PVGIS 5.3](https://re.jrc.ec.europa.eu/pvg_tools/en/) (Commissione Europea, JRC), radiazione SARAH-3 | Produzione simulata per 1 kWp a Costanza, moduli a sud inclinati di 35°, perdite 14% | Riuso libero con citazione della fonte |
| Meteo | [Open-Meteo](https://open-meteo.com/), archivio storico | Temperatura, irraggiamento, nuvolosità a Costanza | CC BY 4.0 |
| Prezzo | [Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE), dati Bundesnetzagentur / SMARD.de | Prezzo day-ahead della zona DE-AT-LU | CC BY 4.0 |
| Intensità di carbonio | [Energy-Charts](https://energy-charts.info/) (Fraunhofer ISE), produzione elettrica tedesca per fonte ogni 15 minuti | Calcolata dal mix di produzione | CC BY 4.0 |

L'esplorazione del dataset è in [notebooks/01_exploration.ipynb](notebooks/01_exploration.ipynb).

### Pulizia

- **Indice unico.** Tutte le serie sono su un indice orario in UTC, riferito all'inizio dell'ora. I giorni sono quelli dell'ora locale di Costanza: i giorni del cambio dell'ora hanno 23 o 25 ore.
- **Giorni non validi.** Nel dataset del carico molte letture di contatore sono state interpolate dal fornitore. Un'interpolazione ripartisce in parti uguali l'energia delle ore coinvolte: il totale è corretto, l'andamento orario no. I buchi fino a 6 ore sono accettati; i giorni che contengono buchi più lunghi sono marcati come non validi (50 su 826, tutti nei dati di addestramento).
- **Valori anomali.** Un consumo negativo è trattato come valore mancante. La produzione fotovoltaica con irraggiamento nullo è portata a zero.
- **Divisione.** In ordine temporale e per giorni interi: addestramento fino al 31 maggio 2017 (70%), validazione dal 1° giugno al 1° ottobre 2017 (15%), test dal 2 ottobre 2017 al 3 febbraio 2018 (15%).

### Scelte sui dati

- **Utenze.** Il carico è la somma di due abitazioni (`residential4` e `residential5`), le sole con misure complete e poco interpolate per più di due anni. Con più abitazioni il profilo sarebbe più regolare, ma il periodo utile scenderebbe a 15 mesi e i giorni da escludere salirebbero a oltre il 20%.
- **Consumo dietro il contatore.** Per `residential4`, che ha il fotovoltaico a valle del contatore, il consumo è ricostruito come prelievo + produzione − immissione.
- **Riscalatura del carico.** Il carico aggregato reale ha media di 1,1 kW e picco di 6,3 kW. Viene moltiplicato per 7,95 per portare il picco a 50 kW, coerente con un impianto da 100 kWp. La forma del profilo è reale, la scala no: equivale a studiare le due abitazioni con un impianto otto volte più piccolo.
- **Fotovoltaico simulato.** La produzione non è misurata ma simulata da PVGIS a partire dalla radiazione satellitare delle stesse ore. Il valore di ogni ora è quello dell'istante osservato dal satellite, circa 10 minuti dopo l'inizio dell'ora.
- **Zona di mercato.** Nel periodo la Germania era nella zona DE-AT-LU, divisa in DE-LU e AT dal 1° ottobre 2018.
- **Intensità di carbonio.** È la media dei fattori di emissione sul ciclo di vita (IPCC AR5, Annex III) pesata sulla produzione nazionale di ogni fonte. I fattori sono in `config/default.yaml`.

## Metodo

### Previsione

Alla mezzanotte fra il giorno D e il giorno D+1 si prevedono carico e fotovoltaico di ogni ora di D+1, con un modello LightGBM per ciascuna grandezza.

- **Feature.** Calendario (ora, giorno della settimana, festività del Baden-Württemberg, stagione, con codifica seno/coseno), meteo del giorno D+1, valori di 24, 48 e 168 ore prima, media del giorno precedente e dei 7 giorni precedenti.
- **Nessuna informazione futura.** Le feature usano solo misure fino alla fine del giorno D; i giorni non validi non vengono usati. Un test verifica che modificare i dati del giorno D+1 non cambi le sue feature.
- **Iperparametri.** Scelti con cross-validation a 4 finestre espandibili sui soli dati di addestramento.
- **Riaddestramento.** Durante validazione e test il modello è riaddestrato ogni 7 giorni con tutte le misure disponibili fino a quel momento.
- **Confronto.** Previsione ingenua: stessa ora di 7 giorni prima per il carico, stessa ora del giorno prima per il fotovoltaico.

Risultati sul test set (3001 ore), in kW:

| Grandezza | Previsione | MAE | RMSE | MAE normalizzato |
|---|---|---|---|---|
| Carico | ingenua | 3,34 | 5,63 | 0,275 |
| Carico | LightGBM statico | 3,84 | 5,72 | 0,316 |
| Carico | LightGBM riaddestrato ogni 7 giorni | 3,46 | 5,05 | 0,285 |
| Fotovoltaico | ingenua | 5,79 | 14,09 | 0,804 |
| Fotovoltaico | LightGBM statico | 3,57 | 8,25 | 0,496 |
| Fotovoltaico | LightGBM riaddestrato ogni 7 giorni | 3,53 | 8,16 | 0,490 |

Sul fotovoltaico il modello riduce il MAE del 39%. Sul carico, nel test set, **non batte la previsione ingenua** sull'errore medio (+4%), pur commettendo meno errori grandi (RMSE −10%); in validazione la batteva del 25%. La causa è un cambio di abitudini in una delle due abitazioni nell'inverno 2017–18, che rende il carico molto regolare da una settimana all'altra. L'analisi è in [notebooks/02_forecasting.ipynb](notebooks/02_forecasting.ipynb).

### Ottimizzazione

Per ogni giorno un modello MILP sceglie, ora per ora, carica e scarica della batteria, acquisto e vendita alla rete e taglio del fotovoltaico. La formulazione completa è in [docs/formulation.md](docs/formulation.md).

```
min  Σ_t [ c_t · g_imp_t − s_t · g_exp_t + λ · e_t · g_imp_t ] · Δt
```

- **Vincoli.** Bilancio di potenza, dinamica e limiti dello stato di carica, limiti di potenza di batteria e rete, taglio del fotovoltaico non superiore alla produzione.
- **Fine giornata.** La batteria deve chiudere ogni giorno almeno al livello con cui l'ha iniziato.
- **Variabili binarie.** Impediscono carica e scarica simultanee e acquisto e vendita simultanei. Con `τ` = 0,20 €/kWh il prezzo di acquisto è sempre positivo e non cambiano il valore ottimo; servono quando acquistare è pagato.
- **Implementazione.** Pyomo con solver HiGHS. I test verificano bilancio di potenza, limiti dello stato di carica, assenza di operazioni simultanee, equivalenza con il caso senza batteria quando la capacità è zero, e un giorno con soluzione calcolata a mano.

Piano del 12 settembre 2017, calcolato su carico e fotovoltaico previsti:

![Piano di un giorno](results/optimize_example.png)

La batteria copre il carico notturno, si svuota vendendo nell'ora mattutina di prezzo più alto, si ricarica con il surplus solare e vende di nuovo nel picco serale, chiudendo al livello iniziale.

### Simulazione

Per ogni giorno del test set il MILP pianifica sulle previsioni, poi il piano è eseguito sui dati reali; l'energia rimasta in batteria è lo stato iniziale del giorno dopo. Il piano si può eseguire in due modi.

- **Alla lettera.** La batteria segue il programma; la differenza fra previsto e reale è assorbita dalla rete.
- **Con correzione in tempo reale.** Si segue lo scambio con la rete previsto dal piano e si lascia che sia la batteria ad assorbire l'errore, con due limiti: carica al più il surplus reale e scarica al più il deficit reale, oltre agli scambi con la rete che il piano aveva deciso di proposito. La regola è stata scelta sul periodo di validazione, fra due alternative.

In entrambi i casi la batteria resta nei limiti dello stato di carica, e il fotovoltaico si taglia se il prezzo di vendita è negativo o se la vendita supererebbe `G_max`.

### Strategie a confronto

| Strategia | Descrizione |
|---|---|
| Nessuna batteria | Il fotovoltaico copre il carico, il resto va alla rete o viene dalla rete |
| Regola ingenua | Carica quando c'è surplus solare, scarica quando c'è deficit |
| Previsione + MILP | Piano calcolato sulle previsioni ed eseguito alla lettera |
| Previsione + MILP + correzione | Stesso piano, eseguito con la correzione in tempo reale |
| Previsione perfetta + MILP | Piano calcolato sui dati reali: riferimento teorico |

## Risultati

Test set, 125 giorni (2 ottobre 2017 – 3 febbraio 2018):

| Strategia | Costo | Emissioni | Risparmio di costo | Risparmio di emissioni | Quota del risparmio teorico |
|---|---|---|---|---|---|
| Nessuna batteria | 6 069 € | 10 496 kgCO₂ | — | — | — |
| Regola ingenua | 4 691 € | 7 878 kgCO₂ | 22,7% | 24,9% | 94% |
| Previsione + MILP | 5 554 € | 9 527 kgCO₂ | 8,5% | 9,2% | 35% |
| Previsione + MILP + correzione | 4 842 € | 8 208 kgCO₂ | 20,2% | 21,8% | 84% |
| Previsione perfetta + MILP | 4 602 € | 7 802 kgCO₂ | 24,2% | 25,7% | 100% |

- **Perché la regola ingenua vince.** Con 0,20 €/kWh di oneri un kWh autoconsumato vale circa sette volte la differenza di prezzo tipica fra le ore di un giorno. Conta quasi solo l'autoconsumo, e per massimizzarlo basta reagire al surplus reale: non serve prevederlo. Fra regola ingenua e previsione perfetta c'è meno del 2% del costo.
- **Costo dell'errore di previsione.** Eseguendo il piano alla lettera il costo reale supera del 31% quello pianificato (5 554 € contro 4 251 €). Il costo in più dipende dall'errore sul fotovoltaico (correlazione +0,71 con il MAE del giorno), non da quello sul carico (−0,08).
- **Valore della previsione del carico.** Sostituire il modello con la previsione ingenua del carico costa 26 € in più: per questo problema si equivalgono.

### Sensitività

![Risparmio al variare di un parametro alla volta](results/sensitivity.png)

- **Oneri sull'acquisto.** È il parametro che decide chi vince. Con oneri nulli il MILP porta il costo da 496 € a 7 €, mentre la regola ingenua perde 47 € rispetto a non avere la batteria; le emissioni del MILP però raddoppiano, perché compra e rivende energia di rete. Con 0,05 €/kWh il piano con correzione risparmia 465 € contro 310 €; con 0,10 €/kWh sono alla pari; da 0,20 €/kWh vince la regola ingenua.
- **Capacità della batteria.** Rendimenti decrescenti: da 100 a 200 kWh la regola ingenua guadagna 377 €, da 200 a 400 kWh 192 €. Il piano eseguito alla lettera resta fermo a circa 500 € con qualunque capacità.
- **Potenza del fotovoltaico.** Più sole c'è, più la batteria rende, per tutte le strategie.

### Fronte di Pareto costo–emissioni

![Costo ed emissioni al variare di λ](results/pareto.png)

Con previsione perfetta, pesare le emissioni fino a 5 €/kgCO₂ le riduce del 2,6% e aumenta il costo dell'1,8%. Il grosso arriva subito: a 1 €/kgCO₂ si evitano 188 kg con 43 € in più, circa 230 € per tonnellata. Le emissioni dipendono da quanta energia si acquista più che da quando.

## Limiti e semplificazioni

- **Test set invernale e unico.** Il confronto è su 125 giorni di autunno e inverno, la stagione con più carico e meno fotovoltaico. Non rappresenta un anno intero.
- **Meteo reale al posto delle previsioni.** Il meteo del giorno D+1 è quello osservato: per il periodo di studio non esistono previsioni archiviate. L'errore sul fotovoltaico, e quindi il costo dell'errore di previsione, è ottimistico.
- **Tariffa semplificata.** Oneri fissi sull'acquisto, vendita al prezzo day-ahead, nessuna tariffa sulla potenza di picco né incentivo all'immissione. Il valore di `τ` è un'ipotesi, e i risultati ne dipendono molto.
- **Intensità di carbonio nota in anticipo.** Il MILP usa il valore reale del giorno D+1. Inoltre non considera gli scambi con l'estero e usa lo stesso fattore per lignite e carbone: le emissioni sono sottostimate.
- **La previsione perfetta non è un limite superiore assoluto.** Il MILP pianifica un giorno alla volta e deve chiudere ogni giorno al livello di partenza. Con una batteria da 400 kWh regola ingenua e piano con correzione lo raggiungono o lo superano.
- **Energia residua.** Regola ingenua e piano con correzione finiscono il periodo con la batteria al minimo, le altre strategie a metà: la differenza vale circa 18 € e non cambia la classifica.
- **Scala del carico.** Il profilo è reale, la scala no (fattore 7,95).
- **Ore senza dati satellitari.** In 69 ore su 19 824 la produzione simulata è zero mentre Open-Meteo indica un irraggiamento superiore a 50 W/m²; in 57 di queste PVGIS riporta irraggiamento nullo, per esempio per tutto il 16 ottobre 2016. Sono con ogni probabilità ore in cui manca il dato del satellite. Non vengono corrette; solo 2 cadono nel test set.
- **Scelte fatte conoscendo il test.** Il riaddestramento settimanale dei modelli è stato introdotto dopo aver visto il risultato del modello statico sul test. Non usa dati futuri e gli iperparametri non sono cambiati.
- **Emissione a mezzanotte.** Un vero mercato del giorno prima richiede la previsione entro mezzogiorno del giorno D, con mezza giornata di misure in meno.
- **Batteria ideale.** Rendimenti costanti, nessuna autoscarica, nessun costo di degradazione.

## Possibili sviluppi

- **Ri-ottimizzazione infragiornaliera.** Ripianificare ogni ora con le misure più recenti (controllo predittivo) invece di correggere con una regola.
- **Previsione probabilistica e ottimizzazione stocastica.** Pianificare su più scenari di fotovoltaico, la grandezza il cui errore costa.
- **Previsioni meteo vere** al posto del meteo osservato, e un test set di un anno intero.
- **Orizzonte più lungo.** Pianificare su 48 ore, o dare un valore all'energia residua, per togliere il vincolo di fine giornata.
- **Tariffe diverse.** Prezzi dinamici al dettaglio, tariffa sulla potenza di picco: sono i casi in cui l'ottimizzazione ha più da guadagnare.
- **Costo di degradazione** della batteria nell'obiettivo.
- **Carichi spostabili**, come la ricarica di veicoli elettrici.
- **Dashboard interattiva** per scegliere taglia della batteria e peso `λ`.

## Struttura

```
config/default.yaml     parametri
docs/formulation.md     formulazione matematica del MILP
notebooks/              esplorazione, previsione, risultati
results/                tabelle e grafici prodotti dal pipeline
scripts/run_all.py      esegue tutto
src/microgrid/          codice
tests/                  test
```

Il piano di lavoro seguito è in [microgrid-roadmap.md](microgrid-roadmap.md). Licenza: [MIT](LICENSE).
