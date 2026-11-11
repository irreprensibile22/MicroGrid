# Microgrid EMS

Gestione energetica di una microgrid con approccio *predict-then-optimize*: un modello di machine learning prevede carico e produzione fotovoltaica del giorno dopo, un modello MILP decide ora per ora come usare batteria e rete per minimizzare costo ed emissioni.

> Progetto in costruzione. Il piano di lavoro è in [microgrid-roadmap.md](microgrid-roadmap.md).

## Problema

### Domanda

**Quanto si risparmia, in costo e in CO₂, gestendo la batteria con previsione + ottimizzazione rispetto a una regola semplice?**

### Scenario

Un piccolo campus (o un edificio del terziario) connesso alla rete elettrica, dotato di:

- un impianto fotovoltaico,
- una batteria di accumulo,
- un punto di connessione alla rete, da cui può acquistare e a cui può vendere energia.

Il gestore decide ogni giorno il piano di carica e scarica della batteria per il giorno successivo. Il carico e la produzione fotovoltaica non sono controllabili: si possono solo prevedere.

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
