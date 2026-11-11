# Formulazione matematica

Modello di ottimizzazione che decide, per ogni ora del giorno D+1, quanto caricare o scaricare la batteria e quanto scambiare con la rete. È un problema di programmazione lineare mista intera (MILP), risolto una volta al giorno.

I valori dei parametri sono in [`config/default.yaml`](../config/default.yaml); la chiave corrispondente è indicata accanto a ogni simbolo.

## 1. Insiemi

| Simbolo | Significato |
|---|---|
| $\mathcal{T} = \{1, \dots, T\}$ | Ore del giorno locale D+1. $T = 24$, tranne nei giorni del cambio dell'ora ($T = 23$ o $T = 25$). |

L'ora $t$ è l'intervallo di durata $\Delta t = 1\ \text{h}$ (`time.step_hours`) che finisce all'istante $t$. Le potenze sono medie sull'intervallo; lo stato di carica $soc_t$ è riferito alla fine dell'intervallo.

## 2. Parametri

### Serie orarie

| Simbolo | Unità | Significato | Origine |
|---|---|---|---|
| $L_t$ | kW | Carico previsto | `load_forecast` |
| $PV_t$ | kW | Produzione fotovoltaica prevista | `pv_forecast` |
| $\pi_t$ | €/kWh | Prezzo day-ahead | `price_eur_kwh` |
| $e_t$ | kgCO₂/kWh | Intensità di carbonio della rete | `carbon_kg_kwh` |

Prezzo e intensità di carbonio del giorno D+1 sono considerati noti al momento della decisione (vedi [Ipotesi](#7-ipotesi-e-limiti)).

### Prezzi di scambio

| Simbolo | Unità | Definizione | Chiave |
|---|---|---|---|
| $\tau$ | €/kWh | Oneri, tasse e costi di rete sull'energia acquistata | `grid.import_surcharge_eur_kwh` |
| $c_t$ | €/kWh | Prezzo di acquisto: $c_t = \pi_t + \tau$ | — |
| $s_t$ | €/kWh | Prezzo di vendita: $s_t = \pi_t$ | — |

Poiché $\tau \ge 0$, vale sempre $s_t \le c_t$: comprare e rivendere nella stessa ora non conviene mai. Il prezzo di vendita può essere negativo: in quelle ore vendere costa.

### Impianto

| Simbolo | Unità | Significato | Chiave |
|---|---|---|---|
| $E^{max}$ | kWh | Capacità della batteria | `battery.capacity_kwh` |
| $P^{max}$ | kW | Potenza massima di carica e di scarica | `battery.max_power_kw` |
| $\eta_c,\ \eta_d$ | — | Rendimenti di carica e di scarica | `battery.eta_charge`, `battery.eta_discharge` |
| $SOC^{min},\ SOC^{max}$ | kWh | Limiti dello stato di carica: frazione di $E^{max}$ | `battery.soc_min_frac`, `battery.soc_max_frac` |
| $SOC_0$ | kWh | Energia in batteria all'inizio del giorno | vedi sotto |
| $G^{max}$ | kW | Potenza massima scambiabile con la rete | `grid.max_power_kw` |
| $\lambda$ | €/kgCO₂ | Peso delle emissioni nell'obiettivo | `objective.carbon_weight_eur_kg` |

$SOC_0$ vale `battery.soc_initial_frac` $\cdot E^{max}$ il primo giorno della simulazione; nei giorni successivi è l'energia realmente rimasta in batteria alla fine del giorno precedente.

## 3. Variabili

| Simbolo | Dominio | Unità | Significato |
|---|---|---|---|
| $p^{ch}_t$ | $\ge 0$ | kW | Potenza di carica della batteria |
| $p^{dis}_t$ | $\ge 0$ | kW | Potenza di scarica della batteria |
| $g^{imp}_t$ | $\ge 0$ | kW | Potenza acquistata dalla rete |
| $g^{exp}_t$ | $\ge 0$ | kW | Potenza venduta alla rete |
| $curt_t$ | $\ge 0$ | kW | Produzione fotovoltaica non utilizzata |
| $soc_t$ | $\in \mathbb{R}$ | kWh | Energia in batteria alla fine dell'ora $t$ |
| $u_t$ | $\{0, 1\}$ | — | 1 se la batteria può caricare, 0 se può scaricare |
| $v_t$ | $\{0, 1\}$ | — | 1 se si può acquistare, 0 se si può vendere |

Le potenze di carica e scarica sono misurate sul lato dell'impianto: $p^{ch}_t$ è la potenza assorbita dalla batteria, $p^{dis}_t$ quella che la batteria restituisce. Le perdite sono nella dinamica dello stato di carica.

Con $T = 24$ il modello ha 144 variabili continue e 48 binarie.

## 4. Funzione obiettivo

$$
\min \ \sum_{t \in \mathcal{T}} \Big[ \underbrace{c_t \, g^{imp}_t - s_t \, g^{exp}_t}_{\text{costo netto}} + \lambda \, \underbrace{e_t \, g^{imp}_t}_{\text{emissioni}} \Big] \, \Delta t
$$

Il primo termine è il costo netto dell'energia scambiata con la rete [€]. Il secondo sono le emissioni dell'energia acquistata [kgCO₂], convertite in euro dal peso $\lambda$. L'energia venduta non riceve alcun credito di emissioni.

Con $\lambda = 0$ si minimizza il solo costo. Aumentando $\lambda$ il piano sposta gli acquisti verso le ore in cui la rete è più pulita: la sequenza delle soluzioni al variare di $\lambda$ è il fronte di Pareto costo–emissioni.

## 5. Vincoli

Tutti valgono per ogni $t \in \mathcal{T}$, salvo dove indicato. Nella (2), per $t = 1$ il termine $soc_{t-1}$ è il parametro $SOC_0$.

**(1) Bilancio di potenza.** Ciò che entra nel nodo dell'impianto è uguale a ciò che esce.

$$
PV_t - curt_t + p^{dis}_t + g^{imp}_t = L_t + p^{ch}_t + g^{exp}_t
$$

**(2) Dinamica dello stato di carica.**

$$
soc_t = soc_{t-1} + \eta_c \, p^{ch}_t \, \Delta t - \frac{p^{dis}_t \, \Delta t}{\eta_d}
$$

**(3) Limiti dello stato di carica.**

$$
SOC^{min} \le soc_t \le SOC^{max}
$$

**(4) Carica e scarica non simultanee.**

$$
p^{ch}_t \le P^{max} \, u_t \qquad\qquad p^{dis}_t \le P^{max} \, (1 - u_t)
$$

**(5) Acquisto e vendita non simultanei.**

$$
g^{imp}_t \le G^{max} \, v_t \qquad\qquad g^{exp}_t \le G^{max} \, (1 - v_t)
$$

**(6) Taglio del fotovoltaico.** Non si può tagliare più di quanto si produce.

$$
curt_t \le PV_t
$$

**(7) Fine giornata.** La batteria non può finire il giorno più scarica di come l'ha iniziato.

$$
soc_T \ge SOC_0
$$

## 6. Proprietà

**Il modello ha sempre soluzione** se $SOC^{min} \le SOC_0 \le SOC^{max}$ e $L_t \le G^{max}$ per ogni $t$. Basta non usare la batteria: se $PV_t < L_t$ si acquista la differenza, altrimenti il surplus si vende fino a $G^{max}$ e il resto si taglia. Nei dati il carico non supera 50 kW e $G^{max} = 100$ kW.

**I coefficienti dei vincoli (4) e (5) sono i più stretti possibili.** $P^{max}$ e $G^{max}$ sono già i limiti fisici delle variabili: non servono costanti "big-M" arbitrarie, e il rilassamento lineare è forte.

**A cosa servono le variabili binarie.** Senza $u_t$ e $v_t$ il problema è un programma lineare (LP). Le binarie cambiano la soluzione solo quando sprecare energia conviene:

- *Carica e scarica simultanee* dissipano energia per via dei rendimenti. Togliendo $\delta$ dalla carica e $\eta_c \eta_d \, \delta$ dalla scarica lo stato di carica non cambia e si libera la potenza $(1 - \eta_c \eta_d)\,\delta > 0$, che riduce l'acquisto o aumenta il taglio. Se $c_t + \lambda e_t \ge 0$ il costo non peggiora: lo spreco non conviene mai. Se invece $c_t + \lambda e_t < 0$, acquistare è pagato e il programma lineare caricherebbe e scaricherebbe insieme per consumare di più: il vincolo (4) lo impedisce.
- *Acquisto e vendita simultanei* di una potenza $\delta$ costano $(c_t + \lambda e_t - s_t)\,\delta = (\tau + \lambda e_t)\,\delta \ge 0$. Non convengono mai; con $\tau = 0$ e $\lambda = 0$ sono indifferenti, e il vincolo (5) esclude le soluzioni fisicamente prive di senso.

Con i parametri di default ($\tau = 0{,}20$ €/kWh) il prezzo di acquisto è sempre positivo: il prezzo day-ahead più basso del periodo è −0,13 €/kWh. Il valore ottimo del MILP coincide quindi con quello del rilassamento lineare. Le binarie restano nel modello perché garantiscono un piano eseguibile per qualunque scelta di $\tau$ e $\lambda$, anche nell'analisi di sensitività.

**Il vincolo (7) evita l'effetto di fine orizzonte.** Senza di esso il modello svuoterebbe la batteria nell'ultima ora, perché l'energia rimasta non ha valore nell'obiettivo. Il vincolo usa $SOC_0$ del giorno stesso: è sempre soddisfacibile, qualunque sia lo stato in cui la batteria arriva.

## 7. Ipotesi e limiti

- **Prezzi noti.** I prezzi day-ahead di D+1 sono pubblicati verso le 12:45 del giorno D: alla mezzanotte, quando si decide, sono noti. L'intensità di carbonio di D+1 invece non lo è: si usa il valore reale, come se fosse una previsione perfetta. I risultati sulle emissioni sono quindi ottimistici.
- **Tariffa semplificata.** Il prezzo di acquisto è il prezzo day-ahead più un onere fisso $\tau$; quello di vendita è il prezzo day-ahead. Non ci sono tariffe sulla potenza di picco né incentivi all'immissione. Il valore di $\tau$ è un'ipotesi: determina quanto vale l'autoconsumo rispetto all'arbitraggio sui prezzi, e va variato nell'analisi di sensitività.
- **Batteria ideale.** Rendimenti costanti, nessuna autoscarica, nessun costo di degradazione, potenza massima indipendente dallo stato di carica.
- **Orizzonte di un giorno.** Il piano non vede oltre la mezzanotte. Il vincolo (7) obbliga a chiudere ogni giorno almeno al livello di partenza: l'energia accumulata non può essere trasferita al giorno dopo in quantità inferiore a $SOC_0$, anche quando converrebbe.
- **Modello deterministico.** Le previsioni sono trattate come certe. Gli scostamenti fra previsto e reale sono assorbiti dalla rete durante la simulazione (Fase 7), non dal modello.
- **Bilancio di sola potenza attiva**, su un unico nodo: nessuna rete interna, nessuna perdita di linea.

## 8. Estensioni possibili

- Costo di degradazione: termine $\gamma \, (p^{ch}_t + p^{dis}_t) \, \Delta t$ nell'obiettivo.
- Tariffa sulla potenza di picco: variabile $g^{peak} \ge g^{imp}_t$ con un costo nell'obiettivo.
- Valore dell'energia residua al posto del vincolo (7): termine $-\,\omega \, soc_T$ nell'obiettivo.
- Orizzonte di 48 ore, applicando solo le prime 24.
