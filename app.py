"""Regret statt Fehler: die beste Prognose bestellt nicht die beste Menge - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Wurzel der Reihe „Entscheidungsorientiertes Lernen“: Newsvendor mit Merkmalen, Regret, zweistufige Pipeline (Prognose, dann Entscheidungsregel).
Aufbau und Regeln stehen im DEMO-PLAYBOOK.md (§2).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import rsf_algorithm as A
import rsf_constants as C
from rsf_evaluation import (
    Settings,
    analyse_cell,
    analyse_instance,
    analyse_shift,
    de,
    learning_rows,
    row_for,
    vormessung,
    sweep_rows,
    tau_list,
    verdict_cell,
    verdict_kernfrage,
    verdict_learning,
    verdict_shift,
)
from rsf_layout import dunkles_design, kopf_css
from rsf_presets import (
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from rsf_visualization import build_cell, build_instance, build_kern, build_learning, build_shift

st.set_page_config(page_title="Regret statt Fehler – Sebastian Hanisch", layout="wide")

SPINNER = "Rechne 30 feste Seeds (je Zelle einige Sekunden) …"


@st.cache_data(show_spinner=SPINNER)
def _sweep(p, n, taus):
    return sweep_rows(p, n, taus)


@st.cache_data(show_spinner=SPINNER)
def _learning(p, tau):
    return learning_rows(p, tau)


@st.cache_data(show_spinner="Rechne die Einzelinstanz …")
def _instance(settings):
    return analyse_instance(settings)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in rsf_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("📉 Regret statt Fehler: die beste Prognose bestellt nicht die beste Menge")
st.markdown(
    """
Eine Prognose wird meist danach bewertet, wie nah sie an der Wahrheit liegt. Entscheidend ist aber, **was die Entscheidung kostet**, die man aus ihr ableitet:
der **Regret** ist der Kostenunterschied zwischen der gewählten und der besten Menge bei bekannter Verteilung. Am **Newsvendor mit Merkmalen** (eine Menge bestellen,
die Nachfrage hängt von Merkmalen ab) zeigt die Demo: Wer bei ungleichen Kosten das **Mittel** bestellt, zahlt auch mit richtigem Modell und 1000 Trainingsfällen bei τ = 0,8 im Median 42,5 % und bei τ = 0,95 im Median 286,7 %
mehr als nötig (drei Merkmale, 30 feste Seeds, Experiment 2); eine Regel mit **Quantil** kostet dort unter 1 % mehr. Erstes Stück der Reihe „Entscheidungsorientiertes Lernen“.

Weiter unten: **🔬 Experiment 2 bis 4**, **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("So funktioniert Regret statt Fehler", expanded=True):
    st.markdown(
        """
1. **Kosten:** Jede fehlende Einheit kostet **b**, jede überschüssige **h**. Die beste Menge ist das **τ-Quantil** der Nachfrage mit **τ = b / (b + h)**, im Allgemeinen nicht ihr Mittel.
2. **Zweistufig:** Ein Modell schätzt aus Merkmalen und Daten die Nachfrage (Kleinste Quadrate, Prognose ŷ). Danach entscheidet eine Regel: **Mittel bestellen** (q = ŷ)
   oder **Quantil** (q = ŷ plus z · Reststreuung, z das τ-Quantil der Normalverteilung).
3. **Regret:** erwartete Kosten der gewählten Menge minus erwartete Kosten der besten Menge bei bekannter bedingter Verteilung. Die Demo nennt ihn in **Prozent Mehrkosten** gegen die beste Menge.
4. **Warum Fehler und Regret auseinanderlaufen:** Das Mittel hat den kleinsten Quadratfehler, aber bei ungleichen Kosten nicht die kleinsten Kosten. Der Regret hängt an der **Entscheidungsregel**, nicht allein an der Prognose.
        """
    )

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.markdown("**Kosten (alle Experimente)**")
    b = st.slider("Fehlmenge b je Einheit", *bounds("b_slider"), key="b_slider", help="Kosten je fehlender Einheit (Nachfrage über der Menge).")
    h = st.slider("Überschuss h je Einheit", *bounds("h_slider"), key="h_slider", help="Kosten je überschüssiger Einheit (Menge über der Nachfrage).")
    st.caption(f"Kritisches Verhältnis τ = b / (b + h) = {de(A.tau_of(b, h), 3)}")
    st.markdown("**Einzelinstanz (Experiment 2)**")
    seed = st.number_input("Zufalls-Seed der Einzelinstanz", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neue Instanz generieren", width="stretch", on_click=randomize_seed)

# --- Ergebnis: Experiment 1, eine Zelle von Hand ---------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis: Experiment 1, eine Zelle von Hand")
st.caption("Eine Zelle sind Fälle mit gleichen Merkmalen: ihre Nachfragen sind bekannt und gleich wahrscheinlich. Kosten, Mittel und beste Menge lassen sich von Hand nachrechnen.")
cell = st.selectbox("Nachfragen der Zelle", list(C.CELLS), key="cell_select", format_func=lambda k: C.CELLS[k][0])
cell_a = analyse_cell(Settings(b=int(b), h=int(h), cell=cell))
m1, m2, m3 = st.columns(3)
# Delta immer als „mein Wert minus Referenz“; die Farbe regelt delta_color="inverse" (Vorzeichen-Falle, DEMO-PLAYBOOK §2).
m1.metric("Kosten bei Menge Mittel", de(cell_a.mean_cost, 2), delta=f"{de(cell_a.mean_cost - cell_a.optimum.cost, 2)} gegen beste Menge", delta_color="inverse")
m2.metric("Kosten der besten Menge", de(cell_a.optimum.cost, 2), delta="Referenz", delta_color="off")
m3.metric("Mehrkosten des Mittels", f"{de(cell_a.excess)} %", delta="in % der besten Kosten", delta_color="off")
st.plotly_chart(build_cell(cell_a), width="stretch", key="cell_chart")
st.caption("Blau: erwartete Kosten über der Menge. Rot: Menge „Mittel“. Grün: das τ-Quantil der Zelle; die grüne Fläche zeigt alle gleich guten Mengen (bei Gleichstand eine Strecke).")
kind, text = verdict_cell(cell_a)
getattr(st, kind)(text)

st.markdown("---")

# --- Kernfrage: Experiment 2 -------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage (Experiment 2): Was kostet das Mittel, wenn das Modell perfekt ist?")
st.markdown(
    "Welt: Nachfrage D = max(0, 60 + 10·β′x + 8·z) mit z ~ N(0, 1), das lineare Modell ist **richtig spezifiziert**. Trotzdem unterscheiden sich die Entscheidungsregeln. "
    "Verglichen werden **Mittel bestellen** und **KQ plus Normalquantil** (die für diese Welt richtige Regel) gegen die wahre Quantilregel."
)
c1, c2 = st.columns(2)
p = c1.radio("Merkmale p", C.P_CHOICES, key="p_radio", horizontal=True, help="Zahl der Merkmale der Welt (die Wirkung β jedes Merkmals ist zufällig aus [−1, 1] gezogen, β auf Länge 1 normiert).")
n = c2.select_slider("Trainingsfälle n", options=C.N_CHOICES, key="n_select", help="Zahl der Trainingsfälle je Instanz; die Kennzahlen laufen über 30 feste Seeds.")
settings = Settings(b=int(b), h=int(h), cell=cell, p=int(p), n=int(n), s=float(st.session_state["s_slider"]), seed=int(seed))
tau = settings.tau
rows = _sweep(settings.p, settings.n, tau_list(tau))
st.plotly_chart(build_kern(rows), width="stretch", key="kern_chart")
st.caption(f"Balken: Median der Mehrkosten über {len(C.SWEEP_SEEDS)} feste Seeds, Fehlerbalken: Minimum bis Maximum. Die Kosten der Entscheidung werden je Testpunkt exakt gerechnet (20 000 Testpunkte je Seed).")
kind, text = verdict_kernfrage(row_for(rows, tau), settings.p, settings.n)
getattr(st, kind)(text)

with st.expander("Vormessung mit anderem Zufallsstrom (eingefroren, nicht live)"):
    st.caption("Dieselbe Welt, andere Zufallszahlen (numpy statt SplitMix64) und Kosten an gezogenen statt exakt gerechneten Nachfragen; Median [Minimum, Maximum] über 30 Seeds, Rohausgabe `out_eo_v2_news.txt`.")
    vm = []
    for r in rows:
        v = vormessung("linear", settings.p, settings.n, r["tau"])
        if v:
            vm.append({"τ": de(r["tau"], 2), "Mittel bestellen": f"{de(v['mittel'][0])} [{de(v['mittel'][1])}; {de(v['mittel'][2])}] %",
                       "KQ plus Normalquantil": f"{de(v['kq_gauss'][0])} [{de(v['kq_gauss'][1])}; {de(v['kq_gauss'][2])}] %"})
    if vm:
        st.dataframe(vm, hide_index=True, width="stretch")
    else:
        st.caption("Für dieses τ liegt keine Vormessung vor (gemessen wurden τ = 0,5, 0,8 und 0,95).")

st.markdown("**Eine Instanz im Einzelnen** (Zufalls-Seed in der Seitenleiste)")
view = _instance(settings)
i1, i2, i3 = st.columns(3)
i1.metric("Mittel bestellen", f"{de(view.excess_mean)} %", delta="Mehrkosten gegen wahre Quantilregel", delta_color="off")
i2.metric("KQ plus Normalquantil", f"{de(view.excess_quantile)} %", delta="Mehrkosten gegen wahre Quantilregel", delta_color="off")
i3.metric("Wahres Mittel bestellen", f"{de(view.excess_exact_mean)} %", delta="ohne Schätzfehler", delta_color="off")
st.plotly_chart(build_instance(view), width="stretch", key="instance_chart")
st.caption("Graue Punkte: die Trainingsfälle dieser Instanz. Rot: das wahre Mittel, grün: das wahre τ-Quantil. Das letzte Maß bestellt das exakte Mittel statt der Schätzung.")

st.markdown("---")

# --- Experiment 3 -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 3: Prognosefehler gegen Regret")
st.caption("Die Prognose wird um s Reststreuungen vom wahren Mittel weggeschoben (Normalverteilung ohne Stutzung). Der Quadratfehler hat sein Minimum bei s = 0, die Kosten bei s = z.")
s = st.slider("Verschiebung s der Prognose (in Reststreuungen)", *bounds("s_slider"), key="s_slider", step=0.1, format="%.1f",
              help="s = 0 ist die Prognose mit dem kleinsten Quadratfehler (das Mittel); s = z ist die kostenbeste Menge.")
shift_a = analyse_shift(tau, float(s))
sh1, sh2, sh3 = st.columns(3)
sh1.metric("Quadratfehler bei s", f"{de(shift_a.mse_at_s, 0)} %", delta="in % der Reststreuungs-Varianz", delta_color="off")
sh2.metric("Kosten bei s gegen beste Menge", f"{de(shift_a.excess_at_s)} %", delta="Mehrkosten", delta_color="off")
sh3.metric("Verschiebung der besten Menge z", de(shift_a.z, 2), delta="s mit den kleinsten Kosten", delta_color="off")
st.plotly_chart(build_shift(tau, float(s)), width="stretch", key="shift_chart")
kind, text = verdict_shift(shift_a)
getattr(st, kind)(text)

st.markdown("---")

# --- Experiment 4 -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Experiment 4: Lernkurve")
st.caption("Mehrkosten über der Zahl der Trainingsfälle n (30, 100, 1000) für beide Regeln; τ aus der Seitenleiste, p aus Experiment 2.")
lrows = _learning(settings.p, tau)
st.plotly_chart(build_learning(lrows), width="stretch", key="learning_chart")
kind, text = verdict_learning(lrows, tau, settings.p)
getattr(st, kind)(text)

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Eine Welt:** Normalrauschen, Stutzung bei 0, lineares Modell, konstante Streuung | Die Zahlen gelten für diese Welt; Messung mit Echtdaten fehlt. In der Vormessung (Welt „Streuung linear“, p = 3, n = 1000, τ = 0,95) kostet KQ plus Normalquantil 33,7 % mehr (eine Streuungsform, eine Messung); die Regel setzt konstante Streuung voraus. | direktes Lernen der Menge auf den Kosten (Quantilregression) |
| **Das Modell ist richtig spezifiziert** | Hier gibt es keine Fehlspezifikation; ob „KQ plus Quantil“ bei falscher Modellklasse mithält, zeigt diese Demo nicht. | Vergleich zweistufig gegen integriert bei falscher Modellklasse |
| **Wenig Daten, viele Merkmale** | Auch die Quantilregel verliert: bei p = 10, n = 30, τ = 0,95 im Median mehr als 30 % Mehrkosten (Experiment 2). | regularisierte Quantilregression |
| **Eine Menge je Merkmalszeile, ohne Kopplung** | Bei einem Löser mit Nebenbedingungen (Weg im Netz) ist die beste Entscheidung das Ergebnis eines Optimierungsproblems; die Quantilformel dieser Demo trägt dort nicht. | Entscheidungen mit Löser, etwa der kürzeste Weg im Netz |
| **Bekannte Verteilung gegen geschätzte** | `newsvendor-evpi-demo` kennt die Verteilung und fragt nach dem Wert der Information; hier ist sie unbekannt und wird aus Daten geschätzt. | – |
"""
)
st.markdown(
    "Verwandte Demos: [verweilzeit-demo](https://github.com/sebastian-hanisch/verweilzeit-demo) (derselbe Mechanismus an einem Hafen: Prognose, dann Einlagerungsregel), "
    "[forecast-inventory-demo](https://github.com/sebastian-hanisch/forecast-inventory-demo) (Sicherheitsbestand aus Prognoseunsicherheit bei Zeitreihen) und "
    "[newsvendor-evpi-demo](https://github.com/sebastian-hanisch/newsvendor-evpi-demo) (Wert der Information bei bekannter Verteilung)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Lesehilfe.** Gegeben sind Merkmale $x$, Daten $(x_i, D_i)$ und Kosten $b$, $h$; gesucht ist eine Bestellmenge $q(x)$ mit kleinem Regret.

| Symbol | Bedeutung |
|---|---|
| $D$ | Nachfrage (nicht negativ) |
| $q$ | Bestellmenge |
| $b$, $h$ | Kosten je fehlende bzw. überschüssige Einheit |
| $\tau$ | kritisches Verhältnis $b/(b+h)$ |
| $z_\tau$ | $\tau$-Quantil der Standardnormalverteilung |
| $\hat y(x)$ | Prognose (Kleinste Quadrate) |
| $\hat\sigma$ | gepoolte Reststreuung |
| $s$ | Verschiebung der Prognose in Einheiten der Reststreuung |

**Kosten und beste Menge.**

$$C(q, D) = b\,(D - q)^+ + h\,(q - D)^+, \qquad \tau = \frac{b}{b + h}$$

$$q^*(x) = \min\{\, q : F_{D \mid x}(q) \ge \tau \,\}$$

Bei Gleichstand ($F = \tau$ auf einer Strecke) sind alle Mengen der Strecke gleich gut; die Demo nennt die kleinste und zeigt die Strecke.

**Regret.**

$$\mathrm{Regret}(q) = \mathbb{E}\big[C(q(X), D)\big] - \mathbb{E}\big[C(q^*(X), D)\big], \qquad \text{Mehrkosten} = 100 \cdot \frac{\mathrm{Regret}}{\mathbb{E}[C(q^*(X), D)]}$$

**Welt.** $D = \max(0,\, 60 + 10\,\beta^\top x + 8 z)$ mit $z \sim \mathcal N(0,1)$ und $\|\beta\| = 1$. Zweistufige Regeln aus dem Kleinste-Quadrate-Modell:

$$q_{\text{Mittel}}(x) = \hat y(x), \qquad q_{\text{Quantil}}(x) = \hat y(x) + z_\tau\,\hat\sigma, \qquad \hat\sigma^2 = \frac{1}{n - p - 1}\sum_i (D_i - \hat y(x_i))^2$$

**Erwartete Kosten** bei $Y \sim \mathcal N(\mu, \sigma^2)$ ohne Stutzung, $d = (q - \mu)/\sigma$:

$$\mathbb{E}\,C(q, Y) = \sigma\Big[(b + h)\,\varphi(d) + d\,\big((b + h)\,\Phi(d) - b\big)\Big]$$

Mit Stutzung $D = \max(0, Y)$ und $q \ge 0$ kommt der Abzug $h\,\sigma\,[\varphi(d_0) + d_0\,\Phi(d_0)]$ mit $d_0 = -\mu/\sigma$ hinzu.

**Verschiebung $s$ (Experiment 3).** Die Prognose $q = \mu + s\sigma$ hat den Quadratfehler und die Kosten

$$\mathbb{E}(Y - q)^2 = \sigma^2\,(1 + s^2), \qquad \mathbb{E}\,C = \sigma\Big[(b + h)\,\varphi(s) + s\,\big((b + h)\,\Phi(s) - b\big)\Big]$$

Die Ableitung der Kosten nach $s$ ist $\sigma\,[(b + h)\Phi(s) - b]$, sie verschwindet bei $\Phi(s) = \tau$, also $s = z_\tau$. Das wahre Mittel ($s = 0$) kostet damit

$$\frac{\mathbb{E}\,C(\mu)}{\mathbb{E}\,C(\mu + z_\tau\sigma)} = \frac{\varphi(0)}{\varphi(z_\tau)} = e^{z_\tau^2/2}, \qquad \text{Mehrkosten des Mittels} = 100\,\big(e^{z_\tau^2/2} - 1\big)\ \%$$

(Normalverteilung ohne Stutzung, exaktes Mittel; unabhängig von $\mu$, $\sigma$ und, bei festem $\tau$, von $b$ und $h$.)

**Literatur.** Das kritische Verhältnis ist das klassische Ergebnis des Newsvendor-Modells. Die Zahlen dieser Demo sind Rechnungen und Messungen der Demo selbst; keine Aussage stützt sich auf eine Quelle.
        """
    )

sync_query_params({
    "b_slider": int(b), "h_slider": int(h), "cell_select": cell, "p_radio": int(p), "n_select": int(n), "s_slider": float(s), "seed_input": int(seed),
})

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
