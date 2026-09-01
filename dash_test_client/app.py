"""
app.py — Client Dash de test + comparaison de performance pour Rexobot

Ce script appelle l'API FastAPI du backend (uc202-ipn-rex) en HTTP :
    1) POST {BACKEND_URL}/hybrid            -> recherche hybride (semantic + lexical + RRF)
    2) POST {BACKEND_URL}/answer_generation  -> génération de la réponse structurée par le LLM

Il ne contient AUCUNE logique métier : c'est volontaire, pour pouvoir modifier
le backend (prompts, agents, etc.) sans jamais toucher à ce fichier.

En plus du test fonctionnel, il garde un historique de toutes les requêtes
lancées pendant la session (tokens, temps de recherche, temps de génération,
nombre de chunks utilisés...) pour pouvoir comparer objectivement deux
configurations du backend (ex: pipeline standard vs pipeline agentique).

Installation :
    pip install dash requests

Lancement :
    python app.py
    -> ouvrez http://127.0.0.1:8050

Variables d'environnement utiles :
    REXOBOT_BACKEND_URL   URL du backend FastAPI (défaut: http://localhost:5001)

Prérequis :
    Le backend doit tourner en parallèle, depuis uc202-ipn-rex :
    uv run python -m uvicorn src.api.main:app --host 0.0.0.0 --port 5001 --reload
"""

import os
import io
import json
import time
import requests
import dash
from dash import dcc, html, Input, Output, State, dash_table

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

BACKEND_URL = os.environ.get("REXOBOT_BACKEND_URL", "http://localhost:5001")

# Ces deux routers n'ont pas de prefix dans le backend actuel (voir main.py) :
# app.include_router(hybrid_search.router_hybrid)
# app.include_router(answer_generation.router_LLM)
HYBRID_SEARCH_ENDPOINT = f"{BACKEND_URL}/hybrid"
ANSWER_GENERATION_ENDPOINT = f"{BACKEND_URL}/answer_generation"

REQUEST_TIMEOUT_S = 180

# ----------------------------------------------------------------------
# App Dash
# ----------------------------------------------------------------------

app = dash.Dash(__name__)
app.title = "Rexobot — Banc de test RAG"

CARD_STYLE = {
    "border": "1px solid #ddd",
    "borderRadius": "8px",
    "padding": "16px",
    "marginBottom": "16px",
    "backgroundColor": "#fafafa",
}

STAT_CARD_STYLE = {
    "border": "1px solid #ddd",
    "borderRadius": "8px",
    "padding": "12px 16px",
    "backgroundColor": "white",
    "minWidth": "150px",
    "textAlign": "center",
}

app.layout = html.Div(
    style={"maxWidth": "1100px", "margin": "40px auto", "fontFamily": "sans-serif"},
    children=[
        html.H2("Rexobot — Banc de test RAG"),
        html.P(f"Backend ciblé : {BACKEND_URL}", style={"color": "#666"}),

        html.Div(
            style=CARD_STYLE,
            children=[
                html.Label("Question"),
                dcc.Textarea(
                    id="query-input",
                    value="Dans le cadre du projet EPR2, quelles sont les contraintes sur les produits dangereux ICPE ?",
                    style={"width": "100%", "height": "80px"},
                ),

                html.Div(
                    style={"display": "flex", "gap": "24px", "marginTop": "12px", "flexWrap": "wrap"},
                    children=[
                        html.Div([
                            html.Label("Top K"),
                            dcc.Input(id="topk-input", type="number", value=5, min=1, max=50, step=1),
                        ]),
                        html.Div([
                            html.Label("Mode de génération"),
                            dcc.Dropdown(
                                id="mode-dropdown",
                                options=[
                                    {"label": "fast", "value": "fast"},
                                    {"label": "deep", "value": "deep"},
                                ],
                                value="fast",
                                style={"width": "160px"},
                            ),
                        ]),
                        html.Div([
                            html.Label("Pipeline testé"),
                            dcc.Dropdown(
                                id="pipeline-dropdown",
                                options=[
                                    {"label": "standard (actuel)", "value": "standard"},
                                    {"label": "agentique (à venir)", "value": "agentic"},
                                ],
                                value="standard",
                                style={"width": "220px"},
                            ),
                        ]),
                        html.Div([
                            html.Label("Dictionnaire de sigles"),
                            dcc.Checklist(
                                id="use-dictionary-checklist",
                                options=[{"label": " activer", "value": "on"}],
                                value=[],
                            ),
                        ]),
                    ],
                ),

                html.Div(
                    style={"marginTop": "16px", "display": "flex", "gap": "10px"},
                    children=[
                        html.Button(
                            "Rechercher + Générer",
                            id="run-button",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "backgroundColor": "#0072ce",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                            },
                        ),
                        html.Button(
                            "Réinitialiser les statistiques",
                            id="reset-stats-button",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "backgroundColor": "#eee",
                                "border": "1px solid #ccc",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                            },
                        ),
                        html.Button(
                            "Exporter CSV",
                            id="export-csv-button",
                            n_clicks=0,
                            style={
                                "padding": "10px 20px",
                                "backgroundColor": "#eee",
                                "border": "1px solid #ccc",
                                "borderRadius": "6px",
                                "cursor": "pointer",
                            },
                        ),
                        dcc.Download(id="download-csv"),
                    ],
                ),
            ],
        ),

        dcc.Loading(
            type="circle",
            children=html.Div(id="results-container"),
        ),

        html.H3("Comparatif des exécutions (session en cours)"),
        html.Div(id="comparison-summary"),
        html.Div(
            dash_table.DataTable(
                id="history-table",
                columns=[
                    {"name": "#", "id": "run"},
                    {"name": "Heure", "id": "time"},
                    {"name": "Pipeline", "id": "pipeline"},
                    {"name": "Mode", "id": "mode"},
                    {"name": "Top K", "id": "top_k"},
                    {"name": "Chunks reçus", "id": "n_chunks"},
                    {"name": "Refs citées", "id": "n_refs_cited"},
                    {"name": "Couverture refs (%)", "id": "ref_coverage_pct"},
                    {"name": "Tokens entrée", "id": "tokens_in"},
                    {"name": "Tokens sortie", "id": "tokens_out"},
                    {"name": "Tokens totaux", "id": "tokens_total"},
                    {"name": "Temps recherche (s)", "id": "search_time_s"},
                    {"name": "Temps génération (s)", "id": "generation_time_s"},
                    {"name": "Temps total (s)", "id": "total_time_s"},
                ],
                data=[],
                style_table={"overflowX": "auto"},
                style_cell={"fontFamily": "sans-serif", "fontSize": "13px", "padding": "6px"},
                style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
                page_size=10,
                sort_action="native",
            ),
            style={"marginTop": "12px"},
        ),

        dcc.Store(id="run-history", data=[]),

        html.Details(
            style={"marginTop": "24px"},
            children=[
                html.Summary("JSON brut (debug)"),
                html.Pre(id="raw-json-output", style={
                    "whiteSpace": "pre-wrap",
                    "backgroundColor": "#1e1e1e",
                    "color": "#d4d4d4",
                    "padding": "12px",
                    "borderRadius": "6px",
                    "maxHeight": "400px",
                    "overflowY": "auto",
                }),
            ],
        ),
    ],
)


# ----------------------------------------------------------------------
# Callback principal : lance la recherche + génération, met à jour l'historique
# ----------------------------------------------------------------------

@app.callback(
    Output("results-container", "children"),
    Output("raw-json-output", "children"),
    Output("run-history", "data"),
    Input("run-button", "n_clicks"),
    Input("reset-stats-button", "n_clicks"),
    State("query-input", "value"),
    State("topk-input", "value"),
    State("mode-dropdown", "value"),
    State("pipeline-dropdown", "value"),
    State("use-dictionary-checklist", "value"),
    State("run-history", "data"),
    prevent_initial_call=True,
)
def run_pipeline(n_clicks_run, n_clicks_reset, query, top_k, mode, pipeline, use_dictionary_value, history):
    triggered_id = dash.callback_context.triggered[0]["prop_id"].split(".")[0]
    history = history or []

    if triggered_id == "reset-stats-button":
        return html.Div("Statistiques réinitialisées."), "", []

    if not query or not query.strip():
        return html.Div("Merci de saisir une question.", style={"color": "red"}), "", history

    use_dictionary = "on" in (use_dictionary_value or [])
    debug_log = {}
    t_total_start = time.perf_counter()

    # --- 1) Recherche hybride ---
    try:
        search_resp = requests.post(
            HYBRID_SEARCH_ENDPOINT,
            json={
                "query": query,
                "top_k": int(top_k or 5),
                "use_dictionary": use_dictionary,
                "filters": None,
            },
            timeout=REQUEST_TIMEOUT_S,
        )
        search_resp.raise_for_status()
        search_data = search_resp.json()
    except Exception as e:
        return (
            html.Div(f"Erreur lors de l'appel à {HYBRID_SEARCH_ENDPOINT} : {e}", style={"color": "red"}),
            "",
            history,
        )

    if "error" in search_data:
        return (
            html.Div(f"Erreur backend (recherche) : {search_data['error']}", style={"color": "red"}),
            json.dumps(search_data, ensure_ascii=False, indent=2),
            history,
        )

    chunks = search_data.get("chunks", [])
    search_timings = search_data.get("timings", {}) or {}
    # timings renvoyés par combined_search : semantic_ms, encoding_ms, lexical_ms
    search_time_s = sum(v for v in search_timings.values() if isinstance(v, (int, float))) / 1000.0
    debug_log["search"] = search_data

    if not chunks:
        elapsed = time.perf_counter() - t_total_start
        history = _append_history_row(
            history, pipeline, mode, top_k, n_chunks=0, n_refs_cited=0,
            tokens_in=0, tokens_out=0, search_time_s=search_time_s,
            generation_time_s=0.0, total_time_s=elapsed,
        )
        return (
            html.Div("Aucun chunk trouvé pour cette question.", style={"color": "#b8860b"}),
            json.dumps(debug_log, ensure_ascii=False, indent=2),
            history,
        )

    # --- 2) Génération de la réponse ---
    # Le champ "pipeline" est envoyé au backend pour anticiper un futur aiguillage
    # standard/agentique côté serveur. Tant que le backend ne le lit pas, il est
    # simplement ignoré (payload accepté en Dict[str, Any]).
    try:
        gen_resp = requests.post(
            ANSWER_GENERATION_ENDPOINT,
            json={"query": query, "chunks": chunks, "mode": mode, "pipeline": pipeline},
            timeout=REQUEST_TIMEOUT_S,
        )
        gen_resp.raise_for_status()
        gen_data = gen_resp.json()
    except Exception as e:
        elapsed = time.perf_counter() - t_total_start
        history = _append_history_row(
            history, pipeline, mode, top_k, n_chunks=len(chunks), n_refs_cited=0,
            tokens_in=0, tokens_out=0, search_time_s=search_time_s,
            generation_time_s=0.0, total_time_s=elapsed,
        )
        return (
            html.Div(f"Erreur lors de l'appel à {ANSWER_GENERATION_ENDPOINT} : {e}", style={"color": "red"}),
            json.dumps(debug_log, ensure_ascii=False, indent=2),
            history,
        )

    if "error" in gen_data:
        elapsed = time.perf_counter() - t_total_start
        history = _append_history_row(
            history, pipeline, mode, top_k, n_chunks=len(chunks), n_refs_cited=0,
            tokens_in=0, tokens_out=0, search_time_s=search_time_s,
            generation_time_s=0.0, total_time_s=elapsed,
        )
        return (
            html.Div(f"Erreur backend (génération) : {gen_data['error']}", style={"color": "red"}),
            json.dumps({**debug_log, "generation": gen_data}, ensure_ascii=False, indent=2),
            history,
        )

    debug_log["generation"] = gen_data
    answer = gen_data.get("answer", [])
    tokens_in = gen_data.get("num_tokens_entry", 0) or 0
    tokens_out = gen_data.get("num_tokens_output", 0) or 0
    generation_time_s = (gen_data.get("timings", {}) or {}).get("portail_ms", 0) / 1000.0

    n_refs_cited = _count_unique_references(answer)
    total_time_s = time.perf_counter() - t_total_start

    history = _append_history_row(
        history, pipeline, mode, top_k, n_chunks=len(chunks), n_refs_cited=n_refs_cited,
        tokens_in=tokens_in, tokens_out=tokens_out, search_time_s=search_time_s,
        generation_time_s=generation_time_s, total_time_s=total_time_s,
    )

    # --- Rendu ---
    answer_blocks = render_answer(answer)
    chunk_blocks = render_chunks(chunks)

    stat_cards = html.Div(
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "16px"},
        children=[
            _stat_card("Temps recherche", f"{search_time_s:.2f} s"),
            _stat_card("Temps génération", f"{generation_time_s:.2f} s"),
            _stat_card("Temps total", f"{total_time_s:.2f} s"),
            _stat_card("Tokens entrée", str(tokens_in)),
            _stat_card("Tokens sortie", str(tokens_out)),
            _stat_card("Tokens totaux", str(tokens_in + tokens_out)),
            _stat_card("Chunks reçus", str(len(chunks))),
            _stat_card("Refs citées", f"{n_refs_cited} / {len(chunks)}"),
        ],
    )

    result_view = html.Div([
        html.H3("Réponse générée"),
        stat_cards,
        html.Div(answer_blocks, style=CARD_STYLE),
        html.H4(f"Chunks utilisés pour la recherche ({len(chunks)})"),
        html.Div(chunk_blocks),
    ])

    return result_view, json.dumps(debug_log, ensure_ascii=False, indent=2), history


# ----------------------------------------------------------------------
# Callback : met à jour le tableau + le résumé comparatif à partir de l'historique
# ----------------------------------------------------------------------

@app.callback(
    Output("history-table", "data"),
    Output("comparison-summary", "children"),
    Input("run-history", "data"),
)
def update_history_view(history):
    history = history or []
    if not history:
        return [], html.P("Aucune exécution enregistrée pour l'instant.", style={"color": "#666"})

    summary = _build_comparison_summary(history)
    return history, summary


@app.callback(
    Output("download-csv", "data"),
    Input("export-csv-button", "n_clicks"),
    State("run-history", "data"),
    prevent_initial_call=True,
)
def export_csv(n_clicks, history):
    history = history or []
    if not history:
        return None

    columns = [
        "run", "time", "pipeline", "mode", "top_k", "n_chunks", "n_refs_cited",
        "ref_coverage_pct", "tokens_in", "tokens_out", "tokens_total",
        "search_time_s", "generation_time_s", "total_time_s",
    ]
    buf = io.StringIO()
    buf.write(",".join(columns) + "\n")
    for row in history:
        buf.write(",".join(str(row.get(c, "")) for c in columns) + "\n")

    return dict(content=buf.getvalue(), filename="rexobot_stats.csv")


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _stat_card(label, value):
    return html.Div([
        html.Div(value, style={"fontSize": "20px", "fontWeight": "bold"}),
        html.Div(label, style={"fontSize": "12px", "color": "#666"}),
    ], style=STAT_CARD_STYLE)


def _count_unique_references(answer):
    """Compte le nombre de chunk_id distincts effectivement cités dans les références."""
    refs = set()
    for theme_block in answer or []:
        for sub in theme_block.get("subthemes", []) or []:
            for r in sub.get("references", []) or []:
                refs.add(r)
    return len(refs)


def _append_history_row(history, pipeline, mode, top_k, n_chunks, n_refs_cited,
                         tokens_in, tokens_out, search_time_s, generation_time_s, total_time_s):
    ref_coverage_pct = round(100 * n_refs_cited / n_chunks, 1) if n_chunks else 0.0
    row = {
        "run": len(history) + 1,
        "time": time.strftime("%H:%M:%S"),
        "pipeline": pipeline,
        "mode": mode,
        "top_k": top_k,
        "n_chunks": n_chunks,
        "n_refs_cited": n_refs_cited,
        "ref_coverage_pct": ref_coverage_pct,
        "tokens_in": tokens_in,
        "tokens_out": tokens_out,
        "tokens_total": tokens_in + tokens_out,
        "search_time_s": round(search_time_s, 2),
        "generation_time_s": round(generation_time_s, 2),
        "total_time_s": round(total_time_s, 2),
    }
    return history + [row]


def _build_comparison_summary(history):
    """Regroupe les runs par pipeline (standard/agentic) et affiche les moyennes clés."""
    groups = {}
    for row in history:
        groups.setdefault(row.get("pipeline", "standard"), []).append(row)

    def _avg(rows, key):
        vals = [r.get(key, 0) for r in rows if isinstance(r.get(key, 0), (int, float))]
        return sum(vals) / len(vals) if vals else 0.0

    cards = []
    for pipeline_name, rows in groups.items():
        cards.append(html.Div([
            html.H4(f"Pipeline : {pipeline_name} ({len(rows)} exécution(s))"),
            html.Div(
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
                children=[
                    _stat_card("Tokens totaux (moy.)", f"{_avg(rows, 'tokens_total'):.0f}"),
                    _stat_card("Temps recherche (moy.)", f"{_avg(rows, 'search_time_s'):.2f} s"),
                    _stat_card("Temps génération (moy.)", f"{_avg(rows, 'generation_time_s'):.2f} s"),
                    _stat_card("Temps total (moy.)", f"{_avg(rows, 'total_time_s'):.2f} s"),
                    _stat_card("Couverture refs (moy.)", f"{_avg(rows, 'ref_coverage_pct'):.1f} %"),
                ],
            ),
        ], style=CARD_STYLE))

    return html.Div(cards)


def render_answer(answer):
    """answer: liste de {theme, subthemes: [{subtheme, content, references}]}"""
    if not answer:
        return html.P("Réponse vide.")

    blocks = []
    for theme_block in answer:
        theme = theme_block.get("theme", "")
        subthemes = theme_block.get("subthemes", [])
        sub_blocks = []
        for sub in subthemes:
            sub_blocks.append(html.Div([
                html.H5(sub.get("subtheme", "")),
                html.P(sub.get("content", "")),
                html.P(
                    "Références : " + ", ".join(sub.get("references", [])) if sub.get("references") else "Références : aucune",
                    style={"fontSize": "12px", "color": "#888"},
                ),
            ], style={"marginBottom": "12px", "paddingLeft": "12px", "borderLeft": "3px solid #0072ce"}))
        blocks.append(html.Div([html.H4(theme), *sub_blocks], style={"marginBottom": "20px"}))

    return blocks


def render_chunks(chunks):
    blocks = []
    for c in chunks:
        content = c.get("chunk_content") or ""
        preview = content[:300]
        blocks.append(html.Div([
            html.P(f"score={c.get('score'):.3f} | source={c.get('source')} | chunk_id={c.get('chunk_id')}",
                   style={"fontSize": "12px", "color": "#888", "marginBottom": "4px"}),
            html.P(preview + ("…" if len(content) > 300 else "")),
        ], style=CARD_STYLE))
    return blocks


if __name__ == "__main__":
    app.run(debug=True, port=8050) 