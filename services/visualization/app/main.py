import os
from pathlib import Path

import plotly.express as px
import requests
from dash import Dash, Input, Output, State, callback, dash_table, dcc, html, no_update

WEB_MASTER_URL = os.getenv("WEB_MASTER_URL", "http://localhost:18000").rstrip("/")
TIMEOUT = 300
session = requests.Session()

app = Dash(
    __name__,
    title="Research Gap Explorer",
    suppress_callback_exceptions=True,
    assets_folder=str(Path(__file__).parent / "assets"),
)
server = app.server


def api(method: str, path: str, **kwargs) -> dict:
    response = session.request(method, f"{WEB_MASTER_URL}{path}", timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    return response.json()


def error_message(error: requests.RequestException) -> str:
    if error.response is not None:
        try:
            return str(error.response.json().get("detail", error.response.text))
        except ValueError:
            return error.response.text
    return str(error)


def back_link() -> dcc.Link:
    return dcc.Link("← На главную", href="/", className="back-link")


def page_title(title: str, subtitle: str) -> html.Div:
    return html.Div(
        [back_link(), html.H1(title), html.P(subtitle, className="page-subtitle")],
        className="page-heading",
    )


def empty_state(title: str, text: str) -> html.Div:
    return html.Div(
        [html.Div("∅", className="empty-icon"), html.H3(title), html.P(text)],
        className="empty-state",
    )


def status_pill(text: str, kind: str = "neutral") -> html.Span:
    return html.Span(text, className=f"status-pill {kind}")


def home_page() -> html.Main:
    cards = [
        ("/import", "01", "Импорт данных", "Загрузить подготовленный корпус публикаций", "violet"),
        ("/visualizations", "02", "Способы визуализации", "Обучить визуализаторы", "cyan"),
        ("/models", "03", "Модели", "Обучить и сравнить кластеризаторы", "amber"),
        ("/results", "04", "Результаты", "Исследовать карту, темы и research gaps", "rose"),
    ]
    return html.Main(
        [
            html.Section(
                [
                    html.Div("RESEARCH INTELLIGENCE PLATFORM", className="eyebrow"),
                    html.H1(["Research Gap", html.Br(), html.Span("Explorer")]),
                    html.P(
                        "Интерперируемый поиск неизведанных полянок."
                    ),
                ],
                className="hero",
            ),
            html.Section(
                [
                    dcc.Link(
                        html.Article(
                            [
                                html.Div(number, className="card-number"),
                                html.Div("↗", className="card-arrow"),
                                html.H2(title),
                                html.P(description),
                            ],
                            className=f"nav-card {color}",
                        ),
                        href=href,
                        className="card-link",
                    )
                    for href, number, title, description, color in cards
                ],
                className="nav-grid",
            ),
        ],
        className="home-page page-enter",
    )


def import_page() -> html.Main:
    return html.Main(
        [
            dcc.Interval(id="import-refresh", interval=10_000, n_intervals=0),
            page_title("Импорт данных", "Подготовленный корпус CSV/NPY будет проверен и сохранён в PostgreSQL."),
            html.Section(
                [
                    html.Div(id="import-dataset-status", className="dataset-status"),
                    dcc.Loading(
                        html.Div(id="import-action-message", className="action-message"),
                        type="circle",
                        color="#8b5cf6",
                    ),
                    html.Button("Импортировать данные", id="import-page-button", className="primary-button"),
                ],
                className="center-card",
            ),
            html.Div(id="import-footnote", className="success-footnote"),
        ],
        className="content-page page-enter",
    )


PROJECTION_COLUMNS = [
    {"name": "ID", "id": "short_id"},
    {"name": "Метод", "id": "algorithm"},
    {"name": "Статус", "id": "status"},
    {"name": "Параметры", "id": "parameters"},
    {"name": "Создана", "id": "created_at"},
]


def visualizations_page() -> html.Main:
    return html.Main(
        [
            dcc.Store(id="projection-job-trigger"),
            dcc.Interval(id="projection-refresh", interval=1_500, n_intervals=0),
            page_title("Способы визуализации", "Проекция строится один раз и может использоваться многими моделями."),
            html.Div(id="visualization-data-warning"),
            html.Div(id="projection-running-state"),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Новая визуализация"),
                            html.Label("Тип визуализации"),
                            dcc.Dropdown(
                                id="viz-algorithm",
                                options=[
                                    {"label": "PCA — быстро и линейно", "value": "pca"},
                                    {"label": "UMAP — локальная структура", "value": "umap"},
                                ],
                                value="pca",
                                clearable=False,
                                className="dark-dropdown",
                            ),
                            html.Div(
                                [
                                    html.Label("Whitening"),
                                    dcc.Checklist(
                                        id="pca-whiten",
                                        options=[{"label": "Нормировать дисперсию компонент", "value": "yes"}],
                                        value=[],
                                        className="checklist",
                                    ),
                                ],
                                id="pca-parameters",
                                className="parameter-group",
                            ),
                            html.Div(
                                [
                                    html.Div([html.Label("n_neighbors"), dcc.Input(id="umap-neighbors", type="number", value=15, min=2, max=200)], className="field"),
                                    html.Div([html.Label("min_dist"), dcc.Input(id="umap-min-dist", type="number", value=0.1, min=0, max=0.99, step=0.05)], className="field"),
                                    html.Div(
                                        [
                                            html.Label("metric"),
                                            dcc.Dropdown(
                                                id="umap-metric",
                                                options=[{"label": "Cosine", "value": "cosine"}, {"label": "Euclidean", "value": "euclidean"}],
                                                value="cosine",
                                                clearable=False,
                                                className="dark-dropdown",
                                            ),
                                        ],
                                        className="field",
                                    ),
                                ],
                                id="umap-parameters",
                                className="parameter-grid",
                            ),
                            html.Div([html.Label("random_state"), dcc.Input(id="viz-random-state", type="number", value=42)], className="field"),
                            html.Button("Добавить визуализацию", id="add-projection-button", className="primary-button"),
                            dcc.Loading(html.Div(id="projection-action-message", className="action-message"), type="circle"),
                        ],
                        className="form-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.H2("Обученные визуализации"), html.Button("Удалить выбранные", id="delete-projections-button", className="danger-button")],
                                className="section-toolbar",
                            ),
                            dash_table.DataTable(
                                id="projection-table",
                                columns=PROJECTION_COLUMNS,
                                data=[],
                                row_selectable="multi",
                                sort_action="native",
                                page_size=8,
                                style_table={"overflowX": "auto"},
                                style_header={"textAlign": "center"},
                            ),
                            html.Div(id="projection-delete-message", className="action-message"),
                        ],
                        className="table-card",
                    ),
                ],
                className="split-layout",
            ),
        ],
        className="content-page page-enter",
    )


MODEL_COLUMNS = [
    {"name": "ID", "id": "short_id"},
    {"name": "Алгоритм", "id": "algorithm"},
    {"name": "Кластеры", "id": "clusters", "type": "numeric"},
    {"name": "Silhouette ↑", "id": "silhouette", "type": "numeric"},
    {"name": "Davies–Bouldin ↓", "id": "davies_bouldin", "type": "numeric"},
    {"name": "Calinski–Harabasz ↑", "id": "calinski_harabasz", "type": "numeric"},
    {"name": "Inertia ↓", "id": "inertia", "type": "numeric"},
    {"name": "Статус", "id": "status"},
]


def models_page() -> html.Main:
    return html.Main(
        [
            dcc.Store(id="model-job-trigger"),
            dcc.Interval(id="model-refresh", interval=1_500, n_intervals=0),
            page_title("Модели", "Обучайте кластеризаторы и сравнивайте их по единому набору метрик."),
            html.Div(id="model-data-warning"),
            html.Div(id="model-running-state"),
            html.Section(
                [
                    html.Div(
                        [
                            html.H2("Новая модель"),
                            html.Label("Алгоритм кластеризации"),
                            dcc.Dropdown(
                                id="model-algorithm",
                                options=[{"label": "MiniBatchKMeans", "value": "minibatch_kmeans"}, {"label": "KMeans", "value": "kmeans"}],
                                value="minibatch_kmeans",
                                clearable=False,
                                className="dark-dropdown",
                            ),
                            html.Div(
                                [
                                    html.Div([html.Label("Число кластеров"), dcc.Input(id="model-clusters", type="number", value=12, min=2, max=100)], className="field"),
                                    html.Div([html.Label("n_init"), dcc.Input(id="model-n-init", type="number", value=5, min=1, max=50)], className="field"),
                                    html.Div([html.Label("max_iter"), dcc.Input(id="model-max-iter", type="number", value=300, min=10, max=2000)], className="field"),
                                    html.Div([html.Label("batch_size"), dcc.Input(id="model-batch-size", type="number", value=256, min=16, max=4096)], id="batch-size-field", className="field"),
                                    html.Div([html.Label("random_state"), dcc.Input(id="model-random-state", type="number", value=42)], className="field"),
                                ],
                                className="parameter-grid",
                            ),
                            html.Button("Добавить модель", id="add-model-button", className="primary-button"),
                            dcc.Loading(html.Div(id="model-action-message", className="action-message"), type="circle"),
                        ],
                        className="form-card",
                    ),
                    html.Div(
                        [
                            html.Div(
                                [html.H2("Бенчмарк моделей"), html.Button("Удалить выбранные", id="delete-models-button", className="danger-button")],
                                className="section-toolbar",
                            ),
                            html.P("Нажмите на заголовок метрики для сортировки.", className="muted"),
                            dash_table.DataTable(
                                id="model-table",
                                columns=MODEL_COLUMNS,
                                data=[],
                                row_selectable="multi",
                                sort_action="native",
                                sort_mode="multi",
                                page_size=10,
                                style_table={"overflowX": "auto"},
                                style_header={"textAlign": "center"},
                            ),
                            html.Div(id="model-delete-message", className="action-message"),
                        ],
                        className="table-card wide-table",
                    ),
                ],
                className="split-layout model-layout",
            ),
        ],
        className="content-page page-enter",
    )


def results_page() -> html.Main:
    return html.Main(
        [
            dcc.Interval(id="results-refresh", interval=750, n_intervals=0, max_intervals=1),
            page_title("Результаты", "Совместите модель и визуализацию, чтобы исследовать тематическую структуру."),
            html.Div(id="results-warning"),
            html.Section(
                [
                    html.Div([html.Label("Модель"), dcc.Dropdown(id="result-model-select", placeholder="Выберите модель", className="dark-dropdown")], className="selector-field"),
                    html.Div([html.Label("Визуализация"), dcc.Dropdown(id="result-projection-select", placeholder="Выберите проекцию", className="dark-dropdown")], className="selector-field"),
                    html.Div([html.Label("Год от"), dcc.Input(id="result-year-from", type="number", value=2018)], className="field compact"),
                    html.Div([html.Label("Год до"), dcc.Input(id="result-year-to", type="number", value=2026)], className="field compact"),
                ],
                className="results-toolbar",
            ),
            dcc.Loading(html.Div(id="results-content"), type="circle", color="#22d3ee"),
        ],
        className="content-page page-enter",
    )


def not_found_page() -> html.Main:
    return html.Main([back_link(), empty_state("Страница не найдена", "Вернитесь на главную страницу.")], className="content-page")


app.layout = html.Div(
    [dcc.Location(id="url", refresh=False), html.Div(id="page-content")],
    className="app-shell",
)


@callback(Output("page-content", "children"), Input("url", "pathname"))
def route(pathname):
    pages = {"/": home_page, "/import": import_page, "/visualizations": visualizations_page, "/models": models_page, "/results": results_page}
    return pages.get(pathname, not_found_page)()


@callback(
    Output("import-action-message", "children"),
    Input("import-page-button", "n_clicks"),
    prevent_initial_call=True,
    running=[(Output("import-page-button", "disabled"), True, False)],
)
def import_dataset(_clicks):
    try:
        result = api("POST", "/api/imports")
        return html.Div([html.Span("✓", className="success-icon"), f" Данные успешно импортированы: {result['imported']} публикаций."], className="success-message")
    except requests.RequestException as error:
        return html.Div(f"Ошибка импорта: {error_message(error)}", className="error-message")


@callback(
    Output("import-dataset-status", "children"),
    Output("import-footnote", "children"),
    Input("import-refresh", "n_intervals"),
)
def refresh_import_status(_tick):
    try:
        count = api("GET", "/api/dataset")["stored_papers"]
    except requests.RequestException:
        return status_pill("Storage недоступен", "error"), ""
    if count:
        return status_pill(f"В базе: {count} публикаций", "success"), "Данные успешно импортированы и готовы к анализу."
    return status_pill("База пока пуста"), "После импорта здесь появится подтверждение."


@callback(Output("pca-parameters", "style"), Output("umap-parameters", "style"), Input("viz-algorithm", "value"))
def toggle_projection_parameters(algorithm):
    return ({"display": "block"}, {"display": "none"}) if algorithm == "pca" else ({"display": "none"}, {"display": "grid"})


@callback(
    Output("projection-action-message", "children"),
    Output("projection-job-trigger", "data"),
    Input("add-projection-button", "n_clicks"),
    State("viz-algorithm", "value"),
    State("pca-whiten", "value"),
    State("umap-neighbors", "value"),
    State("umap-min-dist", "value"),
    State("umap-metric", "value"),
    State("viz-random-state", "value"),
    prevent_initial_call=True,
    running=[
        (Output("add-projection-button", "disabled"), True, False),
        (Output("add-projection-button", "children"), "Запускаем…", "Добавить визуализацию"),
    ],
)
def create_projection(_clicks, algorithm, whiten, neighbors, min_dist, metric, random_state):
    payload = {
        "algorithm": algorithm,
        "whiten": "yes" in (whiten or []),
        "n_neighbors": int(neighbors or 15),
        "min_dist": float(min_dist if min_dist is not None else 0.1),
        "metric": metric or "cosine",
        "random_state": int(random_state or 42),
    }
    try:
        result = api("POST", "/api/projections", json=payload)
        return f"Визуализация запущена: {result['projection_id'][:8]}.", result
    except requests.RequestException as error:
        return html.Div(f"Ошибка: {error_message(error)}", className="error-message"), None


@callback(
    Output("visualization-data-warning", "children"),
    Output("projection-running-state", "children"),
    Output("projection-table", "data"),
    Output("add-projection-button", "disabled"),
    Output("projection-refresh", "disabled"),
    Input("projection-refresh", "n_intervals"),
    Input("projection-job-trigger", "data"),
)
def refresh_projections(_tick, _job):
    try:
        no_data = api("GET", "/api/dataset")["stored_papers"] == 0
        projections = api("GET", "/api/projections")["items"]
    except requests.RequestException as error:
        return html.Div(f"API недоступен: {error_message(error)}", className="error-message"), "", [], True, True
    running = any(item["status"] == "running" for item in projections)
    warning = empty_state("Данных пока нет", "Сначала перейдите в раздел «Импорт данных».") if no_data else ""
    running_state = html.Div([html.Span(className="spinner"), " Визуализация обучается…"], className="running-message") if running else ""
    rows = [
        {
            "projection_id": item["projection_id"],
            "short_id": item["projection_id"][:8],
            "algorithm": item["algorithm"].upper(),
            "status": item["status"],
            "parameters": ", ".join(f"{key}={value}" for key, value in item["parameters"].items()),
            "created_at": str(item["created_at"])[:19],
        }
        for item in projections
    ]
    return warning, running_state, rows, no_data or running, not running


@callback(
    Output("projection-delete-message", "children"),
    Input("delete-projections-button", "n_clicks"),
    State("projection-table", "data"),
    State("projection-table", "selected_rows"),
    prevent_initial_call=True,
)
def delete_projections(_clicks, rows, selected):
    if not selected:
        return "Выберите хотя бы одну визуализацию."
    deleted, errors = 0, []
    for index in selected:
        try:
            api("DELETE", f"/api/projections/{rows[index]['projection_id']}")
            deleted += 1
        except requests.RequestException as error:
            errors.append(error_message(error))
    text = f"Удалено визуализаций: {deleted}."
    return html.Div([text, html.Br(), "; ".join(errors)] if errors else text, className="error-message" if errors else "success-message")


@callback(Output("batch-size-field", "style"), Input("model-algorithm", "value"))
def toggle_batch_size(algorithm):
    return {"display": "block"} if algorithm == "minibatch_kmeans" else {"display": "none"}


@callback(
    Output("model-action-message", "children"),
    Output("model-job-trigger", "data"),
    Input("add-model-button", "n_clicks"),
    State("model-algorithm", "value"),
    State("model-clusters", "value"),
    State("model-n-init", "value"),
    State("model-max-iter", "value"),
    State("model-batch-size", "value"),
    State("model-random-state", "value"),
    prevent_initial_call=True,
    running=[
        (Output("add-model-button", "disabled"), True, False),
        (Output("add-model-button", "children"), "Модель обучается…", "Добавить модель"),
    ],
)
def create_model(_clicks, algorithm, clusters, n_init, max_iter, batch_size, random_state):
    payload = {
        "algorithm": algorithm,
        "clusters": int(clusters or 12),
        "n_init": int(n_init or 5),
        "max_iter": int(max_iter or 300),
        "batch_size": int(batch_size or 256),
        "random_state": int(random_state or 42),
    }
    try:
        result = api("POST", "/api/models/train", json=payload)
        return f"Модель поставлена в очередь: {result['model_id'][:8]}.", result
    except requests.RequestException as error:
        return html.Div(f"Ошибка: {error_message(error)}", className="error-message"), None


@callback(
    Output("model-data-warning", "children"),
    Output("model-running-state", "children"),
    Output("model-table", "data"),
    Output("add-model-button", "disabled"),
    Output("model-refresh", "disabled"),
    Input("model-refresh", "n_intervals"),
    Input("model-job-trigger", "data"),
)
def refresh_models(_tick, _job):
    try:
        no_data = api("GET", "/api/dataset")["stored_papers"] == 0
        models = api("GET", "/api/models")["items"]
    except requests.RequestException as error:
        return html.Div(f"API недоступен: {error_message(error)}", className="error-message"), "", [], True, True
    running = any(item["status"] == "running" for item in models)
    warning = empty_state("Данных пока нет", "Сначала импортируйте данные.") if no_data else ""
    running_state = html.Div([html.Span(className="spinner"), " Модель обучается…"], className="running-message") if running else ""
    rows = []
    for model in models:
        metrics = model.get("metrics") or {}
        rows.append(
            {
                "model_id": model["model_id"],
                "short_id": model["model_id"][:8],
                "algorithm": model["algorithm"],
                "clusters": model["clusters"],
                "silhouette": metrics.get("silhouette_score"),
                "davies_bouldin": metrics.get("davies_bouldin_index"),
                "calinski_harabasz": metrics.get("calinski_harabasz_index"),
                "inertia": metrics.get("inertia"),
                "status": model["status"],
            }
        )
    return warning, running_state, rows, no_data or running, not running


@callback(
    Output("model-delete-message", "children"),
    Input("delete-models-button", "n_clicks"),
    State("model-table", "data"),
    State("model-table", "selected_rows"),
    prevent_initial_call=True,
)
def delete_models(_clicks, rows, selected):
    if not selected:
        return "Выберите хотя бы одну модель."
    deleted, errors = 0, []
    for index in selected:
        try:
            api("DELETE", f"/api/models/{rows[index]['model_id']}")
            deleted += 1
        except requests.RequestException as error:
            errors.append(error_message(error))
    text = f"Удалено моделей: {deleted}."
    return html.Div([text, html.Br(), "; ".join(errors)] if errors else text, className="error-message" if errors else "success-message")


@callback(
    Output("results-warning", "children"),
    Output("result-model-select", "options"),
    Output("result-model-select", "value"),
    Output("result-projection-select", "options"),
    Output("result-projection-select", "value"),
    Input("results-refresh", "n_intervals"),
    State("result-model-select", "value"),
    State("result-projection-select", "value"),
)
def refresh_result_selectors(_tick, selected_model, selected_projection):
    try:
        dataset = api("GET", "/api/dataset")
        models = api("GET", "/api/models")["items"]
        projections = api("GET", "/api/projections")["items"]
    except requests.RequestException as error:
        return html.Div(f"API недоступен: {error_message(error)}", className="error-message"), [], None, [], None
    current_version = dataset["dataset_version"]
    completed_models = [
        item for item in models
        if item["status"] == "completed" and item.get("dataset_version") == current_version
    ]
    completed_projections = [
        item for item in projections
        if item["status"] == "completed" and item.get("dataset_version") == current_version
    ]
    warning = "" if dataset["stored_papers"] and completed_models and completed_projections else empty_state("Результатов пока нет", "Импортируйте данные, независимо постройте визуализацию и обучите модель кластеризации для текущей версии корпуса.")
    model_options = [{"label": f"{item['algorithm']} · {item['clusters']} кластеров · {item['model_id'][:8]}", "value": item["model_id"]} for item in completed_models]
    projection_options = [{"label": f"{item['algorithm'].upper()} · {item['projection_id'][:8]}", "value": item["projection_id"]} for item in completed_projections]
    model_ids = [item["model_id"] for item in completed_models]
    projection_ids = [item["projection_id"] for item in completed_projections]
    model_value = no_update
    projection_value = no_update
    if selected_model is not None and selected_model not in model_ids:
        model_value = None
    if selected_projection is not None and selected_projection not in projection_ids:
        projection_value = None
    return warning, model_options, model_value, projection_options, projection_value


def metrics_view(metrics: dict) -> html.Div:
    values = [
        ("Silhouette", metrics.get("silhouette_score"), "выше — лучше"),
        ("Davies–Bouldin", metrics.get("davies_bouldin_index"), "ниже — лучше"),
        ("Calinski–Harabasz", metrics.get("calinski_harabasz_index"), "выше — лучше"),
        ("Inertia", metrics.get("inertia"), "ниже — лучше"),
    ]
    return html.Div(
        [html.Div([html.Span(label), html.Strong("—" if value is None else f"{value:.4f}"), html.Small(hint)], className="metric-card") for label, value, hint in values],
        className="metrics-grid",
    )


@callback(
    Output("results-content", "children"),
    Input("result-model-select", "value"),
    Input("result-projection-select", "value"),
    Input("result-year-from", "value"),
    Input("result-year-to", "value"),
)
def render_results(model_id, projection_id, year_from, year_to):
    if not model_id or not projection_id:
        return ""
    try:
        model = api("GET", f"/api/models/{model_id}")
        points = api("GET", f"/api/map/{model_id}", params={"projection_id": projection_id, "year_from": year_from, "year_to": year_to})["items"]
        topics = api("GET", f"/api/models/{model_id}/topics")["items"]
        gaps = api("GET", f"/api/models/{model_id}/research-gaps")["items"][:5]
    except requests.RequestException as error:
        return html.Div(f"Не удалось загрузить результаты: {error_message(error)}", className="error-message")
    if not points:
        return empty_state("Нет точек", "Для выбранной комбинации и диапазона годов данные не найдены.")

    scatter = px.scatter(
        points, x="x", y="y", color="topic_name", hover_name="title",
        hover_data=["year", "primary_topic", "cited_by_count", "cluster_id"],
        title=f"Тематическая карта · {len(points)} публикаций", template="plotly_dark",
    )
    scatter.update_traces(marker={"size": 8, "opacity": 0.78})
    scatter.update_layout(height=680, legend_title_text="Тематика")
    trend_rows = [{"year": int(year), "publications": count, "topic": topic["name"]} for topic in topics for year, count in topic["yearly_counts"].items()]
    trends = px.line(trend_rows, x="year", y="publications", color="topic", markers=True, title="Развитие тематик во времени", template="plotly_dark")
    trends.update_layout(height=480, legend_title_text="Тематика")
    topic_rows = [
        {"cluster": item["cluster_id"], "name": item["name"], "keywords": ", ".join(item["keywords"]), "papers": item["size"], "share": f"{item['share'] * 100:.1f}%", "trend": item["trend_label"]}
        for item in topics
    ]
    gap_items = [
        html.Details(
            [
                html.Summary([html.Span(f"{item['topic_a']} × {item['topic_b']}"), html.Strong(f"Gap score {item['gap_score']:.3f}")]),
                html.Div([html.P(item["explanation"]), html.P(f"Косинусная близость: {item['similarity']:.3f}"), html.P(f"Пограничных публикаций: {item['bridge_count']}")], className="gap-details"),
            ],
            className="gap-item",
        )
        for item in gaps
    ]
    return html.Div(
        [
            html.H2("Метрики модели"), metrics_view(model.get("metrics") or {}),
            html.Div(dcc.Graph(figure=scatter), className="chart-card"),
            html.Div(dcc.Graph(figure=trends), className="chart-card"),
            html.Section(
                [html.H2("Тематики"), dash_table.DataTable(data=topic_rows, columns=[{"name": key, "id": key} for key in topic_rows[0]] if topic_rows else [], sort_action="native", page_size=12, style_table={"overflowX": "auto"}, style_header={"textAlign": "center"})],
                className="table-card result-section",
            ),
            html.Section(
                [html.H2("Кандидаты на research gaps"), html.P("Автоматические гипотезы требуют проверки исследователем.", className="muted"), html.Div(gap_items or [html.P("Кандидаты не найдены.")], className="gap-list")],
                className="result-section",
            ),
        ],
        className="results-stack",
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
