import os
import pandas as pd
from datetime import date
import gradio as gr
import networkx as nx
import ast
import plotly.graph_objects as go
import plotly.express as px

# Load the CSV file
df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat2.csv", sep=',', header=0,
                 dtype=str, encoding='utf-8')
df = df.drop_duplicates(subset='DisNo.', keep='first')  #I drop all duplicates for column "DisNo.", keeping the first occurrence

def try_parse_date(y, m, d):
    try:
        if not y or not m or not d:
            return None
        return date(int(float(y)), int(float(m)), int(float(d)))
    except (ValueError, TypeError):
        return None


def plot_cgraph(grp):
    if not grp:
        return None
    source, relations, target = list(zip(*grp))
    kg_df = pd.DataFrame({'source': source, 'target': target, 'edge': relations})
    G = nx.from_pandas_edgelist(kg_df, "source", "target", edge_attr='edge', create_using=nx.MultiDiGraph())

    pos = nx.spring_layout(G, k=1.5, iterations=100)

    # Separate edges based on their color
    edge_colors_dict = {"causes": "red", "prevents": "green"}
    traces = []

    for color in edge_colors_dict.values():
        edge_x = []
        edge_y = []
        for u, v, key in G.edges(keys=True):
            current_color = edge_colors_dict.get(G[u][v][key]['edge'], 'black')
            if current_color == color:
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

        trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=2, color=color), hoverinfo='none', mode='lines')
        traces.append(trace)

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text', text=node_text,
        marker=dict(size=10, color='skyblue', line_width=2),
        textposition="top center", hoverinfo='text'
    )

    traces.append(node_trace)

    fig = go.Figure(data=traces,
                    layout=go.Layout(showlegend=False,
                                     hovermode='closest',
                                     margin=dict(b=20, l=5, r=5, t=40)))

    return fig


def display_info(selected_row_str, country, year, month, day):
    additional_fields = [
        "Country", "ISO", "Subregion", "Region", "Location", "Origin",
        "Disaster Group", "Disaster Subgroup", "Disaster Type", "Disaster Subtype", "External IDs",
        "Event Name", "Associated Types", "OFDA/BHA Response", "Appeal", "Declaration",
        "AID Contribution ('000 US$)", "Magnitude", "Magnitude Scale", "Latitude",
        "Longitude", "River Basin", "Total Deaths", "No. Injured",
        "No. Affected", "No. Homeless", "Total Affected",
        "Reconstruction Costs ('000 US$)", "Reconstruction Costs, Adjusted ('000 US$)",
        "Insured Damage ('000 US$)", "Insured Damage, Adjusted ('000 US$)",
        "Total Damage ('000 US$)", "Total Damage, Adjusted ('000 US$)", "CPI",
        "Admin Units",
    ]

    if selected_row_str is None or selected_row_str == '':
        return ('', '', '', '', '', '', '', None, '', '') + tuple([''] * len(additional_fields))

    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    selected_date = try_parse_date(year, month, day)

    if selected_date:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: (
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'],
                                    "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                    (try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'],
                                    "01" if row['End Day'] == "" else row['End Day']) is not None) and
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'],
                                    "01" if row['Start Day'] == "" else row['Start Day']) <= selected_date <=
                     try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'],
                                    "01" if row['End Day'] == "" else row['End Day']))
            ), axis=1)]
    else:
        if year:
            sstart = None
            eend = None
            if month:
                try:
                    sstart = try_parse_date(year, month, "01")
                    eend = try_parse_date(year, int(float(month)) + 1, "01")
                except Exception as err:
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'],
                                                "01" if row['Start Month'] == "" else row['Start Month'],
                                                "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'],
                                                          "01" if row['Start Month'] == "" else row['Start Month'],
                                                          "01" if row['Start Day'] == "" else row['Start Day']) < eend)
                        ), axis=1)]
            else:
                try:
                    sstart = try_parse_date(year, "01", "01")
                    eend = try_parse_date(year, "12", "31")
                except Exception as err:
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'],
                                                "01" if row['Start Month'] == "" else row['Start Month'],
                                                "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'],
                                                          "01" if row['Start Month'] == "" else row['Start Month'],
                                                          "01" if row['Start Day'] == "" else row['Start Day']) <= eend)
                        ), axis=1)]

    row_data = filtered_df[filtered_df['DisNo.'] == selected_row_str].squeeze()

    if not row_data.empty:
        key_information = row_data.get('key information', '')
        severity = row_data.get('severity', '')
        key_drivers = row_data.get('key drivers', '')
        impacts_exposure_vulnerability = row_data.get('main impacts, exposure, and vulnerability', '')
        likelihood_multi_hazard = row_data.get('likelihood of multi-hazard risks', '')
        best_practices = row_data.get('best practices for managing this risk', '')
        recommendations = row_data.get('recommendations and supportive measures for recovery', '')
        causal_graph_caption = row_data.get('causal graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_plot = plot_cgraph(grp)

        start_date = try_parse_date(row_data['Start Year'], row_data['Start Month'], row_data['Start Day'])
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else 'N/A'

        end_date = try_parse_date(row_data['End Year'], row_data['End Month'], row_data['End Day'])
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else 'N/A'

        additional_data = [row_data.get(field, '') for field in additional_fields]

        return (
            key_information,
            severity,
            key_drivers,
            impacts_exposure_vulnerability,
            likelihood_multi_hazard,
            best_practices,
            recommendations,
            causal_graph_plot,
            start_date_str,
            end_date_str
        ) + tuple(additional_data)
    else:
        return ('', '', '', '', '', '', '', None, '', '') + tuple([''] * len(additional_fields))


def update_row_dropdown(country, year, month, day):
    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    selected_date = try_parse_date(year, month, day)

    if selected_date:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: (
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'],
                                    "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                    (try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'],
                                    "01" if row['End Day'] == "" else row['End Day']) is not None) and
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'],
                                    "01" if row['Start Day'] == "" else row['Start Day']) <= selected_date <=
                     try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'],
                                    "01" if row['End Day'] == "" else row['End Day']))
            ), axis=1)]
    else:
        if year:
            sstart = None
            eend = None
            if month:
                try:
                    sstart = try_parse_date(year, month, "01")
                    eend = try_parse_date(year, int(float(month)) + 1, "01")
                except Exception as err:
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'],
                                                "01" if row['Start Month'] == "" else row['Start Month'],
                                                "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'],
                                                          "01" if row['Start Month'] == "" else row['Start Month'],
                                                          "01" if row['Start Day'] == "" else row['Start Day']) < eend)
                        ), axis=1)]
            else:
                try:
                    sstart = try_parse_date(year, "01", "01")
                    eend = try_parse_date(year, "12", "31")
                except Exception as err:
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'],
                                                "01" if row['Start Month'] == "" else row['Start Month'],
                                                "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'],
                                                          "01" if row['Start Month'] == "" else row['Start Month'],
                                                          "01" if row['Start Day'] == "" else row['Start Day']) <= eend)
                        ), axis=1)]

    choices = filtered_df['DisNo.'].tolist() if not filtered_df.empty else []
    return gr.update(choices=choices, value=choices[0] if choices else None)


def build_interface():
    with gr.Blocks() as interface:
        gr.Markdown("## From Data to Narratives: AI-Enhanced Disaster and Health Threats Storylines")
        gr.Markdown(
            "This Gradio app complements Health Threats and Disaster event data... <br>"
            "Select an event data below..."
        )

        if not df.empty:
            start_years = df["Start Year"].dropna().unique()
            end_years = df["End Year"].dropna().unique()
            years = set(start_years.astype(int).tolist() + end_years.astype(int).tolist())
            year_choices = sorted(years)
        else:
            year_choices = []

        country_dropdown = gr.Dropdown(choices=[''] + df['Country'].unique().tolist(), label="Select Country")
        year_dropdown = gr.Dropdown(choices=[""] + [str(year) for year in year_choices], label="Select Year")
        month_dropdown = gr.Dropdown(choices=[""] + [f"{i:02d}" for i in range(1, 13)], label="Select Month")
        day_dropdown = gr.Dropdown(choices=[""] + [f"{i:02d}" for i in range(1, 32)], label="Select Day")
        row_dropdown = gr.Dropdown(choices=[], label="Select Disaster Event #", interactive=True)

        additional_fields = [
            "Country", "ISO", "Subregion", "Region", "Location", "Origin",
            "Disaster Group", "Disaster Subgroup", "Disaster Type", "Disaster Subtype", "External IDs",
            "Event Name", "Associated Types", "OFDA/BHA Response", "Appeal", "Declaration",
            "AID Contribution ('000 US$)", "Magnitude", "Magnitude Scale", "Latitude",
            "Longitude", "River Basin", "Total Deaths", "No. Injured",
            "No. Affected", "No. Homeless", "Total Affected",
            "Reconstruction Costs ('000 US$)", "Reconstruction Costs, Adjusted ('000 US$)",
            "Insured Damage ('000 US$)", "Insured Damage, Adjusted ('000 US$)",
            "Total Damage ('000 US$)", "Total Damage, Adjusted ('000 US$)", "CPI",
            "Admin Units",
        ]

        with gr.Row():
            with gr.Column():
                country_dropdown
                year_dropdown
                month_dropdown
                day_dropdown
                row_dropdown

                outputs = [
                    gr.Textbox(label="Key Information", interactive=False),
                    gr.Textbox(label="Severity", interactive=False),
                    gr.Textbox(label="Key Drivers", interactive=False),
                    gr.Textbox(label="Main Impacts, Exposure, and Vulnerability", interactive=False),
                    gr.Textbox(label="Likelihood of Multi-Hazard Risks", interactive=False),
                    gr.Textbox(label="Best Practices for Managing This Risk", interactive=False),
                    gr.Textbox(label="Recommendations and Supportive Measures for Recovery", interactive=False),
                    gr.Plot(label="Causal Graph")
                ]

            with gr.Column():
                outputs.extend([
                    gr.Textbox(label="Start Date", interactive=False),
                    gr.Textbox(label="End Date", interactive=False)
                ])
                for field in additional_fields:
                    outputs.append(gr.Textbox(label=field, interactive=False))

        country_dropdown.change(
            fn=update_row_dropdown,
            inputs=[country_dropdown, year_dropdown, month_dropdown, day_dropdown],
            outputs=row_dropdown
        )
        year_dropdown.change(
            fn=update_row_dropdown,
            inputs=[country_dropdown, year_dropdown, month_dropdown, day_dropdown],
            outputs=row_dropdown
        )
        month_dropdown.change(
            fn=update_row_dropdown,
            inputs=[country_dropdown, year_dropdown, month_dropdown, day_dropdown],
            outputs=row_dropdown
        )
        day_dropdown.change(
            fn=update_row_dropdown,
            inputs=[country_dropdown, year_dropdown, month_dropdown, day_dropdown],
            outputs=row_dropdown
        )

        row_dropdown.change(
            fn=display_info,
            inputs=[row_dropdown, country_dropdown, year_dropdown, month_dropdown, day_dropdown],
            outputs=outputs
        )

    return interface


app = build_interface()
app.launch()
