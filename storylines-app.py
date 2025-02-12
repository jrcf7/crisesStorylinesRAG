import os
import pandas as pd
from datetime import datetime, date
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import ast

# # Environment settings
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# os.environ["HF_HUB_CACHE"] = "/eos/jeodpp/home/users/consose/cache/huggingface/hub"
# os.environ["HUGGINGFACE_HUB_CACHE"] = "/eos/jeodpp/home/users/consose/cache/huggingface/hub"
# os.environ["HF_HOME"] = "/eos/jeodpp/home/users/consose/cache/huggingface/hub"

# Load the CSV file
#df = pd.read_csv("emdat2.csv", sep=',', header=0, dtype=str, encoding='utf-8')
#df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat2.csv", sep=',', header=0, dtype=str, encoding='utf-8')
#df = pd.read_csv("/eos/jeodpp/home/users/roncmic/data/crisesStorylinesRAG/procem_graph.csv", sep=',', header=0, dtype=str, encoding='utf-8')
df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/procem_graph.csv", sep=',', header=0, dtype=str, encoding='utf-8')
#df = df.drop_duplicates(subset='DisNo.', keep='first')  #I drop all duplicates for column "DisNo.", keeping the first occurrence


# grp=eval(df.iloc[0]["causal graph"])
#
# source, relations, target = list(zip(*grp))
# kg_df = pd.DataFrame({'source':source, 'target':target, 'edge':relations})
#
# print("ciao")


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
    edge_colors_dict = {"causes": "red", "prevents": "green"}
    edge_color_list = [edge_colors_dict.get(G[u][v][key]['edge'], 'black') for u, v, key in G.edges(keys=True)]

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, k=1.5, iterations=100)
    nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=800, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color=edge_color_list, arrows=True, width=2)
    nx.draw_networkx_labels(G, pos)
    legend_elements = [Line2D([0], [0], color=color, label=edge_type, lw=2) for edge_type, color in
                       edge_colors_dict.items()]
    plt.legend(handles=legend_elements, loc='best')
    plt.axis('off')
    plt.tight_layout()
    return plt.gcf()

def display_info(selected_row_str, country, year, month, day, graph_type):
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
        #"Entry Date", "Last Update"
    ]

    if selected_row_str is None or selected_row_str == '':
        print("No row selected.")
        return ('', '', '', '', '', '', '', None, '', '') + tuple([''] * len(additional_fields))

    print(f"Selected Country: {country}, Selected Row: {selected_row_str}, Date: {year}-{month}-{day}")

    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    selected_date = try_parse_date(year, month, day)

    if selected_date:
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: (
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                    (try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'], "01" if row['End Day'] == "" else row['End Day']) is not None) and
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) <= selected_date <=
                     try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'], "01" if row['End Day'] == "" else row['End Day']))
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
                    print("Invalid selected date.")
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) < eend)
                        ), axis=1)]
            else:
                try:
                    sstart = try_parse_date(year, "01", "01")
                    eend = try_parse_date(year, "12", "31")
                except Exception as err:
                    print("Invalid selected date.")
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) <= eend)
                        ), axis=1)]

        else:
            print("Invalid selected date.")

    # Use the "DisNo." column for selecting the row
    row_data = filtered_df[filtered_df['DisNo.'] == selected_row_str].squeeze()

    if not row_data.empty:
        print(f"Row data: {row_data}")
        key_information = row_data.get('key information', '')
        severity = row_data.get('severity', '')
        key_drivers = row_data.get('key drivers', '')
        impacts_exposure_vulnerability = row_data.get('main impacts, exposure, and vulnerability', '')
        likelihood_multi_hazard = row_data.get('likelihood of multi-hazard risks', '')
        best_practices = row_data.get('best practices for managing this risk', '')
        recommendations = row_data.get('recommendations and supportive measures for recovery', '')
        if graph_type == "LLaMA Graph":
            causal_graph_caption = row_data.get('llama graph', '')
        elif graph_type == "Mixtral Graph":
            causal_graph_caption = row_data.get('mixtral graph', '')
        elif graph_type == "Ensemble Graph":
            causal_graph_caption = row_data.get('ensemble graph', '')
        else:
            causal_graph_caption = ''
        #causal_graph_caption = row_data.get('causal graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_plot = plot_cgraph(grp)

        # Parse and format the start date
        start_date = try_parse_date(row_data['Start Year'], row_data['Start Month'], row_data['Start Day'])
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else str(row_data['Start Year'])+"-"+str(row_data['Start Month'])+"-"+str(row_data['Start Day']) #'N/A'

        # Parse and format the end date
        end_date = try_parse_date(row_data['End Year'], row_data['End Month'], row_data['End Day'])
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else str(row_data['End Year'])+"-"+str(row_data['End Month'])+"-"+str(row_data['End Day']) #'N/A'

        # Collect additional field data
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
        print("No valid data found for the selection.")
        return ('', '', '', '', '', '', '', None, '', '') + tuple([''] * len(additional_fields))

def update_row_dropdown(country, year, month, day):
    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    selected_date = try_parse_date(year, month, day)

    if selected_date:
        # filtered_rows = []
        # for idx, row in filtered_df.iterrows():
        #     if (try_parse_date(row['Start Year'], row['Start Month'], row['Start Day']) is not None) and \
        #             (try_parse_date(row['End Year'], row['End Month'], row['End Day']) is not None) and \
        #             (try_parse_date(row['Start Year'], row['Start Month'], row['Start Day']) <= selected_date <= \
        #              try_parse_date(row['End Year'], row['End Month'], row['End Day'])):
        #         filtered_rows.append(row)
        #
        # filtered_df = pd.DataFrame(filtered_rows)
        filtered_df = filtered_df[filtered_df.apply(
            lambda row: (
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                    (try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'], "01" if row['End Day'] == "" else row['End Day']) is not None) and
                    (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) <= selected_date <=
                     try_parse_date(row['End Year'], "01" if row['End Month'] == "" else row['End Month'], "01" if row['End Day'] == "" else row['End Day']))
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
                    print("Invalid selected date.")
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) < eend)
                        ), axis=1)]
            else:
                try:
                    sstart = try_parse_date(year, "01", "01")
                    eend = try_parse_date(year, "12", "31")
                except Exception as err:
                    print("Invalid selected date.")
                    sstart = None
                    eend = None

                if sstart and eend:
                    filtered_df = filtered_df[filtered_df.apply(
                        lambda row: (
                                (try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) is not None) and
                                (sstart <= try_parse_date(row['Start Year'], "01" if row['Start Month'] == "" else row['Start Month'], "01" if row['Start Day'] == "" else row['Start Day']) <= eend)
                        ), axis=1)]

        else:
            print("Invalid selected date.")



    # Use the "DisNo." column for choices
    choices = filtered_df['DisNo.'].tolist() if not filtered_df.empty else []
    print(f"Available rows for {country} on {year}-{month}-{day}: {choices}")
    return gr.update(choices=choices, value=choices[0] if choices else None)


def build_interface():
    with gr.Blocks() as interface:   
        # Add title and description using text elements
        gr.Markdown("## From Data to Narratives: AI-Enhanced Disaster and Health Threats Storylines")  # Title
        gr.Markdown("This Gradio app complements Health Threats and Disaster event data through generative AI techniques, including the use of Retrieval Augmented Generation (RAG) with the [Europe Media Monitoring (EMM)](https://emm.newsbrief.eu/overview.html) service, "
                    "and Large Language Models (LLMs) from the [GPT@JRC](https://gpt.jrc.ec.europa.eu/) portfolio. <br>"
                    "The app leverages the EMM RAG service to retrieve relevant news chunks for each event data, transforms the unstructured news chunks into structured narratives and causal knowledge graphs using LLMs and text-to-graph techniques, linking health threats and disaster events to their causes and impacts. "
                    "Drawing data from sources like the [EM-DAT](https://www.emdat.be/) database, it augments each event with news-derived information in a storytelling fashion. <br>"
                    "This tool enables decision-makers to better explore health threats and disaster dynamics, identify patterns, and simulate scenarios for improved response and readiness. <br><br>"
                    "Select an event data below. You can filter by country and date period. Below, you will see the AI-generated storyline and causal knowledge graph, while on the right you can see the related EM-DAT data record.  <br><br>")  # Description  -, and constructs disaster-specific ontologies. "

        # Extract and prepare unique years from "Start Year" and "End Year"
        if not df.empty:
            start_years = df["Start Year"].dropna().unique()
            end_years = df["End Year"].dropna().unique()

            # Convert to integers and merge to create a union set
            years = set(start_years.astype(int).tolist() + end_years.astype(int).tolist())
            year_choices = sorted(years)
        else:
            year_choices = []

        country_dropdown = gr.Dropdown(choices=[''] + df['Country'].unique().tolist(), label="Select Country")
        year_dropdown = gr.Dropdown(choices=[""] + [str(year) for year in year_choices], label="Select Year")
        month_dropdown = gr.Dropdown(choices=[""] + [f"{i:02d}" for i in range(1, 13)], label="Select Month")
        day_dropdown = gr.Dropdown(choices=[""] + [f"{i:02d}" for i in range(1, 32)], label="Select Day")
        row_dropdown = gr.Dropdown(choices=[], label="Select Disaster Event #", interactive=True)
        graph_type_dropdown = gr.Dropdown(
            choices=["LLaMA Graph", "Mixtral Graph", "Ensemble Graph"], 
            label="Select Graph Type"
        )

        # Define the additional fields once to use later in both position and function
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
            #"Entry Date", "Last Update"
        ]

        with gr.Row():
            with gr.Column():
                # Main controls and outputs
                country_dropdown
                year_dropdown
                month_dropdown
                day_dropdown
                row_dropdown
                graph_type_dropdown

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
                # Additional information on the right
                outputs.extend([
                    gr.Textbox(label="Start Date", interactive=False),
                    gr.Textbox(label="End Date", interactive=False)
                ])
                for field in additional_fields:
                    outputs.append(gr.Textbox(label=field, interactive=False))

        # Update the selectable rows when any of the filters change
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

        # Update the display information when a row is selected
        row_dropdown.change(
            fn=display_info,
            inputs=[row_dropdown, country_dropdown, year_dropdown, month_dropdown, day_dropdown, graph_type_dropdown],
            outputs=outputs
        )
        graph_type_dropdown.change(
            fn=display_info,
            inputs=[row_dropdown, country_dropdown, year_dropdown, month_dropdown, day_dropdown, graph_type_dropdown],
            outputs=outputs
        )

    return interface


app = build_interface()
app.launch()
#app.launch(share=True)
