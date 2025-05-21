import os
import pandas as pd
from datetime import date
import gradio as gr
from pyvis.network import Network
import ast
from openai import OpenAI
import json 

EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    EMM_RETRIEVERS_OPENAI_API_KEY = config['EMM_RETRIEVERS_OPENAI_API_KEY']
    
client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

def gpt_story(storyline):
    prompt = (
        "Use the information provided to create a short, clear, and useful narrative about a disaster event. "
        "The goal is to help decision-makers (e.g. policy makers, disaster managers, civil protection) understand what happened, why, and what it caused. "
        "Keep it short and focused.\n\n"
        "Include all key information, but keep the text concise and easy to read. Avoid technical jargon.\n\n"
        "Steps to Follow:\n"
        "1. Start with what happened: Briefly describe the disaster event (what, where, when, who was affected).\n"
        "2. Explain why it happened: Use the evidence provided to describe possible causes or triggers (e.g. heavy rainfall, poor infrastructure, heatwave).\n"
        "3. Show the impacts: Highlight key impacts such as fatalities, displacement, health effects, or damage.\n"
        "4. Connect the dots: Show how different factors are linked. Use simple cause-effect language (e.g. drought led to crop failure, which caused food insecurity).\n"
        "5. Mention complexity if needed: If there were multiple contributing factors or reinforcing effects (e.g. climate + conflict), briefly explain them.\n"
        "6. Keep it useful: Write with a decision-maker in mind. Focus on what matters: drivers, impacts, and lessons for preparedness or response.\n\n"
        f"Information: {storyline}"
    )

    completion = client1.chat.completions.create(
        model='gpt-4o',
        messages=[
            {"role": "system", "content": "You are a disaster manager expert in risk dynamics."},
            {"role": "user", "content": prompt}
        ]
    )

    # Extract the content from the response
    message_content = completion.choices[0].message.content
    return message_content


# Load the CSV file
df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat2.csv", sep=',', header=0, dtype=str, encoding='utf-8')

def try_parse_date(y, m, d):
    try:
        if not y or not m or not d:
            return None
        return date(int(float(y)), int(float(m)), int(float(d)))
    except (ValueError, TypeError):
        return None

def plot_cgraph_pyvis(grp):
    if not grp:
        return "<div>No data available to plot.</div>"

    net = Network(notebook=False, directed=True)
    edge_colors_dict = {"causes": "red", "prevents": "green"}

    for src, rel, tgt in grp:
        src = str(src)
        tgt = str(tgt)
        rel = str(rel)
        net.add_node(src, shape="circle", label=src)
        net.add_node(tgt, shape="circle", label=tgt)
        edge_color = edge_colors_dict.get(rel, 'black')
        net.add_edge(src, tgt, title=rel, label=rel, color=edge_color)

    net.repulsion(
        node_distance=200,
        central_gravity=0.2,
        spring_length=200,
        spring_strength=0.05,
        damping=0.09
    )
    net.set_edge_smooth('dynamic')

    html = net.generate_html()
    html = html.replace("'", "\"")

    html_s = f"""<iframe style="width: 200%; height: 800px;margin:0 auto" name="result" allow="midi; geolocation; microphone; camera; 
    display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
    allow-scripts allow-same-origin allow-popups 
    allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
    allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>"""
    
    return html_s

def display_info(selected_row_str, country, year, month, day):

    if selected_row_str is None or selected_row_str == '':
        print("No row selected.")
        return ('', '', '', '', '', '', '', None, '', '')

    print(f"Selected Country: {country}, Selected Row: {selected_row_str}, Date: {year}-{month}-{day}")

    filtered_df = df
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    # Date filtering logic remains the same...

    # Use the "DisNo." column for selecting the row
    row_data = filtered_df[filtered_df['DisNo.'] == selected_row_str].squeeze()

    if not row_data.empty:
        print(f"Row data: {row_data}")

        # Combine the relevant columns into a single storyline with labels
        storyline_parts = [
            f"Key Information: {row_data.get('key information', '')}",
            f"Severity: {row_data.get('severity', '')}",
            f"Key Drivers: {row_data.get('key drivers', '')}",
            f"Main Impacts, Exposure, and Vulnerability: {row_data.get('main impacts, exposure, and vulnerability', '')}",
            f"Likelihood of Multi-Hazard Risks: {row_data.get('likelihood of multi-hazard risks', '')}",
            f"Best Practices for Managing This Risk: {row_data.get('best practices for managing this risk', '')}",
            f"Recommendations and Supportive Measures for Recovery: {row_data.get('recommendations and supportive measures for recovery', '')}"
        ]
        storyline = "\n\n".join(part for part in storyline_parts if part.split(': ')[1])  # Include only non-empty parts

        # Transform the storyline using GPT
        cleaned_storyline = gpt_story(storyline)

        # Directly fetch the LLama graph
        causal_graph_caption = row_data.get('llama graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_html = plot_cgraph_pyvis(grp)

        # Parse and format the start date
        start_date = try_parse_date(row_data['Start Year'], row_data['Start Month'], row_data['Start Day'])
        start_date_str = start_date.strftime('%Y-%m-%d') if start_date else str(row_data['Start Year'])+"-"+str(row_data['Start Month'])+"-"+str(row_data['Start Day'])

        # Parse and format the end date
        end_date = try_parse_date(row_data['End Year'], row_data['End Month'], row_data['End Day'])
        end_date_str = end_date.strftime('%Y-%m-%d') if end_date else str(row_data['End Year'])+"-"+str(row_data['End Month'])+"-"+str(row_data['End Day'])


        return (
            cleaned_storyline,
            causal_graph_html,
            start_date_str,
            end_date_str
        ) 
    else:
        print("No valid data found for the selection.")
        return ('', '', '', '', '', '', '', None, '', '')



def update_row_dropdown(country, year, month, day):
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



    # Use the "DisNo." column for choices
    choices = filtered_df['DisNo.'].tolist() if not filtered_df.empty else []
    print(f"Available rows for {country} on {year}-{month}-{day}: {choices}")
    return gr.update(choices=choices, value=choices[0] if choices else None)


def build_interface():
    with gr.Blocks() as interface:
        gr.Markdown(
        """
        # From Complexity to Clarity: Leveraging AI to Decode Interconnected Risks

        Welcome to our Gradio application, developed and maintained by [JRC](https://joint-research-centre.ec.europa.eu/index_en/) Units: **E1**, **F7**, and **T5**. This is part of the **EMBRACE Portfolio on Risks**. <br><br>

        **Overview**:  
        This application employs advanced AI techniques like Retrieval-Augmented Generation (RAG) on [EMM](https://emm.newsbrief.eu/overview.html/) news. It extracts relevant media content on disaster events recorded in [EM-DAT](https://www.emdat.be/), including floods, wildfires, droughts, epidemics, and disease outbreaks. <br><br>

        **How It Works**:  
        For each selected event (filterable by Year, Country, Disaster Type, Month, and Disaster Number), the app:
        - Retrieves pertinent news chunks via the EMM RAG service.
        - Uses multiple LLMs from the [GPT@JRC](https://gpt.jrc.ec.europa.eu/) portfolio to:
            - Extract critical impact data (e.g., fatalities, affected populations).
            - Transform unstructured news into coherent, structured storylines.
            - Build causal knowledge graphs — *impact chains* — highlighting drivers, impacts, and interactions. <br><br>

        **Explore Events**:  
        Use the selectors above to explore events by **Year**, **Country**, **Disaster Type**, **Month**, and **Disaster Number (DisNo)**. <br>
        Once an event is selected, the app will display the **causal impact-chain graph**, illustrating key factors and their interrelationships. <br>
        Below the graph, you'll find the **AI-generated narrative**, presenting a structured storyline of the event based on relevant news coverage. <br><br>

        **Outcome**:  
        These outputs offer a deeper understanding of disaster dynamics, supporting practitioners, disaster managers, and policy-makers in identifying patterns, assessing risks, and enhancing preparedness and response strategies.
        """
    )

        # Extract and prepare unique years from "Start Year" and "End Year"
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


        with gr.Column():
            country_dropdown
            year_dropdown
            month_dropdown
            day_dropdown
            row_dropdown

            gr.Markdown("### AI-Generated Storyline:")  # Title
            outputs = [
                gr.Textbox(label="Storyline", interactive=False, lines=10),
                gr.HTML(label="Causal Graph")  # Change from gr.Plot to gr.HTML
            ]

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
