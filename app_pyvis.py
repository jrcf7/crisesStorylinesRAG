import os
import pandas as pd
from datetime import date
import gradio as gr
from pyvis.network import Network
import ast
from openai import OpenAI
import json 
import string
from datetime import datetime
import random

EMM_RETRIEVERS_OPENAI_API_BASE_URL="https://api-gpt.jrc.ec.europa.eu/v1"
with open('./data/gpt_token.json', 'r') as file:
    config = json.load(file)
    EMM_RETRIEVERS_OPENAI_API_KEY = config['EMM_RETRIEVERS_OPENAI_API_KEY']
    
client1 = OpenAI(
    api_key=EMM_RETRIEVERS_OPENAI_API_KEY,
    base_url="https://api-gpt.jrc.ec.europa.eu/v1",
)

df = pd.read_csv("https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/ETOHA/storylines/emdat2.csv", sep=',', header=0, dtype=str, encoding='utf-8')


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


# DataFrame to store evaluation data
evaluation_df = pd.DataFrame(columns=["DisNo.", "TPN", "TPL", "FPN", "FPL", "FNN", "FNL", "User ID"])



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

    # Adjust the iframe style to center the graph and fit the container
    html_s = f"""
    <div style="display: flex; justify-content: center; align-items: center;">
        <iframe style="width: 90%; height: 800px; margin: 0 auto;" name="result" allow="midi; geolocation; microphone; camera; 
        display-capture; encrypted-media;" sandbox="allow-modals allow-forms 
        allow-scripts allow-same-origin allow-popups 
        allow-top-navigation-by-user-activation allow-downloads" allowfullscreen="" 
        allowpaymentrequest="" frameborder="0" srcdoc='{html}'></iframe>
    </div>
    """
    
    return html_s

def generate_unique_user_id():
    # Generate a timestamp string
    timestamp_str = datetime.now().strftime("%Y%m%d%H%M%S")
    # Generate a random string of 5 letters
    random_str = ''.join(random.choices(string.ascii_letters, k=5))
    # Combine both to form a unique User ID
    return f"{timestamp_str}_{random_str}"

def save_data_to_csv():
    # Save the evaluation DataFrame to a CSV file
    evaluation_df.to_csv("evaluation_data.csv", index=False)
    print("Data saved to CSV successfully.")

def save_data(dis_no, tpn, tpl, fp_node, fp_link, fn_node, fn_link):
    global evaluation_df

    # Debug: Print input values to ensure they're being received correctly
    print(f"Inputs received - DisNo: {dis_no}, TPN: {tpn}, TPL: {tpl}, FPN: {fp_node}, FPL: {fp_link}, FNN: {fn_node}, FNL: {fn_link}")

    # Check if a valid disaster number has been selected
    if not dis_no or dis_no == "Select a Disaster Event":
        print("Invalid input. Ensure a disaster event is selected.")
        return  # Ensure no output is returned

    # Generate a unique User ID
    user_id = generate_unique_user_id()

    # Append the new data to the DataFrame
    new_data = pd.DataFrame([[dis_no, tpn, tpl, fp_node, fp_link, fn_node, fn_link, user_id]], 
                            columns=["DisNo.", "TPN", "TPL", "FPN", "FPL", "FNN", "FNL", "User ID"])
    evaluation_df = pd.concat([evaluation_df, new_data], ignore_index=True)

    # Debug: Print the updated DataFrame to verify the new row addition
    print("Updated DataFrame:")
    print(evaluation_df)

    # Save the DataFrame to a CSV file
    save_data_to_csv()

    print(f"Data saved: DisNo: {dis_no}, TPN: {tpn}, TPL: {tpl}, FPN: {fp_node}, FPL: {fp_link}, FNN: {fn_node}, FNL: {fn_link}, User ID: {user_id}")

def update_row_dropdown(disaster_type=None, country=None):
    # Start with the entire dataframe
    filtered_df = df

    # Step 1: Filter by Disaster Type
    if disaster_type:
        filtered_df = filtered_df[filtered_df['Disaster Type'] == disaster_type]

    # Step 2: Further filter by Country
    if country:
        filtered_df = filtered_df[filtered_df['Country'] == country]

    # Step 3: Generate the DisNo. choices based on the filtered DataFrame
    choices = filtered_df['DisNo.'].tolist() if not filtered_df.empty else []

    # Add a placeholder option at the beginning
    choices = ["Select a Disaster Event"] + choices

    print(f"Available DisNo. for {disaster_type} in {country}: {choices}")
    
    # Return the update for the dropdown, defaulting to the placeholder
    return gr.update(choices=choices, value=choices[0] if choices else None)

def display_info(selected_row_str, country):
    if not selected_row_str or selected_row_str == 'Select a Disaster Event':
        print("No valid disaster event selected.")
        return ('No valid event selected.', '<div>No graph available.</div>', '', '')

    print(f"Selected Country: {country}, Selected Row: {selected_row_str}")

    # Filter the dataframe for the selected disaster number
    row_data = df[df['DisNo.'] == selected_row_str].squeeze()

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
        cleaned_storyline = gpt_story(storyline)
        causal_graph_caption = row_data.get('llama graph', '')
        grp = ast.literal_eval(causal_graph_caption) if causal_graph_caption else []
        causal_graph_html = plot_cgraph_pyvis(grp)
        
        # Parse and format the start date
        start_date_str = f"{row_data['Start Year']}-{row_data['Start Month']}-{row_data['Start Day']}"

        # Parse and format the end date
        end_date_str = f"{row_data['End Year']}-{row_data['End Month']}-{row_data['End Day']}"

        return (
            cleaned_storyline,
            causal_graph_html,
            start_date_str,
            end_date_str
        ) 
    else:
        print("No valid data found for the selection.")
        return ('No valid data found.', '<div>No graph available.</div>', '', '')

def build_interface():
    with gr.Blocks() as interface:
        gr.Markdown(
        """
        # From Complexity to Clarity: Leveraging AI to Decode Interconnected Risks

        Welcome to our Gradio application, developed and maintained by [JRC](https://joint-research-centre.ec.europa.eu/index_en/) Units: **E1**, **F7**, and **T5**. This is part of the **EMBRACE Portfolio on Risks**. <br><br>

        **Overview**:  
        This application employs advanced AI techniques like Retrieval-Augmented Generation (RAG) on [EMM](https://emm.newsbrief.eu/) news. It extracts relevant media content on disaster events recorded in [EM-DAT](https://www.emdat.be/), including floods, wildfires, droughts, epidemics, and disease outbreaks. <br><br>

        **How It Works**:  
        For each selected event (filterable by Disaster Type, Country, and Disaster Number), the app:
        - Retrieves pertinent news chunks via the EMM RAG service.
        - Uses multiple LLMs from the [GPT@JRC](https://gpt.jrc.ec.europa.eu/) portfolio to:
            - Extract critical impact data (e.g., fatalities, affected populations).
            - Transform unstructured news into coherent, structured storylines.
            - Build causal knowledge graphs — *impact chains* — highlighting drivers, impacts, and interactions. <br><br>

        **Explore Events**:  
        Use the selectors below to explore events by **Disaster Type**, **Country**, and **Disaster Number (DisNo)**. <br>
        Once an event is selected, the app will display the **causal impact-chain graph**, illustrating key factors and their interrelationships. <br>
        Below the graph, you'll find the **AI-generated narrative**, presenting a structured storyline of the event based on relevant news coverage. <br><br>

        **Outcome**:  
        These outputs offer a deeper understanding of disaster dynamics, supporting practitioners, disaster managers, and policy-makers in identifying patterns, assessing risks, and enhancing preparedness and response strategies.
        """
    )

        # Create dropdowns for Disaster Type, Country, and Disaster Event #
        disaster_type_dropdown = gr.Dropdown(
            choices=[''] + df['Disaster Type'].unique().tolist(),
            label="Select Disaster Type"
        )
        country_dropdown = gr.Dropdown(
            choices=[''],  # Initially empty; will be populated based on disaster type
            label="Select Country"
        )
        row_dropdown = gr.Dropdown(
            choices=[],
            label="Select Disaster Event #",
            interactive=True
        )

        with gr.Column():
            disaster_type_dropdown
            country_dropdown
            row_dropdown

            gr.Markdown("### AI-Generated Storyline:")  # Title
            outputs = [
                gr.Textbox(label="Storyline", interactive=False, lines=10),
                gr.HTML(label="Causal Graph")  # Change from gr.Plot to gr.HTML
            ]

            # Inputs for evaluation metrics, placed after the graph
            with gr.Row():
                tpn_input = gr.Number(label="Num of Correct Nodes (TPN)", value=0, interactive=True)
                tpl_input = gr.Number(label="Num of Correct Links (TPL)", value=0, interactive=True)
                fp_node_input = gr.Number(label="False Positive Nodes (FPN)", value=0, interactive=True)
                fp_link_input = gr.Number(label="False Positive Links (FPL)", value=0, interactive=True)
                fn_node_input = gr.Number(label="False Negative Nodes (FNN)", value=0, interactive=True)
                fn_link_input = gr.Number(label="False Negative Links (FNL)", value=0, interactive=True)

            # Button to save the data
            save_button = gr.Button("Save Data")

        # Update country choices based on selected disaster type
        disaster_type_dropdown.change(
            fn=lambda disaster_type: gr.update(
                choices=[''] + df[df['Disaster Type'] == disaster_type]['Country'].unique().tolist(),
                value=''
            ),
            inputs=disaster_type_dropdown,
            outputs=country_dropdown
        )

        # Update DisNo. choices based on selected disaster type and country
        country_dropdown.change(
            fn=update_row_dropdown,
            inputs=[disaster_type_dropdown, country_dropdown],
            outputs=row_dropdown
        )

        # Display information based on selected DisNo.
        row_dropdown.change(
            fn=display_info,
            inputs=[row_dropdown, country_dropdown],
            outputs=outputs
        )

        # Handle saving data on button click
        save_button.click(
            fn=save_data,
            inputs=[row_dropdown, tpn_input, tpl_input, fp_node_input, fp_link_input, fn_node_input, fn_link_input],
            outputs=[]
        )

    return interface

app = build_interface()
app.launch()
