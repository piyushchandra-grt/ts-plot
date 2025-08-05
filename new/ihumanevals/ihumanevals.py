import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap
def load_clean_and_sort_data(filepath):
    """
    Loads the CSV, cleans count and ID columns, and sorts the data by Prompt ID.
    This sorting is the key enhancement to ensure plots follow the desired sequence.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{filepath}' was not found.")
        return None
    # --- ENHANCEMENT: Ensure Prompt ID is numeric for sorting ---
    df['Prompt ID'] = pd.to_numeric(df['Prompt ID'], errors='coerce')
    df.dropna(subset=['Prompt ID'], inplace=True) # Remove rows if Prompt ID is invalid
    df['Prompt ID'] = df['Prompt ID'].astype(int)
    # Clean the count columns to ensure they are numeric
    count_columns = [col for col in df.columns if '(count of 5s)' in col]
    for col in count_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    # Clean topic names for better display
    df['Topic'] = df['Topic'].str.replace('‑', ' ', regex=False)
    # --- KEY ENHANCEMENT: Sort the entire DataFrame by Prompt ID ---
    # This ensures that when we iterate through groups, the topics and industries
    # will appear in the correct numerical sequence based on the Prompt ID.
    df.sort_values(by='Prompt ID', inplace=True)
    return df
def create_industry_plot(language, industry, data_slice, metric_name, filename):
    """
    Creates a detailed bar chart for a specific industry, showing its topics.
    The `data_slice` is pre-sorted by Prompt ID, so topics appear in order.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'o4-mini-high': '#D9534F', 'Gemini 2.5 pro': '#5CB85C', 'Claude Opus 4': '#428BCA'}
    model_names_in_legend = {'g4': 'o4-mini-high', 'gemini': 'Gemini 2.5 pro', 'claude': 'Claude Opus 4'}
    # The topics will now be in the correct sequence due to the initial sort
    subtopics = data_slice['Topic'].tolist()
    x = np.arange(len(subtopics))
    width = 0.25
    g4_col = f'o4-mini-high - {metric_name} (count of 5s)'
    gemini_col = f'Gemini 2.5 Pro - {metric_name} (count of 5s)'
    claude_col = f'Claude Opus 4 - {metric_name} (count of 5s)'
    g4_counts = data_slice[g4_col].tolist()
    gemini_counts = data_slice[gemini_col].tolist()
    claude_counts = data_slice[claude_col].tolist()
    ax.bar(x - width, g4_counts, width, label=model_names_in_legend['g4'], color=colors[model_names_in_legend['g4']])
    ax.bar(x, gemini_counts, width, label=model_names_in_legend['gemini'], color=colors[model_names_in_legend['gemini']])
    ax.bar(x + width, claude_counts, width, label=model_names_in_legend['claude'], color=colors[model_names_in_legend['claude']])
    ax.set_ylabel("Count of '5' Ratings", fontsize=12, fontweight='bold')
    ax.set_xlabel("Topics", fontsize=12, fontweight='bold')
    ax.set_title(f'{language} - {industry}', fontsize=16, fontweight='bold')
    max_count = max(max(g4_counts, default=0), max(gemini_counts, default=0), max(claude_counts, default=0))
    y_limit = max(max_count + 1, 3)
    ax.set_ylim(0, y_limit)
    ax.set_yticks(np.arange(0, y_limit + 1, 1))
    wrapped_subtopics = [textwrap.fill(s, 20) for s in subtopics]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_subtopics, fontsize=9)
    ax.tick_params(axis='x', pad=10) # Add padding to x-axis ticks
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Models', fontsize=11)
    fig.suptitle(f'Evaluation Metric: {metric_name}', fontsize=14, color='black')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")
def create_overall_plot(language, overall_data_slice, metric_name, filename):
    """
    Creates a summary bar chart for a language, showing its industries.
    The `overall_data_slice` is pre-sorted by the minimum Prompt ID in each industry.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = {'o4-mini-high': '#F93E3E', 'Gemini 2.5 pro': '#5CD167', 'Claude Opus 4': '#1F77B4'}
    model_names_in_legend = {'g4': 'o4-mini-high', 'gemini': 'Gemini 2.5 pro', 'claude': 'Claude Opus 4'}
    # The industries will now be in the correct sequence
    industries = overall_data_slice['Industry'].tolist()
    x = np.arange(len(industries))
    width = 0.25
    g4_col = f'o4-mini-high - {metric_name} (count of 5s)'
    gemini_col = f'Gemini 2.5 Pro - {metric_name} (count of 5s)'
    claude_col = f'Claude Opus 4 - {metric_name} (count of 5s)'
    g4_counts = overall_data_slice[g4_col].tolist()
    gemini_counts = overall_data_slice[gemini_col].tolist()
    claude_counts = overall_data_slice[claude_col].tolist()
    ax.bar(x - width, g4_counts, width, label=model_names_in_legend['g4'], color=colors[model_names_in_legend['g4']])
    ax.bar(x, gemini_counts, width, label=model_names_in_legend['gemini'], color=colors[model_names_in_legend['gemini']])
    ax.bar(x + width, claude_counts, width, label=model_names_in_legend['claude'], color=colors[model_names_in_legend['claude']])
    ax.set_ylabel("Total Count of '5' Ratings (All Topics)", fontsize=12, fontweight='bold')
    ax.set_xlabel("Industries", fontsize=12, fontweight='bold')
    ax.set_title(f'{language} - Overall', fontsize=16, fontweight='bold')
    max_count = max(max(g4_counts, default=0), max(gemini_counts, default=0), max(claude_counts, default=0))
    y_limit = max(max_count + 1, 5)
    ax.set_ylim(0, y_limit)
    ax.set_yticks(np.arange(0, y_limit + 1, step=max(1, int(y_limit/5))))
    wrapped_industries = [textwrap.fill(s, 18) for s in industries]
    ax.set_xticks(x)
    ax.set_xticklabels(wrapped_industries, fontsize=10)
    ax.tick_params(axis='x', pad=10) # Add padding to x-axis ticks
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(title='Models', fontsize=11)
    fig.suptitle(f'Evaluation Metric: {metric_name}', fontsize=14, color='black')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"  Saved: {filename}")
if __name__ == '__main__':
    csv_filepath = 'Untitled spreadsheet - Sheet1 (4).csv'
    # Use the new enhanced function to load and sort the data
    df = load_clean_and_sort_data(csv_filepath)
    if df is not None:
        # Define output directories
        output_dir_industry = "Industry_Topic_Graphs"
        output_dir_overall = "Language_Overall_Graphs"
        os.makedirs(output_dir_industry, exist_ok=True)
        os.makedirs(output_dir_overall, exist_ok=True)
        print(f"Detailed graphs will be saved in: '{output_dir_industry}'")
        print(f"Overall graphs will be saved in: '{output_dir_overall}'")
        # --- ENHANCEMENT: Aggregate data for Overall graphs while keeping track of Prompt ID ---
        count_columns = [col for col in df.columns if '(count of 5s)' in col]
        agg_logic = {col: 'sum' for col in count_columns}
        agg_logic['Prompt ID'] = 'min' # Get the first Prompt ID for each industry group
        overall_agg_df = df.groupby(['Code Language', 'Industry']).agg(agg_logic).reset_index()
        # Sort the aggregated industries based on their first Prompt ID
        overall_agg_df.sort_values(by='Prompt ID', inplace=True)
        metrics_to_plot = ['Completeness', 'Correctness', 'Relevance']
        for metric in metrics_to_plot:
            print(f"\n--- Generating graphs for Metric: '{metric}' ---")
            # --- 1. Generate Industry-Specific (Detailed) Graphs ---
            # Grouping with sort=False respects the DataFrame's existing order (which is by Prompt ID)
            print("  Generating detailed industry-topic graphs...")
            industry_grouped = df.groupby(['Code Language', 'Industry'], sort=False)
            for (lang, industry), group_df in industry_grouped:
                display_lang = 'C++' if lang.lower() == 'cpp' else lang.capitalize()
                filename = f"{metric}_{display_lang.replace('C++', 'Cpp')}-{industry.replace(' ', '_')}.png"
                full_path = os.path.join(output_dir_industry, filename)
                create_industry_plot(display_lang, industry, group_df, metric, full_path)
            # --- 2. Generate Language-Specific (Overall) Graphs ---
            # Grouping the pre-sorted overall DataFrame to generate overall graphs
            print("  Generating overall language graphs...")
            overall_grouped = overall_agg_df.groupby('Code Language', sort=False)
            for lang, lang_group_df in overall_grouped:
                display_lang = 'C++' if lang.lower() == 'cpp' else lang.capitalize()
                filename = f"{metric}_{display_lang.replace('C++', 'Cpp')}-Overall.png"
                full_path = os.path.join(output_dir_overall, filename)
                create_overall_plot(display_lang, lang_group_df, metric, full_path)