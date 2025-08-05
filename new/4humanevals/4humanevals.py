import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import textwrap

def load_clean_and_sort_data(filepath):
    """
    Loads the CSV, robustly cleans columns, and most importantly,
    sorts the entire DataFrame by 'Prompt ID' to ensure sequential plotting.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"FATAL ERROR: The file '{filepath}' was not found.")
        return None

    df['Code Language'] = df['Code Language'].str.lower()

    # --- ENHANCEMENT: Clean and sort by Prompt ID ---
    df['Prompt ID'] = pd.to_numeric(df['Prompt ID'], errors='coerce')
    df.dropna(subset=['Prompt ID'], inplace=True)
    df['Prompt ID'] = df['Prompt ID'].astype(int)

    count_columns = [col for col in df.columns if '(count of 5s)' in col]
    for col in count_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df['Topic'] = df['Topic'].str.replace('‑', ' ', regex=False)

    # --- KEY ENHANCEMENT: Sort the entire DataFrame by Prompt ID ---
    df.sort_values(by='Prompt ID', inplace=True)
    return df

def create_and_save_canvas_plot(detailed_df, overall_df, metric_name, filename):
    """
    Generates the final 4x4 canvas. It relies on the pre-sorted
    DataFrames to ensure the x-axis items are in the correct sequence.
    """
    fig, axs = plt.subplots(4, 4, figsize=(24, 22))
    fig.suptitle(f'Human Evaluations - Metric: {metric_name}', fontsize=28, y=0.98)

    colors = {'OpenAI o4-mini-high': '#F93E3E', 'Gemini 2.5 pro': '#5CD167', 'Claude Opus 4': '#1F77B4'}
    model_legend_names = {'g4': 'OpenAI o4-mini-high', 'gemini': 'Gemini 2.5 pro', 'claude': 'Claude Opus 4'}
    language_order = ['C++', 'Python', 'Java', 'JavaScript']
    lang_code_map = {'C++': 'cpp', 'Python': 'python', 'Java': 'java', 'JavaScript': 'javascript'}

    # Updated language colors from 4cross4humanevals.py
    language_label_colors = {
        'C++': '#00FFFF',      # Cyan blue for C++
        'Python': '#3776AB',   # Python blue
        'Java': '#ed8b00',     # Java orange
        'JavaScript': '#f7df1e' # JavaScript yellow
    }

    g4_col = f'o4-mini-high - {metric_name} (count of 5s)'
    gemini_col = f'Gemini 2.5 Pro - {metric_name} (count of 5s)'
    claude_col = f'Claude Opus 4 - {metric_name} (count of 5s)'

    for row_idx, lang_name in enumerate(language_order):
        lang_code = lang_code_map[lang_name]

        # Filter the pre-sorted data for the current language
        lang_overall_data = overall_df[overall_df['Code Language'] == lang_code]
        lang_detailed_data = detailed_df[detailed_df['Code Language'] == lang_code]

        # Calculate a consistent Y-axis limit for the entire row
        row_max_y = 0
        if not lang_overall_data.empty:
            row_max_y = max(row_max_y, lang_overall_data[[g4_col, gemini_col, claude_col]].max().max())
        if not lang_detailed_data.empty:
            row_max_y = max(row_max_y, lang_detailed_data[[g4_col, gemini_col, claude_col]].max().max())
        common_y_limit = max(row_max_y + 2, 5)

        # --- Column 0: Overall Chart ---
        ax_overall = axs[row_idx, 0]
        if not lang_overall_data.empty:
            # The industries are already sorted by Prompt ID from the main block
            industries = lang_overall_data['Industry'].tolist()
            x = np.arange(len(industries))
            width = 0.25

            ax_overall.bar(x - width, lang_overall_data[g4_col], width, color=colors[model_legend_names['g4']])
            ax_overall.bar(x, lang_overall_data[gemini_col], width, color=colors[model_legend_names['gemini']])
            ax_overall.bar(x + width, lang_overall_data[claude_col], width, color=colors[model_legend_names['claude']])

            # Enhanced title with colored background
            # Use the language color as background (facecolor)
            lang_color = language_label_colors[lang_name]
            ax_overall.set_title('Overall', y=1.05, fontweight='bold', color='black', fontsize=16,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=lang_color, alpha=0.7))


            wrapped_labels = [textwrap.fill(l, 12) for l in industries]
            ax_overall.set_xticks(x)
            ax_overall.set_xticklabels(wrapped_labels, fontsize=10, ha='center')
            ax_overall.set_ylim(0, common_y_limit)
            ax_overall.set_yticks(np.arange(0, common_y_limit, step=max(1, int(common_y_limit/5))))
            ax_overall.grid(axis='y', linestyle='--', alpha=0.7)

        # --- Columns 1, 2, 3: Detailed Industry Charts ---
        # --- ENHANCEMENT: Get the ordered list of industries from the sorted overall data ---
        industries_in_lang = lang_overall_data['Industry'].tolist()

        for col_offset, industry_name in enumerate(industries_in_lang[:3]):
            ax = axs[row_idx, col_offset + 1]
            industry_slice = lang_detailed_data[lang_detailed_data['Industry'] == industry_name]

            # The subtopics are already sorted because 'lang_detailed_data' is sorted
            subtopics = industry_slice['Topic'].tolist()
            x = np.arange(len(subtopics))
            width = 0.25

            ax.bar(x - width, industry_slice[g4_col], width, color=colors[model_legend_names['g4']])
            ax.bar(x, industry_slice[gemini_col], width, color=colors[model_legend_names['gemini']])
            ax.bar(x + width, industry_slice[claude_col], width, color=colors[model_legend_names['claude']])

            # Use the language color as background (facecolor)
            lang_color = language_label_colors[lang_name]
            ax.set_title(industry_name, y=1.05, fontweight='bold', color='black', fontsize=16,
            bbox=dict(boxstyle='round,pad=0.3', facecolor=lang_color, alpha=0.7))


            wrapped_subtopics = [textwrap.fill(s, 15) for s in subtopics]
            ax.set_xticks(x)
            ax.set_xticklabels(wrapped_subtopics, rotation=0, ha='center', fontsize=8)
            ax.set_ylim(0, common_y_limit)
            ax.set_yticks(np.arange(0, common_y_limit, step=max(1, int(common_y_limit/5))))
            ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Add Y-axis labels for each subplot
    for row_idx, lang_name in enumerate(language_order):
        # First column: Y-axis label 'Industry'
        axs[row_idx, 0].set_xlabel('Industry', fontsize=12, fontweight='bold')

        # Columns 2,3,4: Y-axis label 'Topic'
        for col_idx in range(1, 4):
            if axs[row_idx, col_idx].has_data():  # Only if subplot has data
                axs[row_idx, col_idx].set_xlabel('Topic', fontsize=10, fontweight='bold')

    # --- Layout and Legend ---
    for ax in axs.flat:
        if not ax.has_data():
            ax.set_visible(False)

    plt.tight_layout(rect=[0.05, 0.08, 1, 0.96], h_pad=3.0)

    # Language labels (keep existing)
    for row_idx, lang_name in enumerate(language_order):
        pos = axs[row_idx, 0].get_position()
        y_fig_coord = pos.y0 + pos.height / 2
        x_fig_coord = 0.04
        lang_box_color = language_label_colors[lang_name]
        fig.text(x_fig_coord, y_fig_coord, lang_name, ha='center', va='center', rotation=90,
                fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", fc=lang_box_color, ec="black", lw=1, alpha=0.5))

    # ALIGNED LEGEND AND CHART GUIDE
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in model_legend_names.values()]

    # Use the same Y-coordinate and consistent vertical alignment
    shared_y_coord = 0.04  # Same level for both elements

    # Models legend - using 'center' vertical alignment
    legend = fig.legend(handles, model_legend_names.values(),
                    loc='center', bbox_to_anchor=(0.35, shared_y_coord),  # Changed to 'center'
                    ncol=3, title='Models', title_fontsize=14, fontsize=12,
                    frameon=True, fancybox=True, shadow=True,
                    edgecolor='gray', framealpha=1, facecolor='white')
    legend.get_frame().set_linewidth(1.5)

    # Chart guide - using 'center' vertical alignment to match legend
    guide_text_raw = "Chart Guide\nX-Axis: Count of '5' Ratings"
    fig.text(0.65, shared_y_coord, guide_text_raw, ha='center', va='center',  # Changed to 'center'
            fontsize=12, linespacing=1.5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='gray', lw=1.5))


    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f" Canvas saved: {filename}")

if __name__ == '__main__':
    csv_filepath = 'Untitled spreadsheet - Sheet1 (4).csv'

    # Use the new function to load, clean, AND sort the data
    df = load_clean_and_sort_data(csv_filepath)
    if df is not None:
        output_dir = "Final_Canvas_Graphs_Final_Theme"
        os.makedirs(output_dir, exist_ok=True)
        print(f"All canvas graphs will be saved in: '{output_dir}'")

        # --- KEY ENHANCEMENT: Aggregate data while keeping track of the first Prompt ID ---
        count_columns = [col for col in df.columns if '(count of 5s)' in col]
        agg_operations = {col: 'sum' for col in count_columns}
        agg_operations['Prompt ID'] = 'min'  # This finds the first Prompt ID for each industry group

        overall_agg_df = df.groupby(['Code Language', 'Industry']).agg(agg_operations).reset_index()

        # --- KEY ENHANCEMENT: Sort the aggregated industries by their first Prompt ID ---
        overall_agg_df.sort_values(by='Prompt ID', inplace=True)

        metrics_to_plot = ['Completeness', 'Correctness', 'Relevance']

        for metric in metrics_to_plot:
            print(f"\n--- Generating canvas for Metric: '{metric}' ---")
            filename = f"Canvas_{metric}.png"
            full_path = os.path.join(output_dir, filename)

            # Pass both the sorted detailed and sorted overall data to the plotting function
            create_and_save_canvas_plot(df, overall_agg_df, metric, full_path)
    else:
        print("Could not load data. Exiting.")