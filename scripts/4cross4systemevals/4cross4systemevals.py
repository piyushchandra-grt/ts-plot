import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import textwrap
import os

def get_overall_data_for_language(df, language):
    """
    Extract overall industry data for a specific language from the dataset.
    This replicates the logic from individualsystemoverall.py
    """
    # Filter data for the specific language
    lang_data = df[df['Code Language'] == language].copy()
    
    # Define industry mappings (from individualsystemoverall.py)
    language_industries = {
        'C++': ['Gaming', 'IoT', 'Security'],
        'Python': ['Machine Learning', 'FinTech', 'EdTech'],
        'Java': ['E‑Commerce', 'CRM', 'Hotel'],
        'JavaScript': ['Social Networking', 'Media', 'Streaming']
    }
    
    # Get industries for this language
    industries = language_industries.get(language, [])
    
    # Score columns (matching both scripts)
    score_cols = [
        'Chat GPT o4 mini-high Codebleu Score',
        'Gemini 2.5 pro Codebleu Score',
        'Claude Opus 4 Codebleu Score'
    ]
    
    # Collect industry data
    industry_data = []
    industry_labels = []
    
    for industry in industries:
        # Look for industry in Domain column
        industry_rows = lang_data[lang_data['Domain'] == industry]
        
        if len(industry_rows) > 0:
            # Take mean of all scores for this industry
            industry_scores = {}
            for col in score_cols:
                if col in industry_rows.columns:
                    scores = industry_rows[col].dropna()
                    if len(scores) > 0:
                        industry_scores[col] = scores.mean()
                    else:
                        industry_scores[col] = 0.0
                else:
                    industry_scores[col] = 0.0
            
            industry_data.append(industry_scores)
            industry_labels.append(industry)
    
    return industry_data, industry_labels

def create_enhanced_cross_language_domain_plot(csv_file):
    """
    Create a comprehensive cross-language domain comparison plot with
    overall graphs as the first subplot in each row
    """
    
    # Check if file exists
    if not os.path.exists(csv_file):
        print(f"❌ Error: File '{csv_file}' not found!")
        return None

    # Load the data
    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Successfully loaded {csv_file}")
        print(f"📊 Dataset shape: {df.shape}")
        print(f"🏷️ Available languages: {df['Code Language'].unique()}")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None

    # Language configuration
    languages = ['C++', 'Python', 'Java', 'JavaScript']
    
    language_domains = {
        'C++': ['Gaming', 'IoT', 'Security'],
        'Python': ['Machine Learning', 'FinTech', 'EdTech'],
        'Java': ['E‑Commerce', 'CRM', 'Hotel'],
        'JavaScript': ['Social Networking', 'Media', 'Streaming']
    }

    # Language colors
    language_colors = {
        "C++": "#00FFFF",      # Cyan
        "Python": "#3776AB",   # Python blue
        "JavaScript": "#F7DF1E", # JavaScript yellow
        "Java": "#ED8B00"      # Java orange
    }

    # Model configuration
    score_cols = [
        'Chat GPT o4 mini-high Codebleu Score',
        'Gemini 2.5 pro Codebleu Score',
        'Claude Opus 4 Codebleu Score'
    ]

    model_display_names = {
        'Chat GPT o4 mini-high Codebleu Score': 'o4-mini-high',
        'Gemini 2.5 pro Codebleu Score': 'Gemini 2.5 Pro',
        'Claude Opus 4 Codebleu Score': 'Claude Opus 4'
    }

    model_colors = {
        'Chat GPT o4 mini-high Codebleu Score': '#F93E3E', # Red
        'Gemini 2.5 pro Codebleu Score': '#5CD167',        # Green
        'Claude Opus 4 Codebleu Score': '#1F77B4'          # Blue
    }

    # Create figure - **INCREASED COLUMNS FOR OVERALL SUBPLOT**
    max_domains = max(len(domains) for domains in language_domains.values())
    n_rows = len(languages)
    n_cols = max_domains + 1  # +1 for overall subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(28, 20))  # Increased width
    fig.suptitle('System Evals - Cross-Language Industry-wise Comparison',
                 fontsize=20, fontweight='bold', y=0.98)

    # Ensure axes is 2D
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    # Track legend handles
    legend_handles = []
    legend_labels = []
    legend_created = False

    # Process each language (row)
    for row_idx, language in enumerate(languages):
        domains = language_domains[language]
        
        # Filter data for this language
        lang_data = df[df['Code Language'] == language].copy()
        print(f"🔍 Processing {language}: {len(lang_data)} records found")

        # **ENHANCED LANGUAGE HIGHLIGHT**
        # **ENHANCED LANGUAGE HIGHLIGHT** - MOVED MUCH FURTHER LEFT
        if n_cols > 0:
            axes[row_idx, 0].text(-0.18, 0.5, language,  # Changed from -0.15 to -0.25
                                transform=axes[row_idx, 0].transAxes,
                                fontsize=14, fontweight='bold',
                                rotation=90, va='center', ha='center',
                                color='black',
                                bbox=dict(boxstyle='round,pad=0.5',
                                            facecolor=language_colors.get(language, 'gray'),
                                            alpha=0.7, edgecolor='white', linewidth=1))


        # **CREATE OVERALL SUBPLOT (First column)**
        overall_ax = axes[row_idx, 0]
        
        # Get overall industry data for this language
        industry_data, industry_labels = get_overall_data_for_language(df, language)
        
        if industry_data and industry_labels:
            n_industries = len(industry_data)
            bar_width = 0.25
            x = np.arange(n_industries)

            # Create bars for overall subplot
            for i, col in enumerate(score_cols):
                values = [data.get(col, 0.0) for data in industry_data]
                x_offset = x + (i - 1) * bar_width
                bars = overall_ax.bar(x_offset, values,
                                    width=bar_width,
                                    label=model_display_names[col],
                                    color=model_colors[col],
                                    alpha=0.8, edgecolor='white', linewidth=1)

                # Collect legend info only once
                if not legend_created and row_idx == 0:
                    legend_handles.append(bars)
                    legend_labels.append(model_display_names[col])

            if not legend_created and row_idx == 0:
                legend_created = True

            # **OVERALL SUBPLOT STYLING**
            overall_ax.set_title('Overall', fontsize=11, fontweight='bold', pad=10,
                               bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor=language_colors[language],
                                         alpha=0.3))
            
            # Set x-axis for overall subplot
            overall_ax.set_xticks(x)
            wrapped_labels = ["\n".join(textwrap.wrap(str(label), 10))
                             for label in industry_labels]
            overall_ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=8)
            
            # **DIFFERENT AXIS LABELS FOR OVERALL**
            overall_ax.set_xlabel('Industries', fontsize=10)
            overall_ax.set_ylabel('CodeBLEU Mean Score', fontsize=10)
            
            # Grid and styling
            overall_ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
            overall_ax.set_axisbelow(True)
            
            # Enhanced subplot border
            for spine in overall_ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color('black')
        else:
            overall_ax.text(0.5, 0.5, f'No overall\ndata available',
                          ha='center', va='center', transform=overall_ax.transAxes,
                          fontsize=10, style='italic')
            overall_ax.set_title('Overall', fontsize=11, fontweight='bold', pad=10,
                               bbox=dict(boxstyle='round,pad=0.3',
                                         facecolor=language_colors[language],
                                         alpha=0.3))
            overall_ax.set_xlabel('Industries', fontsize=10)
            overall_ax.set_ylabel('CodeBLEU Mean Score', fontsize=10)

        # **PROCESS DOMAIN SUBPLOTS (Starting from column 1)**
        for col_idx, domain in enumerate(domains):
            ax = axes[row_idx, col_idx + 1]  # +1 to account for overall subplot

            # Domain filtering and processing (same as original)
            domain_data = lang_data[lang_data['Domain'] == domain].copy()
            print(f" 📂 {domain}: {len(domain_data)} records")

            if len(domain_data) == 0:
                ax.text(0.5, 0.5, f'No data\navailable',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, style='italic')
                ax.set_title(domain, fontsize=11, fontweight='bold', pad=10,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor=language_colors[language],
                                       alpha=0.3))
                # **DOMAIN SUBPLOT AXIS LABELS**
                ax.set_xlabel('Topics', fontsize=10)
                ax.set_ylabel('CodeBLEU Score', fontsize=10)
            else:
                # Get unique Topics for this specific language/domain
                Topics = domain_data['Subtopic'].unique().tolist()
                print(f" 📝 Topics for {language}-{domain}: {Topics}")

                # Ensure we have data for each subtopic
                filtered_data = []
                filtered_Topics = []
                for subtopic in Topics:
                    subtopic_data = domain_data[domain_data['Subtopic'] == subtopic].copy()
                    if len(subtopic_data) > 0:
                        filtered_data.append(subtopic_data.iloc[0])
                        filtered_Topics.append(subtopic)

                if not filtered_data:
                    ax.text(0.5, 0.5, f'No valid\nTopics',
                            ha='center', va='center', transform=ax.transAxes,
                            fontsize=10, style='italic')
                    ax.set_title(domain, fontsize=11, fontweight='bold', pad=10,
                                 bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor=language_colors[language],
                                           alpha=0.3))
                    ax.set_xlabel('Topics', fontsize=10)
                    ax.set_ylabel('CodeBLEU Score', fontsize=10)
                    continue

                # Convert to DataFrame for easier processing
                plot_data = pd.DataFrame(filtered_data)
                n_Topics = len(filtered_Topics)

                # Create bars
                bar_width = 0.25
                x = np.arange(n_Topics)

                for i, col in enumerate(score_cols):
                    if col in plot_data.columns:
                        values = plot_data[col].values
                        x_offset = x + (i - 1) * bar_width
                        bars = ax.bar(x_offset, values,
                                      width=bar_width,
                                      label=model_display_names[col],
                                      color=model_colors[col],
                                      alpha=0.8, edgecolor='white', linewidth=1)

                # Domain subplot styling
                ax.set_title(domain, fontsize=11, fontweight='bold', pad=10,
                             bbox=dict(boxstyle='round,pad=0.3',
                                       facecolor=language_colors[language],
                                       alpha=0.3))

                # Set x-axis with proper subtopic labels
                ax.set_xticks(x)
                wrapped_labels = ["\n".join(textwrap.wrap(str(label), 10))
                                 for label in filtered_Topics]
                ax.set_xticklabels(wrapped_labels, rotation=0, ha='center', fontsize=8)

                # Grid and styling
                ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
                ax.set_axisbelow(True)

                # Enhanced subplot border
                for spine in ax.spines.values():
                    spine.set_linewidth(1.0)
                    spine.set_color('black')

                # **DOMAIN SUBPLOT AXIS LABELS**
                ax.set_xlabel('Topics', fontsize=10)
                ax.set_ylabel('CodeBLEU Score', fontsize=10)

        # Hide extra subplots
        for col_idx in range(len(domains) + 1, n_cols):  # +1 for overall subplot
            axes[row_idx, col_idx].set_visible(False)

    # **ENHANCED CHART GUIDE**
    # **SIMPLIFIED LEGEND - MODELS ONLY, CENTERED**
    if legend_handles:
        # Models legend (centered in entire figure)
        models_legend = fig.legend(legend_handles, legend_labels,
                                loc='center', bbox_to_anchor=(0.5, 0.02),
                                fontsize=12, title='Models', title_fontsize=13,
                                frameon=True, fancybox=True, shadow=True,
                                ncol=3)

    # Enhanced layout
    plt.tight_layout(pad=4.0, h_pad=4.5, w_pad=2.0, rect=[0.08, 0.08, 0.95, 0.99])

    # Save and show
    output_file = 'enhanced_cross_language_analysis_with_overall.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Successfully saved: {output_file}")
    plt.show()

    return output_file

def main():
    """Main function with better error handling"""
    csv_file = 'RLHF data - System evals.csv'  # Update this path as needed
    print("🚀 Creating Enhanced Cross-Language Domain Comparison Plot with Overall Graphs...")
    print("=" * 80)

    try:
        output_file = create_enhanced_cross_language_domain_plot(csv_file)
        if output_file:
            print(f"✅ Successfully created: {output_file}")
            print("📊 Each language row now has an 'Overall' subplot as the first column!")
            print("🏷️ Overall subplots use 'Industries' and 'CodeBLEU Mean Score' as axis labels!")
            print("🏷️ Domain subplots use 'Topics' and 'CodeBLEU Score' as axis labels!")
        else:
            print("❌ Failed to create plot. Check the error messages above.")
    except Exception as e:
        print(f"❌ Error creating plot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()