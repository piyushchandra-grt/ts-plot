# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Rectangle
import textwrap
import os

def create_individual_codebleu_plots(csv_file):
    """
    Create individual CodeBleu analysis plots for each language-domain combination.
    Each plot will be saved as a separate image file with a centered models legend below the plot.
    """
    # Load the data
    df = pd.read_csv(csv_file)
    
    # Define the structure
    languages = ['Python', 'JavaScript', 'Java', 'C++']
    language_domains = {
        'Python': ['Machine Learning', 'FinTech', 'EdTech'],
        'JavaScript': ['Social Networking', 'Media', 'Streaming'],
        'Java': ['E‑Commerce', 'CRM', 'Hotel'],
        'C++': ['Gaming', 'IoT', 'Security']
    }
    
    # Score columns and their display names
    score_cols = [
        'Chat GPT o4 mini-high Codebleu Score',
        'Gemini 2.5 pro Codebleu Score',
        'Claude Opus 4 Codebleu Score'
    ]
    
    model_display_names = {
        'Chat GPT o4 mini-high Codebleu Score': 'o4-mini-high',
        'Gemini 2.5 pro Codebleu Score': 'Gemini 2.5 pro',
        'Claude Opus 4 Codebleu Score': 'Claude Opus 4'
    }
    
    # Colors for models
    model_colors = {
        'Chat GPT o4 mini-high Codebleu Score': '#F93E3E',  # Red
        'Gemini 2.5 pro Codebleu Score': '#5CD167',        # Green
        'Claude Opus 4 Codebleu Score': '#1F77B4'          # Blue
    }
    
    # Colors for language highlighting
    language_colors = {
        "Python": "#3776ab",
        "JavaScript": "#f7df1e",
        "Java": "#ed8b00",
        "C++": "#00FFFF"
    }
    
    # Create output directory if it doesn't exist
    output_dir = "individual_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # Process each language
    for language in languages:
        # Filter data for this language
        lang_data = df[df['Code Language'] == language].copy()
        
        if len(lang_data) == 0:
            print(f"Warning: No data found for {language}")
            continue
            
        domains = language_domains[language]
        
        # Process each domain
        for domain in domains:
            # Create individual figure for this language-domain combination
            # Maintained figure size for single centered legend
            fig, ax = plt.subplots(figsize=(10, 9))
            
            # Filter data for this domain
            domain_data = lang_data[lang_data['Domain'] == domain].copy()
            
            if len(domain_data) == 0:
                ax.text(0.5, 0.5, f'No data\navailable for\n{language} - {domain}', 
                       ha='center', va='center', transform=ax.transAxes, 
                       fontsize=14, style='italic')
                ax.set_title(f'{language} - {domain}', fontsize=16, fontweight='bold', 
                           pad=20, bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=language_colors.get(language, 'lightgray'), 
                           alpha=0.3))
                ax.axis('off')
            else:
                # Get subtopics for x-axis
                subtopics = domain_data['Subtopic'].tolist()
                n_subtopics = len(subtopics)
                
                # Set up bar positions
                bar_width = 0.25
                x = np.arange(n_subtopics)
                
                # Create grouped bars for each model
                for i, col in enumerate(score_cols):
                    values = domain_data[col].values
                    x_offset = x + (i - 1) * bar_width
                    
                    bars = ax.bar(x_offset, values, width=bar_width, 
                                 label=model_display_names[col],
                                 color=model_colors[col], alpha=0.8, 
                                 edgecolor='white', linewidth=1)
                
                # Customize the plot
                ax.set_title(f'{language} - {domain}', fontsize=16, fontweight='bold', 
                           pad=20, bbox=dict(boxstyle='round,pad=0.5', 
                           facecolor=language_colors.get(language, 'lightgray'), 
                           alpha=0.3))
                
                # Set x-axis
                ax.set_xticks(x)
                def wrap_label(label, wrap_width=15):
                    return "\n".join(textwrap.wrap(label, wrap_width))
                
                truncated_and_wrapped_labels = [wrap_label(label, 15) for label in subtopics]
                ax.set_xticklabels(truncated_and_wrapped_labels, rotation=0, 
                                  ha='center', fontsize=10)
                ax.set_xlabel('Topics', fontsize=12, fontweight='bold')
                
                # Set y-axis
                ax.set_ylim(0, 0.6)
                ax.set_ylabel('CodeBLEU Score', fontsize=12, fontweight='bold')
                
                # Grid styling
                ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
                ax.set_axisbelow(True)
                
                # UPDATED: Single centered models legend below the plot with more spacing
                # Models legend - centered below the plot with increased spacing from x-axis label
                legend1 = ax.legend(loc='center', bbox_to_anchor=(0.5, -0.25), 
                                  borderaxespad=0, fontsize=11, title='Models', 
                                  title_fontsize=12, frameon=True, fancybox=True, 
                                  shadow=True, ncol=3)  # ncol=3 for horizontal layout
                
                # REMOVED: Chart guide legend (no longer needed since axes are labeled)
            
            # UPDATED: Adjust layout for single centered legend with more bottom space
            # Increased bottom margin to accommodate the legend with more spacing
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            
            # Generate filename and save
            safe_language = language.replace('+', 'plus')
            safe_domain = domain.replace(' ', '_').replace('‑', '-')
            filename = f"{safe_language}_{safe_domain}_codebleu.png"
            filepath = os.path.join(output_dir, filename)
            
            # Use bbox_inches='tight' to ensure legend is included in saved image
            plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Close figure to free memory
            
            generated_files.append(filepath)
            print(f"✅ Generated: {filepath}")
    
    return generated_files

def print_generation_summary(generated_files):
    """Print a summary of generated files"""
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Total plots generated: {len(generated_files)}")
    print(f"Output directory: individual_plots/")
    print("\nGenerated files:")
    for i, filepath in enumerate(generated_files, 1):
        filename = os.path.basename(filepath)
        print(f"  {i:2d}. {filename}")

def main():
    """Main function to run the individual plot analysis"""
    csv_file = 'RLHF data - System evals.csv'  # Update this path as needed
    
    print("🚀 Starting Individual CodeBleu Plot Generation...")
    
    try:
        generated_files = create_individual_codebleu_plots(csv_file)
        print_generation_summary(generated_files)
        print("\n🎉 All individual plots generated successfully!")
        print("📍 Models legend is now centered below plots with increased spacing!")
        
    except Exception as e:
        print(f"❌ Error creating individual plots: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()