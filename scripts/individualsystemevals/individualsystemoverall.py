# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import textwrap
import os

def diagnose_data_issue(csv_file):
    """Diagnose why OpenAI o4-mini-high isn't showing"""
    
    print("🔍 DIAGNOSING DATA ISSUES...")
    print("="*60)
    
    # Load the data
    df = pd.read_csv(csv_file)
    df['Language'] = df['Language'].fillna(method='ffill')
    
    print("1. AVAILABLE COLUMNS:")
    print(df.columns.tolist())
    
    print("\n2. COLUMNS CONTAINING 'o4' or 'OpenAI':")
    openai_cols = [col for col in df.columns if 'o4' in col.lower() or 'openai' in col.lower()]
    print(openai_cols)
    
    print("\n3. SAMPLE DATA:")
    print(df.head())
    
    print("\n4. DATA BY LANGUAGE:")
    for lang in ['C++', 'Python', 'Java', 'JavaScript']:
        lang_data = df[df['Language'] == lang]
        if len(lang_data) > 0:
            print(f"\n{lang}:")
            for col in openai_cols:
                if col in lang_data.columns:
                    values = lang_data[col].values
                    print(f"  {col}: {values}")
                    print(f"    - Min: {np.min(values)}, Max: {np.max(values)}")
                    print(f"    - Has nulls: {lang_data[col].isnull().any()}")
    
    return openai_cols

def create_fixed_language_industry_plots(csv_file):
    """
    Create plots with automatic column detection and debugging
    """
    
    # First diagnose the issue
    openai_cols = diagnose_data_issue(csv_file)
    
    if not openai_cols:
        print("❌ ERROR: No OpenAI o4-mini-high columns found!")
        print("Please check your CSV column names.")
        return []
    
    # Use the first available OpenAI column
    openai_col = openai_cols[0]
    print(f"\n✅ Using OpenAI column: '{openai_col}'")
    
    # Load the data
    df = pd.read_csv(csv_file)
    df['Language'] = df['Language'].fillna(method='ffill')
    
    # Define model columns - adjust based on your actual data
    score_cols = [
        openai_col,  # Use the detected OpenAI column
        'Gemini 2.5 Pro',
        'Claude Opus 4'
    ]
    
    # Check if other columns exist, if not try alternatives
    if 'Gemini 2.5 Pro' not in df.columns:
        gemini_cols = [col for col in df.columns if 'gemini' in col.lower()]
        if gemini_cols:
            score_cols[1] = gemini_cols[0]
    
    if 'Claude Opus 4' not in df.columns:
        claude_cols = [col for col in df.columns if 'claude' in col.lower()]
        if claude_cols:
            score_cols[2] = claude_cols[0]
    
    print(f"Using columns: {score_cols}")
    
    languages = ['C++', 'Python', 'Java', 'JavaScript']
    language_industries = {
        'C++': ['Gaming', 'IoT', 'Security'],
        'Python': ['Machine Learning', 'FinTech', 'EdTech'],
        'Java': ['E-Commerce', 'CRM', 'Hotel'],
        'JavaScript': ['Social Networking', 'Media', 'Streaming']
    }
    
    # Colors for models
    model_colors = {
        openai_col: '#F93E3E',        # Red
        'Gemini 2.5 Pro': '#5CD167',  # Green
        'Claude Opus 4': '#1F77B4'    # Blue
    }
    
    # Update colors for any alternative column names
    for col in score_cols:
        if col not in model_colors:
            if 'openai' in col.lower() or 'o4' in col.lower():
                model_colors[col] = '#F93E3E'
            elif 'gemini' in col.lower():
                model_colors[col] = '#5CD167'
            elif 'claude' in col.lower():
                model_colors[col] = '#1F77B4'
    
    model_display_names = {
        col: col.replace('OpenAI ', '').replace('o4-mini-high', 'o4-mini-high') 
        for col in score_cols
    }
    
    # Create output directory
    output_dir = "language_industry_plots"
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    
    # Process each language
    for language in languages:
        print(f"\nCreating plot for {language}...")
        
        lang_data = df[df['Language'] == language].copy()
        if len(lang_data) == 0:
            continue
        
        fig, ax = plt.subplots(figsize=(10, 8))
        industries = language_industries[language]
        
        # Collect data with detailed logging
        industry_data = []
        industry_labels = []
        
        for industry in industries:
            industry_row = lang_data[lang_data['Industry'] == industry]
            if len(industry_row) > 0:
                data_point = {'industry': industry}
                
                print(f"  {industry}:")
                for col in score_cols:
                    if col in industry_row.columns:
                        value = float(industry_row[col].iloc[0])
                        data_point[col] = value
                        print(f"    {col}: {value}")
                    else:
                        data_point[col] = 0.0
                        print(f"    {col}: MISSING - using 0.0")
                
                industry_data.append(data_point)
                industry_labels.append(industry)
        
        if industry_data:
            n_industries = len(industry_data)
            bar_width = 0.25
            x = np.arange(n_industries)
            
            # Create bars with explicit debugging
            for i, model_col in enumerate(score_cols):
                values = [data.get(model_col, 0.0) for data in industry_data]
                x_offset = x + (i - 1) * bar_width
                
                print(f"    Plotting {model_col}: {values}")
                
                bars = ax.bar(x_offset, values, width=bar_width,
                             label=model_display_names.get(model_col, model_col),
                             color=model_colors.get(model_col, '#888888'),
                             alpha=0.8, edgecolor='white', linewidth=1)
                
                # Add value labels
                for j, bar in enumerate(bars):
                    height = bar.get_height()
                    if height > 0:  # Only label non-zero bars
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                               f'{height:.3f}', ha='center', va='bottom', fontsize=9)
            
            # Customize plot
            ax.set_title(f'System Evals - {language} Industry-wise Comparison', 
                        fontsize=16, fontweight='bold', pad=20)
            ax.set_xticks(x)
            ax.set_xticklabels(industry_labels, rotation=0, ha='center', fontsize=12)
            ax.set_xlabel('Industry', fontsize=13, fontweight='bold')
            ax.set_ylabel('CodeBLEU Mean Score', fontsize=13, fontweight='bold')
            ax.set_ylim(0, 0.6)
            ax.grid(True, axis='y', linestyle='-', alpha=0.3, color='gray')
            ax.legend(title='Models',loc='center', bbox_to_anchor=(0.5, -0.15), ncol=3)
        
        plt.tight_layout(rect=[0, 0.1, 1, 1])
        
        safe_language = language.replace('+', 'plus')
        filename = f"{safe_language}_industry_comparison_fixed.png"
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        generated_files.append(filepath)
        print(f"✅ Generated: {filepath}")
    
    return generated_files

def main():
    """Main function with enhanced debugging"""
    csv_file = 'sys mean.csv'
    
    print("🚀 Starting Enhanced Language-Industry Plot Generation...")
    print("🔧 With OpenAI o4-mini-high debugging...")
    
    try:
        generated_files = create_fixed_language_industry_plots(csv_file)
        
        if generated_files:
            print(f"\n🎉 Generated {len(generated_files)} plots successfully!")
            print("📍 Check the console output above to see:")
            print("   - Which OpenAI column was used")
            print("   - Actual values for each model/industry")
            print("   - Any missing data issues")
        else:
            print("❌ No plots generated. Check the diagnostic output above.")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()