import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.inter_rater import fleiss_kappa
from typing import List, Dict, Tuple
import os
from datetime import datetime

def generate_irr_report_and_visuals(
    evaluation_df: pd.DataFrame,
    models: List[str],
    dimensions: List[str],
    prompt_id_col: str = 'Prompt ID',
    rating_col_template: str = '{model} Human {dimension}',
    expected_raters_per_prompt: int = 3,
    save_figures: bool = True,
    output_dir: str = None
) -> Tuple[pd.DataFrame, Dict[str, plt.Figure], List[int]]:
    """
    Calculates Inter-Rater Reliability (IRR) and generates corresponding visualizations,
    after validating that each prompt has the expected number of ratings.

    This function produces:
    1. A report DataFrame with Fleiss's Kappa scores.
    2. A dictionary of matplotlib Figure objects for visualization.
    3. A list of Prompt IDs that were excluded from the analysis due to incorrect rating counts.
    
    Parameters:
    -----------
    save_figures : bool, default=True
        Whether to save generated figures to files
    output_dir : str, optional
        Directory to save figures. If None, creates timestamped folder
    """
    
    # --- Setup Output Directory ---
    if save_figures:
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"IRR_Analysis_Results_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Figures will be saved to: {output_dir}")
    
    def _create_safe_filename(base_name: str, extension: str = "png") -> str:
        """Create a safe filename by removing special characters"""
        import re
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', base_name)
        safe_name = re.sub(r'\s+', '_', safe_name)
        safe_name = safe_name.replace('.', '_')
        return f"{safe_name}.{extension}"
    
    def _save_figure(fig: plt.Figure, filename: str, description: str = ""):
        """Save figure with proper error handling"""
        if save_figures:
            try:
                filepath = os.path.join(output_dir, _create_safe_filename(filename))
                fig.savefig(filepath, dpi=300, bbox_inches='tight', 
                          facecolor='white', edgecolor='none')
                print(f"   ✅ Saved: {filename} {description}")
            except Exception as e:
                print(f"   ❌ Failed to save {filename}: {e}")
    
    # --- Nested Helper Functions ---
    def _preprocess_for_fleiss_kappa(df: pd.DataFrame, dimension_column: str, prompt_id: str) -> pd.DataFrame:
        if dimension_column not in df.columns: raise ValueError(f"Column '{dimension_column}' not found.")
        ratings_df = df[[prompt_id, dimension_column]].dropna(subset=[dimension_column])
        if ratings_df.empty: return pd.DataFrame()
        fleiss_matrix = ratings_df.groupby(prompt_id)[dimension_column].value_counts().unstack(fill_value=0)
        for i in range(1, 6):
            if i not in fleiss_matrix.columns: fleiss_matrix[i] = 0
        return fleiss_matrix.reindex(sorted(fleiss_matrix.columns), axis=1)

    def _calculate_disagreement_metrics(df: pd.DataFrame, models: List[str], dimensions: List[str], 
                                      pid_col: str, col_template: str) -> pd.DataFrame:
        """Calculate disagreement metrics for each prompt across all model-dimension combinations"""
        disagreement_data = []
        
        for _, prompt_group in df.groupby(pid_col):
            prompt_id = prompt_group[pid_col].iloc[0]
            
            for model in models:
                for dimension in dimensions:
                    col_name = col_template.format(model=model, dimension=dimension)
                    if col_name in df.columns:
                        ratings = prompt_group[col_name].dropna()
                        
                        if len(ratings) >= 2:  # Need at least 2 ratings to calculate disagreement
                            # Calculate disagreement metrics
                            ratings_list = list(ratings)
                            mean_rating = ratings.mean()
                            std_dev = ratings.std()
                            rating_range = ratings.max() - ratings.min()
                            coeff_variation = std_dev / mean_rating if mean_rating > 0 else 0
                            
                            # Count pairwise disagreements (difference >= 2 points)
                            major_disagreements = 0
                            total_pairs = 0
                            for i in range(len(ratings_list)):
                                for j in range(i+1, len(ratings_list)):
                                    total_pairs += 1
                                    if abs(ratings_list[i] - ratings_list[j]) >= 2:
                                        major_disagreements += 1
                            
                            disagreement_pct = (major_disagreements / total_pairs * 100) if total_pairs > 0 else 0
                            
                            # Classify disagreement level
                            if std_dev >= 1.5 or rating_range >= 3:
                                disagreement_level = "High"
                            elif std_dev >= 1.0 or rating_range >= 2:
                                disagreement_level = "Medium"
                            else:
                                disagreement_level = "Low"
                            
                            disagreement_data.append({
                                'Prompt_ID': prompt_id,
                                'Model': model,
                                'Dimension': dimension,
                                'Ratings': str(ratings_list),
                                'Mean_Rating': round(mean_rating, 2),
                                'Std_Dev': round(std_dev, 3),
                                'Range': rating_range,
                                'Coeff_Variation': round(coeff_variation, 3),
                                'Major_Disagreements_Pct': round(disagreement_pct, 1),
                                'Disagreement_Level': disagreement_level,
                                'Num_Raters': len(ratings)
                            })
        
        return pd.DataFrame(disagreement_data)
    
    def _create_disagreement_visualization(disagreement_df: pd.DataFrame, save_figures: bool, 
                                         output_dir: str, _save_figure) -> plt.Figure:
        """Create comprehensive visualization of disagreement patterns"""
        fig = plt.figure(figsize=(20, 12))
        
        # Define color scheme once for consistency
        disagreement_colors = {'Low': '#2E8B57', 'Medium': '#FFA500', 'High': '#DC143C'}
        
        # Create subplot layout
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1], 
                             hspace=0.3, wspace=0.3)
        
        # 1. Distribution of disagreement levels
        ax1 = fig.add_subplot(gs[0, 0])
        disagreement_counts = disagreement_df['Disagreement_Level'].value_counts()
        bars1 = ax1.bar(disagreement_counts.index, disagreement_counts.values, 
                       color=[disagreement_colors[x] for x in disagreement_counts.index], alpha=0.8)
        ax1.set_title('Distribution of Disagreement Levels', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Number of Model-Dimension Cases')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Standard deviation distribution
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(disagreement_df['Std_Dev'], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax2.axvline(disagreement_df['Std_Dev'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {disagreement_df["Std_Dev"].mean():.3f}')
        ax2.set_title('Distribution of Standard Deviations', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Standard Deviation of Ratings')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Disagreement by model
        ax3 = fig.add_subplot(gs[0, 2])
        model_disagreement = disagreement_df.groupby(['Model', 'Disagreement_Level']).size().unstack(fill_value=0)
        model_disagreement.plot(kind='bar', ax=ax3, color=list(disagreement_colors.values()), alpha=0.8)
        ax3.set_title('Disagreement Levels by Model', fontweight='bold', fontsize=12)
        ax3.set_xlabel('Model')
        ax3.set_ylabel('Count')
        ax3.legend(title='Disagreement Level')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Disagreement by dimension
        ax4 = fig.add_subplot(gs[1, 0])
        dim_disagreement = disagreement_df.groupby(['Dimension', 'Disagreement_Level']).size().unstack(fill_value=0)
        dim_disagreement.plot(kind='bar', ax=ax4, color=list(disagreement_colors.values()), alpha=0.8)
        ax4.set_title('Disagreement Levels by Dimension', fontweight='bold', fontsize=12)
        ax4.set_xlabel('Evaluation Dimension')
        ax4.set_ylabel('Count')
        ax4.legend(title='Disagreement Level')
        ax4.tick_params(axis='x', rotation=45)
        
        # 5. Scatter plot: Std Dev vs Range
        ax5 = fig.add_subplot(gs[1, 1])
        for level in ['Low', 'Medium', 'High']:
            subset = disagreement_df[disagreement_df['Disagreement_Level'] == level]
            ax5.scatter(subset['Std_Dev'], subset['Range'], 
                       c=disagreement_colors[level], label=level, alpha=0.7, s=30)
        ax5.set_title('Standard Deviation vs Range', fontweight='bold', fontsize=12)
        ax5.set_xlabel('Standard Deviation')
        ax5.set_ylabel('Rating Range (Max - Min)')
        ax5.legend(title='Disagreement Level')
        ax5.grid(True, alpha=0.3)
        
        # 6. High disagreement prompts (top 10)
        ax6 = fig.add_subplot(gs[1, 2])
        high_disagreement = disagreement_df[disagreement_df['Disagreement_Level'] == 'High'].nlargest(10, 'Std_Dev')
        if not high_disagreement.empty:
            bars6 = ax6.barh(range(len(high_disagreement)), high_disagreement['Std_Dev'], 
                            color=disagreement_colors['High'], alpha=0.8)
            ax6.set_yticks(range(len(high_disagreement)))
            ax6.set_yticklabels([f"P{pid}-{model[:8]}-{dim[:4]}" 
                                for pid, model, dim in zip(high_disagreement['Prompt_ID'], 
                                                          high_disagreement['Model'], 
                                                          high_disagreement['Dimension'])], 
                               fontsize=9)
            ax6.set_title('Top 10 Highest Disagreement Cases', fontweight='bold', fontsize=12)
            ax6.set_xlabel('Standard Deviation')
        else:
            ax6.text(0.5, 0.5, 'No High Disagreement Cases', ha='center', va='center', 
                    transform=ax6.transAxes, fontsize=12)
            ax6.set_title('Top 10 Highest Disagreement Cases', fontweight='bold', fontsize=12)
        
        # 7. Agreement statistics summary
        ax7 = fig.add_subplot(gs[2, :])
        ax7.axis('off')
        
        # Calculate summary statistics
        total_cases = len(disagreement_df)
        high_disagreement_count = len(disagreement_df[disagreement_df['Disagreement_Level'] == 'High'])
        medium_disagreement_count = len(disagreement_df[disagreement_df['Disagreement_Level'] == 'Medium'])
        low_disagreement_count = len(disagreement_df[disagreement_df['Disagreement_Level'] == 'Low'])
        
        avg_std = disagreement_df['Std_Dev'].mean()
        avg_range = disagreement_df['Range'].mean()
        avg_major_disagreements = disagreement_df['Major_Disagreements_Pct'].mean()
        
        unique_prompts_with_high_disagreement = disagreement_df[
            disagreement_df['Disagreement_Level'] == 'High']['Prompt_ID'].nunique()
        
        summary_text = f"""
        📊 DISAGREEMENT ANALYSIS SUMMARY
        
        📈 Overall Statistics:
        • Total Model-Dimension Cases: {total_cases}
        • Average Standard Deviation: {avg_std:.3f}
        • Average Rating Range: {avg_range:.1f}
        • Average Major Disagreements: {avg_major_disagreements:.1f}%
        
        🚨 Disagreement Breakdown:
        • High Disagreement Cases: {high_disagreement_count} ({high_disagreement_count/total_cases*100:.1f}%)
        • Medium Disagreement Cases: {medium_disagreement_count} ({medium_disagreement_count/total_cases*100:.1f}%)
        • Low Disagreement Cases: {low_disagreement_count} ({low_disagreement_count/total_cases*100:.1f}%)
        
        🎯 Prompts Requiring Review:
        • Unique Prompts with High Disagreement: {unique_prompts_with_high_disagreement}
        • Recommended Action: Review prompts with high disagreement for clarity and re-evaluate if necessary
        """
        
        ax7.text(0.05, 0.5, summary_text, transform=ax7.transAxes, fontsize=11,
                verticalalignment='center', bbox=dict(boxstyle="round,pad=0.5", 
                facecolor='lightgray', alpha=0.8))
        
        plt.suptitle('Inter-Rater Disagreement Analysis Dashboard', fontsize=16, fontweight='bold', y=0.98)
        
        if save_figures:
            _save_figure(fig, "05_Disagreement_Analysis_Dashboard", "(Comprehensive disagreement patterns)")
        
        return fig

    def _calculate_inter_rater_reliability(
        df: pd.DataFrame, model_list: List[str], dimension_list: List[str],
        pid_col: str, col_template: str
    ) -> pd.DataFrame:
        report_data = []
        for model in model_list:
            for dimension in dimension_list:
                dimension_col = col_template.format(model=model, dimension=dimension)
                if dimension_col not in df.columns:
                    print(f"Warning: Column '{dimension_col}' not found. Skipping IRR calculation.")
                    continue
                try:
                    fleiss_matrix = _preprocess_for_fleiss_kappa(df, dimension_col, pid_col)
                    if fleiss_matrix.empty or fleiss_matrix.sum(axis=1).min() < 2:
                        interpretation = "Skipped: Not enough ratings for a prompt."
                        kappa_value = "N/A"
                    else:
                        kappa = fleiss_kappa(fleiss_matrix.to_numpy())
                        kappa_value = f"{kappa:.4f}"
                        if kappa < 0: interpretation = "Poor agreement"
                        elif 0 <= kappa <= 0.20: interpretation = "Slight agreement"
                        elif 0.21 <= kappa <= 0.40: interpretation = "Fair agreement"
                        elif 0.41 <= kappa <= 0.60: interpretation = "Moderate agreement"
                        elif 0.61 <= kappa <= 0.80: interpretation = "Substantial agreement"
                        else: interpretation = "Almost perfect agreement"

                    report_data.append({"Model": model, "Dimension": dimension, "Fleiss's Kappa": kappa_value, "Interpretation": interpretation})
                except ValueError as e:
                    report_data.append({"Model": model, "Dimension": dimension, "Fleiss's Kappa": "N/A", "Interpretation": f"Error: {e}"})
        return pd.DataFrame(report_data)

    # --- Main Function Logic Starts Here ---
    
    # --- Data Summary ---
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)
    total_prompts = evaluation_df[prompt_id_col].nunique()
    total_ratings = len(evaluation_df)
    unique_prompt_ids = sorted(evaluation_df[prompt_id_col].unique())
    
    print(f"📊 Total unique prompts: {total_prompts}")
    print(f"📈 Total ratings: {total_ratings}")
    print(f"🔢 Prompt ID range: {min(unique_prompt_ids)} - {max(unique_prompt_ids)}")
    print(f"🎯 Expected raters per prompt: {expected_raters_per_prompt}")
    print(f"📋 Models being analyzed: {', '.join(models)}")
    print(f"📏 Dimensions being analyzed: {', '.join(dimensions)}")
    
    # --- Data Validation Step ---
    print("\n--- Validating Data: Checking for correct number of ratings per prompt ---")
    rating_counts = evaluation_df.groupby(prompt_id_col).size()
    invalid_prompt_ids = rating_counts[rating_counts != expected_raters_per_prompt].index.tolist()

    if invalid_prompt_ids:
        print(f"⚠️  Warning: {len(invalid_prompt_ids)} Prompt ID(s) do not have exactly {expected_raters_per_prompt} ratings and will be excluded:")
        if len(invalid_prompt_ids) <= 20:
            for pid in invalid_prompt_ids:
                print(f"   - Prompt ID {pid}: Found {rating_counts[pid]} ratings")
        else:
            print(f"   - First 10: {invalid_prompt_ids[:10]}")
            print(f"   - Last 10: {invalid_prompt_ids[-10:]}")
            print(f"   - (and {len(invalid_prompt_ids)-20} others)")
        validated_df = evaluation_df[~evaluation_df[prompt_id_col].isin(invalid_prompt_ids)].copy()
        print(f"✅ Using {len(validated_df)} validated ratings from {validated_df[prompt_id_col].nunique()} prompts for analysis")
    else:
        print(f"✅ Data validation successful: All prompts have exactly {expected_raters_per_prompt} ratings")
        validated_df = evaluation_df.copy()
    
    # Part 1: Fleiss's Kappa Calculation
    irr_report_df = _calculate_inter_rater_reliability(
        validated_df, models, dimensions, prompt_id_col, rating_col_template
    )
    
    # Save the report as CSV
    if save_figures:
        report_filename = os.path.join(output_dir, "IRR_Analysis_Report.csv")
        irr_report_df.to_csv(report_filename, index=False)
        print(f"📊 Saved analysis report: IRR_Analysis_Report.csv")
    
    figures = {}
    irr_report_df["Fleiss's Kappa Numeric"] = pd.to_numeric(irr_report_df["Fleiss's Kappa"], errors='coerce')

    # Graph 0: Overall IRR Heatmap (Enhanced)
    print("\nGenerating Overall IRR Heatmap...")
    if not irr_report_df.empty and not irr_report_df["Fleiss's Kappa Numeric"].isnull().all():
        try:
            heatmap_data = irr_report_df.pivot(index='Model', columns='Dimension', values="Fleiss's Kappa Numeric")
            heatmap_data = heatmap_data.reindex(index=models, columns=dimensions)
            
            # Enhanced figure size and styling
            fig0, ax0 = plt.subplots(figsize=(14, 8))
            
            # Create custom colormap for better interpretation
            sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", linewidths=1, 
                       ax=ax0, cbar_kws={'label': "Fleiss's Kappa Score"}, 
                       vmin=-0.1, vmax=0.8, center=0.4)
            
            # Enhanced title with interpretation guide
            ax0.set_title("Overall Inter-Rater Reliability (Fleiss's Kappa)\n" + 
                         "Green: Good Agreement (>0.4) | Yellow: Fair (0.2-0.4) | Red: Poor (<0.2)", 
                         fontsize=16, pad=25)
            ax0.set_xlabel("Evaluation Dimension", fontsize=14, fontweight='bold')
            ax0.set_ylabel("Model", fontsize=14, fontweight='bold')
            
            # Improve tick formatting
            ax0.set_xticklabels(ax0.get_xticklabels(), rotation=0, fontsize=12)
            ax0.set_yticklabels(ax0.get_yticklabels(), rotation=0, fontsize=12)
            
            plt.tight_layout()
            figures['overall_kappa_heatmap'] = fig0
            _save_figure(fig0, "01_Overall_Kappa_Heatmap", "(Summary of all models and dimensions)")
            print(" -> Done.")
        except Exception as e:
            print(f" -> Could not generate overall heatmap. Error: {e}")

    # Graph 1: Kappa Score Overview Bar Chart (Enhanced)
    print("\nGenerating Graph 1: Kappa Score Overview...")
    if not irr_report_df.empty:
        fig1, ax1 = plt.subplots(figsize=(16, 10))
        irr_report_df['Category'] = irr_report_df['Model'] + ' - ' + irr_report_df['Dimension']
        
        # Sort by kappa score for better visualization
        irr_report_df_sorted = irr_report_df.sort_values('Fleiss\'s Kappa Numeric', ascending=True)
        
        # Create horizontal bar plot with improved color scheme
        model_colors = {
            'Chat GPT o4-mini-high': '#2E8B57',    # Sea Green
            'Gemini 2.5 pro': '#4682B4',          # Steel Blue  
            'Claude Opus 4': '#8B4513'            # Saddle Brown
        }
        
        # Map colors to each bar based on model
        bar_colors = [model_colors.get(row['Model'], '#666666') for _, row in irr_report_df_sorted.iterrows()]
        
        bars = ax1.barh(range(len(irr_report_df_sorted)), 
                       irr_report_df_sorted['Fleiss\'s Kappa Numeric'],
                       color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        
        # Set y-tick labels
        ax1.set_yticks(range(len(irr_report_df_sorted)))
        ax1.set_yticklabels(irr_report_df_sorted['Category'], fontsize=11)
        
        # Add interpretation reference lines with better styling
        reference_lines = [
            (0.00, 'darkred', '-', 'Poor Agreement (< 0)'),
            (0.20, 'red', '--', 'Slight Agreement (≥ 0.20)'),
            (0.40, 'orange', '--', 'Fair Agreement (≥ 0.40)'),
            (0.60, 'green', '--', 'Moderate Agreement (≥ 0.60)'),
            (0.80, 'blue', '--', 'Substantial Agreement (≥ 0.80)')
        ]
        
        for value, color, style, label in reference_lines:
            ax1.axvline(value, color=color, linestyle=style, alpha=0.7, linewidth=2, label=label)
        
        # Enhanced title and labels
        ax1.set_title("Inter-Rater Agreement (Fleiss's Kappa) by Model and Dimension\n" +
                     "Bars sorted by agreement level - Higher values indicate better agreement", 
                     fontsize=16, pad=25, fontweight='bold')
        ax1.set_xlabel("Fleiss's Kappa Score", fontsize=14, fontweight='bold')
        ax1.set_ylabel("Model - Evaluation Dimension", fontsize=14, fontweight='bold')
        ax1.set_xlim(-0.1, 1.0)
        
        # Add consistent value labels with professional positioning
        
        for i, (bar, (_, row)) in enumerate(zip(bars, irr_report_df_sorted.iterrows())):
            width = bar.get_width()
            if pd.notna(width):
                # Consistent positioning: always place text to the right of bars
                # For very small bars, place at minimum readable distance
                label_text = f'{width:.3f}'
                
                if width < 0.15:  # For small bars
                    x_pos = 0.16  # Fixed position for readability
                    bbox_props = dict(boxstyle="round,pad=0.3", facecolor='white', 
                                    alpha=0.9, edgecolor='gray', linewidth=0.8)
                else:  # For larger bars
                    x_pos = width + 0.03
                    bbox_props = dict(boxstyle="round,pad=0.2", facecolor='white', 
                                    alpha=0.8, edgecolor='lightgray', linewidth=0.5)
                
                ax1.text(x_pos, bar.get_y() + bar.get_height()/2, label_text,
                        ha='left', va='center', fontsize=10, fontweight='bold',
                        bbox=bbox_props, color='black')
        
        # Create separate legends for models and thresholds
        # Model legend
        model_patches = [plt.Rectangle((0,0),1,1, facecolor=color, alpha=0.8, edgecolor='white') 
                        for color in model_colors.values()]
        model_legend = ax1.legend(model_patches, model_colors.keys(), 
                                title='Models', loc='upper right', 
                                bbox_to_anchor=(0.98, 0.98),
                                fontsize=10, title_fontsize=11,
                                frameon=True, fancybox=True, shadow=True)
        
        # Threshold legend  
        threshold_handles = [plt.Line2D([0], [0], color=color, linestyle=style, linewidth=2)
                           for _, color, style, _ in reference_lines]
        threshold_labels = [label for _, _, _, label in reference_lines]
        threshold_legend = ax1.legend(threshold_handles, threshold_labels,
                                    title='Agreement Thresholds', 
                                    loc='lower right', bbox_to_anchor=(0.98, 0.02),
                                    fontsize=9, title_fontsize=10,
                                    frameon=True, fancybox=True, shadow=True)
        
        # Add the model legend back (matplotlib removes previous legends)
        ax1.add_artist(model_legend)
        
        # Improve overall layout
        ax1.grid(axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
        ax1.set_axisbelow(True)
        
        # Adjust layout to prevent overlap
        plt.subplots_adjust(left=0.38, right=0.75, top=0.92, bottom=0.08)
        plt.tight_layout()
        figures['kappa_overview'] = fig1
        _save_figure(fig1, "02_Kappa_Overview_BarChart", "(Detailed comparison by model-dimension)")
        print(" -> Done.")

    # Graph 2: Rating Distribution Grouped Bar Charts
    print("\nGenerating Graph 2: Rating Distributions...")
    id_vars = [prompt_id_col]
    value_vars = [rating_col_template.format(model=m, dimension=d) for m in models for d in dimensions if rating_col_template.format(model=m, dimension=d) in validated_df.columns]
    
    if not value_vars:
        print(" -> Warning: No valid rating columns found for distribution plots.")
    else:
        melted_df = validated_df.melt(id_vars=id_vars, value_vars=value_vars, var_name='Category', value_name='Rating')
        melted_df[['Model', 'Human', 'Dimension']] = melted_df['Category'].str.rsplit(' ', n=2, expand=True)
        for dimension in dimensions:
            print(f" -> Plotting distribution for '{dimension}'...")
            fig2, ax2 = plt.subplots(figsize=(12, 7))
            dim_data = melted_df[melted_df['Dimension'] == dimension]
            if not dim_data.empty:
                sns.countplot(x='Model', hue='Rating', data=dim_data, ax=ax2, palette='viridis', order=models)
                ax2.set_title(f'Rating Distribution for "{dimension}" Dimension', fontsize=16)
                ax2.set_xlabel("Model", fontsize=12)
                ax2.set_ylabel("Count of Ratings", fontsize=12)
                ax2.legend(title='Rating Score', bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.xticks(rotation=10, ha='right')
                plt.tight_layout()
                figures[f'distribution_{dimension.lower()}'] = fig2
                _save_figure(fig2, f"03_Rating_Distribution_{dimension}", f"(Distribution for {dimension} dimension)")
            else:
                plt.close(fig2)
        print(" -> Done.")

    # Graph 3: Agreement Heatmaps per Prompt (Improved Layout)
    print("\nGenerating Graph 3: Agreement Heatmaps...")
    for model in models:
        for dimension in dimensions:
            dimension_col = rating_col_template.format(model=model, dimension=dimension)
            if dimension_col in validated_df.columns:
                print(f" -> Generating heatmap for '{model} - {dimension}'...")
                fleiss_matrix = _preprocess_for_fleiss_kappa(validated_df, dimension_col, prompt_id_col)
                if not fleiss_matrix.empty:
                    num_prompts = len(fleiss_matrix)
                    
                    # Intelligent figure sizing based on number of prompts
                    if num_prompts <= 10:
                        fig_height = 8
                        show_all_labels = True
                    elif num_prompts <= 30:
                        fig_height = max(10, num_prompts * 0.3)
                        show_all_labels = True
                    elif num_prompts <= 100:
                        fig_height = 15
                        show_all_labels = False
                        step = max(1, num_prompts // 20)  # Show ~20 labels max
                    else:
                        fig_height = 18
                        show_all_labels = False
                        step = max(1, num_prompts // 15)  # Show ~15 labels max for very large datasets
                    
                    fig3, ax3 = plt.subplots(figsize=(12, fig_height))
                    
                    # Create heatmap with improved formatting
                    if num_prompts <= 50:
                        sns.heatmap(fleiss_matrix, annot=True, cmap="YlGnBu", linewidths=.5, ax=ax3, fmt='d', 
                                  cbar_kws={'label': 'Number of Raters'})
                    else:
                        # For large datasets, remove annotations to reduce clutter
                        sns.heatmap(fleiss_matrix, annot=False, cmap="YlGnBu", linewidths=0.1, ax=ax3,
                                  cbar_kws={'label': 'Number of Raters'})
                    
                    # Improve y-axis labels
                    if not show_all_labels:
                        # Show only subset of prompt IDs
                        y_ticks = range(0, num_prompts, step)
                        y_labels = [fleiss_matrix.index[i] for i in y_ticks]
                        ax3.set_yticks([i + 0.5 for i in y_ticks])
                        ax3.set_yticklabels(y_labels, rotation=0, fontsize=9)
                    else:
                        # Show all labels but with better formatting
                        ax3.set_yticklabels(fleiss_matrix.index, rotation=0, fontsize=9)
                    
                    # Improve x-axis labels
                    ax3.set_xticklabels(fleiss_matrix.columns, rotation=0, fontsize=10)
                    
                    # Enhanced title and labels
                    ax3.set_title(f'Rater Agreement Heatmap: {model} - {dimension}\n({num_prompts} prompts analyzed)', 
                                fontsize=14, pad=20)
                    ax3.set_xlabel("Rating Score (1-5)", fontsize=12)
                    ax3.set_ylabel("Prompt ID", fontsize=12)
                    
                    # Add grid for better readability
                    ax3.grid(False)
                    
                    plt.tight_layout()
                    figures[f'heatmap_{model.replace(" ", "_")}_{dimension.lower()}'] = fig3
                    
                    # Create descriptive filename
                    safe_model = model.replace(" ", "_").replace("-", "_")
                    filename = f"04_Agreement_Heatmap_{safe_model}_{dimension}"
                    _save_figure(fig3, filename, f"({model} - {dimension})")
                else:
                    print(f"   -> No data available for {model} - {dimension}")
                    plt.close(fig3)
    print(" -> Done.")
    
    # Graph 5: Disagreement Analysis (New Addition)
    print("\nGenerating Graph 5: Disagreement Analysis...")
    disagreement_df = _calculate_disagreement_metrics(validated_df, models, dimensions, 
                                                     prompt_id_col, rating_col_template)
    
    if not disagreement_df.empty:
        # Create disagreement visualization
        disagreement_fig = _create_disagreement_visualization(disagreement_df, save_figures, 
                                                            output_dir, _save_figure)
        figures['disagreement_analysis'] = disagreement_fig
        
        # Save detailed disagreement data to CSV
        if save_figures:
            # Full disagreement analysis
            full_csv_path = os.path.join(output_dir, "Disagreement_Analysis_Full.csv")
            disagreement_df.to_csv(full_csv_path, index=False)
            print(f"   ✅ Saved: Disagreement_Analysis_Full.csv (All model-dimension combinations)")
            
            # High disagreement cases only
            high_disagreement = disagreement_df[disagreement_df['Disagreement_Level'] == 'High']
            if not high_disagreement.empty:
                high_csv_path = os.path.join(output_dir, "High_Disagreement_Cases.csv")
                high_disagreement.to_csv(high_csv_path, index=False)
                print(f"   ✅ Saved: High_Disagreement_Cases.csv ({len(high_disagreement)} high disagreement cases)")
            
            # Medium + High disagreement cases (action needed)
            action_needed = disagreement_df[disagreement_df['Disagreement_Level'].isin(['High', 'Medium'])]
            if not action_needed.empty:
                action_csv_path = os.path.join(output_dir, "Prompts_Requiring_Review.csv")
                action_needed.to_csv(action_csv_path, index=False)
                print(f"   ✅ Saved: Prompts_Requiring_Review.csv ({len(action_needed)} cases needing review)")
            
            # Summary by prompt (aggregated across all model-dimension combinations)
            prompt_summary = disagreement_df.groupby('Prompt_ID').agg({
                'Std_Dev': ['mean', 'max'],
                'Range': ['mean', 'max'], 
                'Major_Disagreements_Pct': ['mean', 'max'],
                'Disagreement_Level': lambda x: 'High' if 'High' in x.values else ('Medium' if 'Medium' in x.values else 'Low')
            }).round(3)
            
            # Flatten column names
            prompt_summary.columns = ['Avg_Std_Dev', 'Max_Std_Dev', 'Avg_Range', 'Max_Range', 
                                    'Avg_Major_Disagreements_Pct', 'Max_Major_Disagreements_Pct', 'Overall_Disagreement_Level']
            prompt_summary = prompt_summary.reset_index()
            
            # Add context information if available
            if 'Domain' in validated_df.columns:
                prompt_context = validated_df.groupby(prompt_id_col)['Domain'].first().reset_index()
                prompt_summary = prompt_summary.merge(prompt_context, left_on='Prompt_ID', right_on=prompt_id_col, how='left')
            
            prompt_summary_path = os.path.join(output_dir, "Prompt_Disagreement_Summary.csv")
            prompt_summary.to_csv(prompt_summary_path, index=False)
            print(f"   ✅ Saved: Prompt_Disagreement_Summary.csv (Summary by prompt ID)")
        
        # Print summary statistics
        total_cases = len(disagreement_df)
        high_count = len(disagreement_df[disagreement_df['Disagreement_Level'] == 'High'])
        medium_count = len(disagreement_df[disagreement_df['Disagreement_Level'] == 'Medium'])
        unique_prompts_high = disagreement_df[disagreement_df['Disagreement_Level'] == 'High']['Prompt_ID'].nunique()
        
        print(f"   📊 Disagreement Analysis Summary:")
        print(f"      • Total model-dimension cases analyzed: {total_cases}")
        print(f"      • High disagreement cases: {high_count} ({high_count/total_cases*100:.1f}%)")
        print(f"      • Medium disagreement cases: {medium_count} ({medium_count/total_cases*100:.1f}%)")
        print(f"      • Unique prompts with high disagreement: {unique_prompts_high}")
        print(f"      • Average standard deviation: {disagreement_df['Std_Dev'].mean():.3f}")
        print(" -> Done.")
    else:
        print("   ⚠️ No disagreement data available for analysis")
    
    # Create and save analysis summary
    if save_figures:
        _create_analysis_summary(output_dir, evaluation_df, validated_df, irr_report_df, 
                                models, dimensions, invalid_prompt_ids, prompt_id_col)
    
    return irr_report_df, figures, invalid_prompt_ids

def _create_analysis_summary(output_dir: str, original_df: pd.DataFrame, validated_df: pd.DataFrame, 
                           report_df: pd.DataFrame, models: List[str], dimensions: List[str], 
                           invalid_ids: List[int], prompt_id_col: str):
    """Create a comprehensive analysis summary file"""
    summary_path = os.path.join(output_dir, "Analysis_Summary.txt")
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("INTER-RATER RELIABILITY ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset Information
        f.write("DATASET INFORMATION:\n")
        f.write("-" * 40 + "\n")
        f.write(f"• Total unique prompts in original data: {original_df[prompt_id_col].nunique()}\n")
        f.write(f"• Total ratings in original data: {len(original_df)}\n")
        f.write(f"• Prompts used in analysis: {validated_df[prompt_id_col].nunique()}\n")
        f.write(f"• Ratings used in analysis: {len(validated_df)}\n")
        f.write(f"• Models analyzed: {', '.join(models)}\n")
        f.write(f"• Dimensions analyzed: {', '.join(dimensions)}\n")
        
        if invalid_ids:
            f.write(f"• Excluded prompts (insufficient ratings): {len(invalid_ids)}\n")
            f.write(f"  Excluded Prompt IDs: {invalid_ids}\n")
        f.write("\n")
        
        # Statistical Summary
        numeric_kappas = report_df["Fleiss's Kappa Numeric"].dropna()
        if not numeric_kappas.empty:
            f.write("STATISTICAL SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Average Kappa Score: {numeric_kappas.mean():.4f}\n")
            f.write(f"• Median Kappa Score: {numeric_kappas.median():.4f}\n")
            f.write(f"• Standard Deviation: {numeric_kappas.std():.4f}\n")
            f.write(f"• Range: {numeric_kappas.min():.4f} to {numeric_kappas.max():.4f}\n\n")
            
            # Agreement Distribution
            poor = sum(numeric_kappas < 0)
            slight = sum((numeric_kappas >= 0) & (numeric_kappas <= 0.20))
            fair = sum((numeric_kappas > 0.20) & (numeric_kappas <= 0.40))
            moderate = sum((numeric_kappas > 0.40) & (numeric_kappas <= 0.60))
            substantial = sum((numeric_kappas > 0.60) & (numeric_kappas <= 0.80))
            perfect = sum(numeric_kappas > 0.80)
            
            f.write("AGREEMENT LEVEL DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            f.write(f"• Poor Agreement (κ < 0.00):        {poor:2d} cases ({poor/len(numeric_kappas)*100:.1f}%)\n")
            f.write(f"• Slight Agreement (0.00-0.20):     {slight:2d} cases ({slight/len(numeric_kappas)*100:.1f}%)\n")
            f.write(f"• Fair Agreement (0.21-0.40):       {fair:2d} cases ({fair/len(numeric_kappas)*100:.1f}%)\n")
            f.write(f"• Moderate Agreement (0.41-0.60):   {moderate:2d} cases ({moderate/len(numeric_kappas)*100:.1f}%)\n")
            f.write(f"• Substantial Agreement (0.61-0.80): {substantial:2d} cases ({substantial/len(numeric_kappas)*100:.1f}%)\n")
            f.write(f"• Almost Perfect (κ > 0.80):        {perfect:2d} cases ({perfect/len(numeric_kappas)*100:.1f}%)\n\n")
        
        # Detailed Results
        f.write("DETAILED RESULTS BY MODEL AND DIMENSION:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Model':<25} {'Dimension':<15} {'Kappa':<10} {'Interpretation'}\n")
        f.write("-" * 60 + "\n")
        
        for _, row in report_df.iterrows():
            kappa_value = row["Fleiss's Kappa"]
            f.write(f"{row['Model']:<25} {row['Dimension']:<15} {kappa_value:<10} {row['Interpretation']}\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("DISAGREEMENT ANALYSIS METHODOLOGY:\n")
        f.write("-" * 50 + "\n")
        f.write("This analysis identifies prompts where human raters showed significant disagreement.\n")
        f.write("Understanding disagreement patterns helps improve evaluation quality and consistency.\n\n")
        
        f.write("DISAGREEMENT METRICS CALCULATED:\n")
        f.write("• Standard Deviation: Measures rating variability (higher = more disagreement)\n")
        f.write("• Range: Difference between highest and lowest ratings\n") 
        f.write("• Major Disagreements: Percentage of rating pairs differing by ≥2 points\n")
        f.write("• Coefficient of Variation: Normalized measure of relative variability\n\n")
        
        f.write("DISAGREEMENT LEVEL CLASSIFICATION:\n")
        f.write("• HIGH: Standard Deviation ≥1.5 OR Range ≥3 points\n")
        f.write("  → Requires immediate review and potential re-evaluation\n")
        f.write("• MEDIUM: Standard Deviation 1.0-1.5 OR Range 2 points\n") 
        f.write("  → Should be reviewed for clarity and consistency\n")
        f.write("• LOW: Standard Deviation <1.0 AND Range ≤1 point\n")
        f.write("  → Acceptable level of agreement\n\n")
        
        f.write("RECOMMENDED ACTIONS:\n")
        f.write("1. Review 'High_Disagreement_Cases.csv' for immediate attention\n")
        f.write("2. Use 'Prompts_Requiring_Review.csv' for systematic quality improvement\n")
        f.write("3. Analyze patterns in '05_Disagreement_Analysis_Dashboard.png'\n")
        f.write("4. Consider rater training for dimensions with frequent disagreements\n")
        f.write("5. Revise prompts that consistently show high disagreement\n\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("GENERATED FILES:\n")
        f.write("-" * 20 + "\n")
        f.write("• IRR_Analysis_Report.csv - Detailed numerical results\n")
        f.write("• 01_Overall_Kappa_Heatmap.png - Summary heatmap\n")
        f.write("• 02_Kappa_Overview_BarChart.png - Detailed comparison\n")
        for dim in dimensions:
            f.write(f"• 03_Rating_Distribution_{dim}.png - {dim} rating patterns\n")
        for model in models:
            for dim in dimensions:
                safe_model = model.replace(" ", "_").replace("-", "_")
                f.write(f"• 04_Agreement_Heatmap_{safe_model}_{dim}.png - {model} {dim} agreement\n")
        f.write("• 05_Disagreement_Analysis_Dashboard.png - Comprehensive disagreement analysis\n")
        f.write("• Disagreement_Analysis_Full.csv - Complete disagreement metrics\n")
        f.write("• High_Disagreement_Cases.csv - Cases requiring immediate attention\n") 
        f.write("• Prompts_Requiring_Review.csv - Medium and high disagreement cases\n")
        f.write("• Prompt_Disagreement_Summary.csv - Aggregated disagreement by prompt\n")
        f.write("• Analysis_Summary.txt - This summary file\n")
        
    print(f"📋 Saved comprehensive summary: Analysis_Summary.txt")

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    try:
        main_df = pd.read_csv('Copy of RLHF data - Final rubrics.csv')
        print("Successfully loaded 'Copy of RLHF data - Final rubrics.csv'")
    except FileNotFoundError:
        print("CSV file not found. Using sample data for demonstration.")
        sample_data = {
            'Prompt ID': [380, 380, 380, 381, 381, 381, 382, 382, 382, 383, 383],
            'Chat GPT o4-mini-high Human Correctness':   [4, 3, 4, 4, 5, 4, 5, 5, 5, 4, 4],
            'Chat GPT o4-mini-high Human Relevance':      [5, 4, 5, 5, 5, 4, 4, 3, 4, 5, 5],
            'Chat GPT o4-mini-high Human Completeness':   [2, 3, 3, 5, 4, 5, 4, 4, 4, 3, 3],
            'Gemini 2.5 pro Human Correctness':           [5, 4, 3, 4, 4, 4, 3, 3, 3, 4, 3],
            'Gemini 2.5 pro Human Relevance':             [4, 4, 5, 5, 5, 5, 2, 3, 2, 5, 4],
            'Gemini 2.5 pro Human Completeness':          [5, 5, 5, 4, 4, 4, 3, 4, 3, 4, 4],
            'Claude Opus 4 Human Correctness':            [5, 5, 4, 3, 3, 4, 5, 5, 5, 5, 4],
            'Claude Opus 4 Human Relevance':              [5, 5, 5, 4, 4, 4, 1, 1, 2, 4, 4],
            'Claude Opus 4 Human Completeness':           [5, 5, 4, 5, 5, 4, 2, 2, 3, 5, 5],
        }
        main_df = pd.DataFrame(sample_data)

    models_to_analyze = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
    dimensions_to_analyze = ['Correctness', 'Relevance', 'Completeness']

    report, generated_figures, invalid_ids = generate_irr_report_and_visuals(
        evaluation_df=main_df,
        models=models_to_analyze,
        dimensions=dimensions_to_analyze,
        save_figures=True,  # Enable automatic saving
        output_dir=None     # Use auto-generated timestamped folder
    )

    print("\n" + "="*70)
    print("INTER-RATER RELIABILITY ANALYSIS RESULTS")
    print("="*70)
    print(report.to_string(index=False))
    
    # Add interpretation summary
    print("\n" + "-"*50)
    print("📊 RESULTS INTERPRETATION GUIDE:")
    print("-"*50)
    print("🔴 Poor Agreement:       κ < 0.00  (Red in heatmaps)")
    print("🟡 Slight Agreement:     0.00 ≤ κ ≤ 0.20  (Light colors)")
    print("🟠 Fair Agreement:       0.21 ≤ κ ≤ 0.40  (Yellow/Orange)")
    print("🟢 Moderate Agreement:   0.41 ≤ κ ≤ 0.60  (Light green)")
    print("🟢 Substantial Agreement: 0.61 ≤ κ ≤ 0.80  (Green)")
    print("🟢 Almost Perfect:       κ > 0.80  (Dark green)")
    
    # Summary of results
    numeric_kappas = report["Fleiss's Kappa Numeric"].dropna()
    if not numeric_kappas.empty:
        print(f"\n📈 SUMMARY STATISTICS:")
        print(f"   • Average Kappa: {numeric_kappas.mean():.3f}")
        print(f"   • Median Kappa:  {numeric_kappas.median():.3f}")
        print(f"   • Range: {numeric_kappas.min():.3f} to {numeric_kappas.max():.3f}")
        
        # Count agreements by level
        poor = sum(numeric_kappas < 0)
        slight = sum((numeric_kappas >= 0) & (numeric_kappas <= 0.20))
        fair = sum((numeric_kappas > 0.20) & (numeric_kappas <= 0.40))
        moderate = sum((numeric_kappas > 0.40) & (numeric_kappas <= 0.60))
        substantial = sum((numeric_kappas > 0.60) & (numeric_kappas <= 0.80))
        perfect = sum(numeric_kappas > 0.80)
        
        print(f"\n📋 AGREEMENT DISTRIBUTION:")
        print(f"   • Poor:        {poor} cases")
        print(f"   • Slight:      {slight} cases")
        print(f"   • Fair:        {fair} cases")
        print(f"   • Moderate:    {moderate} cases")
        print(f"   • Substantial: {substantial} cases")
        print(f"   • Perfect:     {perfect} cases")
    
    if invalid_ids:
        print(f"\n⚠️  NOTE: {len(invalid_ids)} Prompt ID(s) were excluded from analysis due to insufficient ratings")
        if len(invalid_ids) <= 10:
            print(f"   Excluded IDs: {invalid_ids}")
        else:
            print(f"   First few excluded IDs: {invalid_ids[:5]}... (and {len(invalid_ids)-5} others)")
        
    print("\n" + "="*70)

    if generated_figures:
        print(f"🎨 VISUALIZATIONS: Generated and saved {len(generated_figures)} graphs...")
        print("   📊 Graph types generated:")
        for name, fig in generated_figures.items():
            print(f"      • {name.replace('_', ' ').title()}")
        print("\n💡 TIP: Look for patterns in the heatmaps where darker colors indicate higher agreement")
        print("📁 All files have been saved to the output directory for future reference")
        
        # Highlight disagreement analysis if available
        if 'disagreement_analysis' in generated_figures:
            print(f"\n🔍 DISAGREEMENT ANALYSIS HIGHLIGHTS:")
            print(f"   📈 New comprehensive dashboard shows disagreement patterns")
            print(f"   📋 CSV files exported for prompts requiring review:")
            print(f"      • High_Disagreement_Cases.csv - Immediate attention needed")
            print(f"      • Prompts_Requiring_Review.csv - Quality improvement candidates") 
            print(f"      • Prompt_Disagreement_Summary.csv - Aggregated by prompt")
            print(f"   🎯 Use these files to improve evaluation consistency and quality")
        
        plt.show()
    else:
        print("❌ No figures were generated.")