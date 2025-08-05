import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from statsmodels.stats.inter_rater import fleiss_kappa
import pandas as pd
from collections import Counter
import seaborn as sns


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_and_process_data(data_filename="RLHF data - Main RLHF.csv", language="Python", missing_strategy="skip"):
    """
    Load the data file and extract rating data for analysis of multiple models by language and domain
   
    Parameters:
    data_filename: str - Name of the data file in the same directory (CSV or Excel)
    language: str - Programming language to filter by ('Python', 'javascript', 'Java')
    missing_strategy: str - How to handle missing values: 'skip', 'impute_mean', 'impute_median', 'report_only'
   
    Returns:
    processed_data: dict - Dictionary containing rating data organized by domain and category
    """
   
    try:
        # Load the data file (CSV or Excel based on extension)
        if data_filename.endswith('.csv'):
            df = pd.read_csv(data_filename)
        else:
            df = pd.read_excel(data_filename)
           
        df.columns = df.columns.str.strip()  # Remove leading/trailing spaces from column names
        print(f"✅ Successfully loaded {data_filename}")
        print(f"📊 Dataset shape: {df.shape}")
       
        # Filter by programming language
        if 'Code Language' in df.columns:
            df = df[df['Code Language'] == language].copy()
            print(f"🔍 Filtered for {language}: {len(df)} records")
        else:
            print("⚠️  No 'Code Language' column found, using all data")
       
        # Data Quality Analysis
        print(f"\n🔍 DATA QUALITY ANALYSIS:")
        print(f"   Total rows after filtering: {len(df)}")
        missing_any = df.isnull().any(axis=1).sum()
        complete_rows = len(df) - missing_any
        print(f"   Rows with missing data: {missing_any}")
        print(f"   Complete rows: {complete_rows}")
       
        # Get unique domains for this language
        domains = df['Domain'].dropna().unique().tolist() if 'Domain' in df.columns else []
        print(f"   Available domains: {domains}")
       
        if missing_any > 0:
            print(f"\n⚠️  MISSING DATA DETECTED:")
            missing_rows = df[df.isnull().any(axis=1)]
            for idx, row in missing_rows.iterrows():
                missing_cols = row.isnull()
                missing_list = [col for col, is_missing in missing_cols.items() if is_missing]
                print(f"   Row {idx}: Missing {len(missing_list)} values")
       
        # Define the models and rating categories with new column naming
        models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
        # Display names for legends (what user wants to see in charts)
        model_display_names = {
            'Chat GPT o4-mini-high': 'o4-mini-high',
            'Gemini 2.5 pro': 'Gemini 2.5 pro',
            'Claude Opus 4': 'Claude Opus 4'
        }
        rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
       
        # Initialize processed data structure: {domain: {category: {model: [ratings]}}}
        processed_data = {'Overall': {}}
       
        # Add domain-specific data structures
        for domain in domains:
            processed_data[domain] = {}
       
        # Initialize category structures
        for domain_key in processed_data.keys():
            for category in rating_categories:
                processed_data[domain_key][category] = {}
                for model in models:
                    processed_data[domain_key][category][model] = []
       
        # Process Overall data (all domains combined)
        print(f"\n📊 PROCESSING OVERALL DATA:")
        for category in rating_categories:
            for model in models:
                # Construct column name: e.g., "Chat GPT o4-mini-high Human Correctness"
                column_name = f"{model} Human {category}"
               
                if column_name in df.columns:
                    # Extract ratings based on missing value strategy
                    if missing_strategy == "skip":
                        ratings = df[column_name].dropna()
                    elif missing_strategy == "impute_mean":
                        ratings = df[column_name].fillna(df[column_name].mean())
                    elif missing_strategy == "impute_median":
                        ratings = df[column_name].fillna(df[column_name].median())
                    elif missing_strategy == "report_only":
                        ratings = df[column_name]
                    else:
                        ratings = df[column_name].dropna()
                   
                    # Convert to numeric and validate (only for non-reporting mode)
                    if missing_strategy != "report_only":
                        ratings = pd.to_numeric(ratings, errors='coerce').dropna()
                        ratings = ratings.round().astype(int)
                        ratings = ratings[(ratings >= 1) & (ratings <= 5)]
                   
                    processed_data['Overall'][category][model] = ratings.tolist()
                    print(f"   {category} - {model}: {len(ratings)} ratings")
                else:
                    print(f"⚠️  Column '{column_name}' not found")
       
        # Process Domain-specific data
        print(f"\n📊 PROCESSING DOMAIN-SPECIFIC DATA:")
        for domain in domains:
            print(f"\n🏷️  Domain: {domain}")
            domain_df = df[df['Domain'] == domain].copy()
            print(f"   Records in {domain}: {len(domain_df)}")
           
            for category in rating_categories:
                for model in models:
                    column_name = f"{model} Human {category}"
                   
                    if column_name in domain_df.columns:
                        # Extract and process ratings for this domain
                        if missing_strategy == "skip":
                            ratings = domain_df[column_name].dropna()
                        elif missing_strategy == "impute_mean":
                            ratings = domain_df[column_name].fillna(domain_df[column_name].mean())
                        elif missing_strategy == "impute_median":
                            ratings = domain_df[column_name].fillna(domain_df[column_name].median())
                        else:
                            ratings = domain_df[column_name].dropna()
                       
                        if missing_strategy != "report_only":
                            ratings = pd.to_numeric(ratings, errors='coerce').dropna()
                            ratings = ratings.round().astype(int)
                            ratings = ratings[(ratings >= 1) & (ratings <= 5)]
                       
                        processed_data[domain][category][model] = ratings.tolist()
                        print(f"     {category} - {model}: {len(ratings)} ratings")
       
        # Summary statistics
        print(f"\n📊 FINAL DATA SUMMARY:")
        for domain_key in processed_data.keys():
            total_ratings = 0
            for category in rating_categories:
                for model in models:
                    total_ratings += len(processed_data[domain_key][category][model])
            print(f"   {domain_key}: {total_ratings} total ratings")
       
        return processed_data
       
    except FileNotFoundError:
        print(f"❌ Error: '{data_filename}' not found in current directory")
        return None
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def load_all_languages_data(data_filename="RLHF data - Main RLHF.csv", missing_strategy="skip"):
    """
    Load data for all three languages and organize by category for cross-language analysis
   
    Parameters:
    data_filename: str - Name of the data file
    missing_strategy: str - How to handle missing values
   
    Returns:
    all_data: dict - {category: {language: {domain: {model: [task_averages]}}}}
    """
   
    languages = ["Python", "javascript", "Java"]
    language_domains = {
        "Python": ["Machine Learning", "FinTech", "EdTech"],
        "javascript": ["Social Networking", "Media", "Streaming"],
        "Java": ["E‑Commerce", "CRM", "Hotel"]
    }
   
    try:
        # Load the data file
        if data_filename.endswith('.csv'):
            df = pd.read_csv(data_filename)
        else:
            df = pd.read_excel(data_filename)
           
        df.columns = df.columns.str.strip()
        print(f"✅ Successfully loaded {data_filename}")
        print(f"📊 Dataset shape: {df.shape}")
       
        models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
        # Display names for legends (what user wants to see in charts)
        model_display_names = {
            'Chat GPT o4-mini-high': 'o4-mini-high',
            'Gemini 2.5 pro': 'Gemini 2.5 pro',
            'Claude Opus 4': 'Claude Opus 4'
        }
        rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
       
        # Initialize data structure: {category: {language: {domain: {model: [task_averages]}}}}
        all_data = {}
       
        for category in rating_categories:
            all_data[category] = {}
           
            for language in languages:
                all_data[category][language] = {}
               
                # Filter data for this language
                language_df = df[df['Code Language'] == language].copy()
                print(f"\n🔍 Processing {language}: {len(language_df)} records")
               
                if len(language_df) == 0:
                    print(f"⚠️  No data found for {language}")
                    continue
               
                # Add Overall domain (all domains combined)
                all_data[category][language]['Overall'] = {}
               
                # Add specific domains for this language
                for domain in language_domains[language]:
                    all_data[category][language][domain] = {}
               
                # Process Overall data (all domains combined for this language)
                for model in models:
                    column_name = f"{model} Human {category}"
                   
                    if column_name in language_df.columns:
                        if missing_strategy == "skip":
                            ratings = language_df[column_name].dropna()
                        else:
                            ratings = language_df[column_name].fillna(language_df[column_name].mean())
                       
                        ratings = pd.to_numeric(ratings, errors='coerce').dropna()
                        ratings = ratings.round().astype(int)
                        ratings = ratings[(ratings >= 1) & (ratings <= 5)]
                       
                        all_data[category][language]['Overall'][model] = ratings.tolist()
                    else:
                        all_data[category][language]['Overall'][model] = []
               
                # Process domain-specific data
                for domain in language_domains[language]:
                    domain_df = language_df[language_df['Domain'] == domain].copy()
                   
                    for model in models:
                        column_name = f"{model} Human {category}"
                       
                        if column_name in domain_df.columns and len(domain_df) > 0:
                            if missing_strategy == "skip":
                                ratings = domain_df[column_name].dropna()
                            else:
                                ratings = domain_df[column_name].fillna(domain_df[column_name].mean())
                           
                            ratings = pd.to_numeric(ratings, errors='coerce').dropna()
                            ratings = ratings.round().astype(int)
                            ratings = ratings[(ratings >= 1) & (ratings <= 5)]
                           
                            all_data[category][language][domain][model] = ratings.tolist()
                        else:
                            all_data[category][language][domain][model] = []
       
        # Print summary
        print(f"\n📊 DATA LOADING SUMMARY:")
        for category in rating_categories:
            total_ratings = 0
            for language in languages:
                lang_total = 0
                for domain in all_data[category][language].keys():
                    for model in models:
                        lang_total += len(all_data[category][language][domain].get(model, []))
                total_ratings += lang_total
                print(f"   {category} - {language}: {lang_total} ratings")
            print(f"   {category} TOTAL: {total_ratings} ratings")
       
        return all_data
       
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        return None


def calculate_rating_percentages(ratings_list):
    """
    Calculate percentage distribution of ratings 1-5
   
    Parameters:
    ratings_list: list - List of individual ratings
   
    Returns:
    percentages: list - Percentages for ratings 1-5
    counts: list - Actual counts for ratings 1-5
    """
    if not ratings_list:
        return [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
   
    # Count occurrences of each rating
    rating_counts = Counter(ratings_list)
    total_ratings = len(ratings_list)
   
    # Calculate percentages and counts for ratings 1-5
    percentages = []
    counts = []
    for rating in range(1, 6):
        count = rating_counts.get(rating, 0)
        percentage = (count / total_ratings) * 100
        percentages.append(percentage)
        counts.append(count)
   
    return percentages, counts


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def create_improved_rating_distribution_charts(processed_data, language="Python"):
    """Create 2x2 grid layout charts showing Overall + Domain-specific rating distributions with task-level aggregation"""
   
    # Get domain keys (Overall + actual domains)
    domain_keys = list(processed_data.keys())
    print(f"📊 Creating charts for domains: {domain_keys}")
   
    if len(domain_keys) == 0:
        print("❌ No valid data to plot")
        return None
   
    # Create 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
   
    # Add main title for the entire figure
    fig.suptitle(f'{language} - Multi-Model Performance Analysis by Domain (Task-Level Averages)',
                 fontsize=18, fontweight='bold', y=0.95)
   
    # Flatten axes for easier indexing
    axes = axes.flatten()
   
    # Color coding for models - Updated for better visual distinction
    model_colors = {
        'Chat GPT o4-mini-high': '#FF6B6B',    # Red
        'Gemini 2.5 pro': '#2ECC71',          # Green
        'Claude Opus 4': '#9B59B6'            # Purple
    }
   
    # Display names for legends (what user wants to see in charts)
    model_display_names = {
        'Chat GPT o4-mini-high': 'o4-mini-high',
        'Gemini 2.5 pro': 'Gemini 2.5 pro',
        'Claude Opus 4': 'Claude Opus 4'
    }
   
    models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
    rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
    ratings = [1, 2, 3, 4, 5]
   
    # Bar width and positions
    bar_width = 0.25
    x_positions = np.arange(len(ratings))  # [0, 1, 2, 3, 4] for ratings 1-5
   
    # Track legend handles for global legend
    legend_handles = []
    legend_labels = []
   
    # Plot each domain (up to 4: Overall + 3 domains)
    for i, domain_key in enumerate(domain_keys[:4]):  # Limit to 4 for 2x2 grid
        ax = axes[i]
        domain_data = processed_data[domain_key]
       
        print(f"\n--- Processing {domain_key} ---")
       
        # Calculate task-level averages instead of category aggregation
        model_task_averages = {model: [] for model in models}
       
        # Get number of tasks by checking the length of any category's data
        sample_category = rating_categories[0]
        num_tasks = len(domain_data[sample_category].get(models[0], []))
       
        print(f"  Number of tasks in {domain_key}: {num_tasks}")
       
        # For each task, calculate average rating across all categories
        for task_idx in range(num_tasks):
            for model in models:
                task_ratings = []
               
                # Collect all category ratings for this specific task
                for category in rating_categories:
                    ratings_list = domain_data[category].get(model, [])
                    if task_idx < len(ratings_list):
                        task_ratings.append(ratings_list[task_idx])
               
                # Calculate average rating for this task (round to nearest integer 1-5)
                if task_ratings:
                    task_avg = np.mean(task_ratings)
                    task_avg_rounded = round(task_avg)
                    task_avg_rounded = max(1, min(5, task_avg_rounded))  # Ensure it's 1-5
                    model_task_averages[model].append(task_avg_rounded)
       
        # Now count the distribution of averaged ratings
        model_totals = {model: [0, 0, 0, 0, 0] for model in models}  # Counts for ratings 1-5
       
        for model in models:
            task_averages = model_task_averages[model]
            for avg_rating in task_averages:
                if 1 <= avg_rating <= 5:
                    model_totals[model][avg_rating - 1] += 1
       
        # Print task-averaged data for verification
        for model in models:
            total_tasks = sum(model_totals[model])
            avg_scores = model_task_averages[model]
            mean_avg = np.mean(avg_scores) if avg_scores else 0
            print(f"  {model}: {total_tasks} tasks, Mean of averages: {mean_avg:.2f}")
            print(f"    Task average distribution: {model_totals[model]}")
            print(f"    Sample task averages: {avg_scores[:5]}...")
       
        # Create grouped bars for each model
        for j, model in enumerate(models):
            x_offset = x_positions + (j - 1) * bar_width  # Offset bars
            counts = model_totals[model]
           
            bars = ax.bar(x_offset, counts,
                         width=bar_width,
                         label=f'{model_display_names[model]} (n={sum(counts)})',  # Custom display label
                         color=model_colors[model],
                         alpha=0.8,
                         edgecolor='white',
                         linewidth=1)
           
            # Collect legend info only from the first subplot
            if i == 0:
                legend_handles.append(bars)
                legend_labels.append(f'{model_display_names[model]} (n={sum(counts)})')
       
        # Customize the subplot
        ax.set_title(f'{domain_key}', fontsize=14, fontweight='bold', pad=15)
       
        # Set x-axis ticks and labels
        ax.set_xticks(x_positions)
        ax.set_xticklabels(['1', '2', '3', '4', '5'])
       
        # Enhanced grid styling
        ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
        ax.set_axisbelow(True)
       
        # Set appropriate y-axis limits based on number of tasks
        max_tasks = max(num_tasks, 15)  # At least 15 for readability
        ax.set_ylim(0, max_tasks)
       
        # Set major ticks every appropriate interval
        tick_interval = max(1, max_tasks // 6)
        ax.set_yticks(range(0, max_tasks + 1, tick_interval))
   
    # Hide unused subplots if we have fewer than 4 domains
    for i in range(len(domain_keys), 4):
        axes[i].set_visible(False)
   
    # Axis labels removed - information moved to enhanced legend below
   
    # Create separate legends for models and axis information with proper spacing
    # Models legend - positioned higher with adequate bottom margin
    models_legend = fig.legend(legend_handles, legend_labels,
                             loc='lower left', bbox_to_anchor=(0.05, 0.02),
                             fontsize=11, title='Models', title_fontsize=12,
                             frameon=True, fancybox=True, shadow=True)
   
    # Axis information legend (horizontal layout like models)
    axis_legend = fig.legend([Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7),
                             Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7)],
                           ['X-Axis: Rating (1-5)', 'Y-Axis: Number of Tasks'],
                           loc='lower left', bbox_to_anchor=(0.35, 0.02),
                           fontsize=10, title='Chart Guide', title_fontsize=11,
                           frameon=True, fancybox=True, shadow=True,
                           ncol=2)  # Horizontal layout
   
    # Add comprehensive information box in bottom-right
    info_text = f"""Chart Guide:
Models: o4-mini-high (Red), Gemini 2.5 pro (Green), Claude Opus 4 (Purple)
📊 X-Axis: Rating Scale (1=Poorest → 5=Best)
📊 Y-Axis: Number of Tasks
• Bars show task count by average rating
• Each task's 7 category ratings are averaged
• Higher bars = more tasks at that performance level  
• Overall shows all domains combined
• Scale matches actual task count (~{max_tasks} max)"""
   
    fig.text(0.95, 0.02, info_text,
             fontsize=9, ha='right', va='bottom',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgray', alpha=0.9))
   
    # Improved spacing with adequate bottom margin for legends
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0, rect=[0.08, 0.18, 1, 0.92])
    plt.savefig(f'{language.lower()}_domain_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
   
    return fig




def create_summary_statistics_chart(processed_data):
    """
    Create a comprehensive summary statistics visualization
    """
   
    categories = list(processed_data.keys())
   
    # Calculate summary statistics
    summary_stats = {}
    for category, ratings in processed_data.items():
        if ratings:
            summary_stats[category] = {
                'mean': np.mean(ratings),
                'median': np.median(ratings),
                'std': np.std(ratings),
                'min': np.min(ratings),
                'max': np.max(ratings),
                'count': len(ratings)
            }
            # Step 3: Print detailed summary statistics for cross-verification
            print(f"\nSummary statistics for {category}:")
            print(f"  Mean: {summary_stats[category]['mean']:.2f}")
            print(f"  Median: {summary_stats[category]['median']:.2f}")
            print(f"  Std: {summary_stats[category]['std']:.2f}")
            print(f"  Min: {summary_stats[category]['min']}")
            print(f"  Max: {summary_stats[category]['max']}")
            print(f"  Count: {summary_stats[category]['count']}")
   
    if not summary_stats:
        print("❌ No data for summary statistics")
        return None
   
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
   
    # Chart 1: Mean ratings
    means = [summary_stats[cat]['mean'] for cat in categories]
    stds = [summary_stats[cat]['std'] for cat in categories]
   
    bars1 = ax1.bar(range(len(categories)), means, yerr=stds,
                    color='#3498DB', alpha=0.8, capsize=5, edgecolor='black')
    ax1.set_title('Average Rating by Category (with Error Bars)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Rating', fontsize=12)
    ax1.set_xticks(range(len(categories)))
    ax1.set_xticklabels([cat.replace(' & ', ' &\n') for cat in categories],
                        rotation=45, ha='right')
    ax1.set_ylim(1, 5)
    ax1.grid(True, axis='y', alpha=0.3)
   
    # Add value labels
    for bar, mean in zip(bars1, means):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
   
    # Chart 2: Rating distribution heatmap
    rating_matrix = []
    for category in categories:
        _, counts = calculate_rating_percentages(processed_data[category])
        rating_matrix.append(counts)
   
    rating_matrix = np.array(rating_matrix)
    im = ax2.imshow(rating_matrix, cmap='YlOrRd', aspect='auto')
    ax2.set_title('Rating Count Heatmap', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Rating (1-5)', fontsize=12)
    ax2.set_ylabel('Categories', fontsize=12)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(['1', '2', '3', '4', '5'])
    ax2.set_yticks(range(len(categories)))
    ax2.set_yticklabels([cat.replace(' & ', ' &\n') for cat in categories])
   
    # Add count labels to heatmap
    for i in range(len(categories)):
        for j in range(5):
            text = ax2.text(j, i, rating_matrix[i, j], ha="center", va="center",
                           color="black", fontweight='bold')
   
    # Add colorbar
    plt.colorbar(im, ax=ax2, label='Count')
   
    # Chart 3: Standard deviation comparison
    bars3 = ax3.bar(range(len(categories)), stds, color='#E74C3C', alpha=0.8, edgecolor='black')
    ax3.set_title('Rating Variability (Standard Deviation)', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Standard Deviation', fontsize=12)
    ax3.set_xticks(range(len(categories)))
    ax3.set_xticklabels([cat.replace(' & ', ' &\n') for cat in categories],
                        rotation=45, ha='right')
    ax3.grid(True, axis='y', alpha=0.3)
   
    # Add value labels
    for bar, std in zip(bars3, stds):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{std:.2f}', ha='center', va='bottom', fontweight='bold')
   
    # Chart 4: Sample size verification
    counts = [summary_stats[cat]['count'] for cat in categories]
    bars4 = ax4.bar(range(len(categories)), counts, color='#2ECC71', alpha=0.8, edgecolor='black')
    ax4.set_title('Sample Size per Category', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Number of Ratings', fontsize=12)
    ax4.set_xticks(range(len(categories)))
    ax4.set_xticklabels([cat.replace(' & ', ' &\n') for cat in categories],
                        rotation=45, ha='right')
    ax4.grid(True, axis='y', alpha=0.3)
   
    # Add value labels
    for bar, count in zip(bars4, counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
   
    plt.tight_layout(pad=3.0)
    plt.savefig('summary_statistics.png', dpi=300, bbox_inches='tight')
    plt.show()
   
    return fig


def create_alternative_agreement_analysis(processed_data):
    """
    Create alternative analysis since Fleiss's Kappa isn't applicable
    """
   
    print("\n" + "="*70)
    print("ALTERNATIVE AGREEMENT ANALYSIS")
    print("(Since Fleiss's Kappa requires multiple raters per task)")
    print("="*70)
   
    categories = list(processed_data.keys())
    agreement_measures = {}
   
    # Calculate various agreement/consistency measures
    for category, ratings in processed_data.items():
        if not ratings or len(ratings) < 2:
            continue
       
        # Measures of consensus/agreement
        mean_rating = np.mean(ratings)
        std_dev = np.std(ratings)
        variance = np.var(ratings)
       
        # Coefficient of variation (lower = more consistent)
        cv = std_dev / mean_rating if mean_rating > 0 else float('inf')
       
        # Range as consistency measure
        rating_range = np.max(ratings) - np.min(ratings)
       
        # Consensus score (0-1, higher = more consensus)
        # Based on inverse of coefficient of variation
        consensus_score = 1 / (1 + cv) if cv != float('inf') else 0
       
        # Percentage agreement (how many agree with mode)
        mode_rating = Counter(ratings).most_common(1)[0][0]
        mode_count = Counter(ratings)[mode_rating]
        percent_agreement = (mode_count / len(ratings)) * 100
       
        agreement_measures[category] = {
            'mean': mean_rating,
            'std_dev': std_dev,
            'cv': cv,
            'consensus_score': consensus_score,
            'rating_range': rating_range,
            'mode_rating': mode_rating,
            'percent_agreement': percent_agreement,
            'sample_size': len(ratings)
        }
       
        print(f"{category:25}: Consensus = {consensus_score:.3f}, "
              f"Agreement with Mode = {percent_agreement:.1f}%, "
              f"CV = {cv:.3f}")
        # Step 4: Print detailed agreement measures for cross-verification
        print(f"  Mean: {mean_rating:.2f}")
        print(f"  Std Dev: {std_dev:.2f}")
        print(f"  Coefficient of Variation: {cv:.3f}")
        print(f"  Consensus Score: {consensus_score:.3f}")
        print(f"  Rating Range: {rating_range}")
        print(f"  Mode Rating: {mode_rating}")
        print(f"  Percent Agreement: {percent_agreement:.1f}%")
        print(f"  Sample Size: {len(ratings)}")
   
    # Create visualization for alternative measures
    if agreement_measures:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
       
        categories_list = list(agreement_measures.keys())
       
        # Chart 1: Consensus scores
        consensus_scores = [agreement_measures[cat]['consensus_score'] for cat in categories_list]
        colors1 = ['#2ECC71' if score > 0.7 else '#F39C12' if score > 0.5 else '#E74C3C'
                   for score in consensus_scores]
       
        bars1 = ax1.bar(range(len(categories_list)), consensus_scores,
                        color=colors1, alpha=0.8, edgecolor='black')
        ax1.set_title('Consensus Score by Category\n(Higher = More Agreement)',
                      fontsize=14, fontweight='bold')
        ax1.set_ylabel('Consensus Score (0-1)', fontsize=12)
        ax1.set_xticks(range(len(categories_list)))
        ax1.set_xticklabels([cat.replace(' & ', ' &\n') for cat in categories_list],
                            rotation=45, ha='right')
        ax1.set_ylim(0, 1)
        ax1.grid(True, axis='y', alpha=0.3)
       
        # Add value labels
        for bar, score in zip(bars1, consensus_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
       
        # Chart 2: Percentage agreement with mode
        percent_agreements = [agreement_measures[cat]['percent_agreement'] for cat in categories_list]
        colors2 = ['#2ECC71' if pct > 70 else '#F39C12' if pct > 50 else '#E74C3C'
                   for pct in percent_agreements]
       
        bars2 = ax2.bar(range(len(categories_list)), percent_agreements,
                        color=colors2, alpha=0.8, edgecolor='black')
        ax2.set_title('Agreement with Most Common Rating\n(Percentage)',
                      fontsize=14, fontweight='bold')
        ax2.set_ylabel('Percentage Agreement (%)', fontsize=12)
        ax2.set_xticks(range(len(categories_list)))
        ax2.set_xticklabels([cat.replace(' & ', ' &\n') for cat in categories_list],
                            rotation=45, ha='right')
        ax2.set_ylim(0, 100)
        ax2.grid(True, axis='y', alpha=0.3)
       
        # Add value labels
        for bar, pct in zip(bars2, percent_agreements):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
       
        plt.tight_layout()
        plt.savefig('alternative_agreement_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
       
        print("✅ Alternative agreement analysis saved as 'alternative_agreement_analysis.png'")
   
    return agreement_measures


def create_summary_table(processed_data, agreement_measures=None):
    """Create a comprehensive summary table for multi-model data"""
   
    summary_data = []
    models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
   
    for category in processed_data.keys():
        for model in models:
            ratings_list = processed_data[category].get(model, [])
            if ratings_list:
                avg_rating = np.mean(ratings_list)
                std_rating = np.std(ratings_list)
                sample_size = len(ratings_list)
               
                # Calculate distribution
                percentages, counts = calculate_rating_percentages(ratings_list)
                mode_rating = percentages.index(max(percentages)) + 1
               
                summary_data.append({
                    'Category': category,
                    'Model': model,
                    'Sample Size': sample_size,
                    'Average Rating': f"{avg_rating:.2f}",
                    'Std Deviation': f"{std_rating:.2f}",
                    'Mode Rating': mode_rating
                })
            else:
                summary_data.append({
                    'Category': category,
                    'Model': model,
                    'Sample Size': 0,
                    'Average Rating': "N/A",
                    'Std Deviation': "N/A",
                    'Mode Rating': "N/A"
                })
   
    summary_df = pd.DataFrame(summary_data)
   
    print("\n" + "="*80)
    print("COMPREHENSIVE MULTI-MODEL SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
   
    return summary_df


def create_category_comparison_charts(all_data, category):
    """
    Create 4x3 grid layout showing all languages and domains for a specific rating category
   
    Parameters:
    all_data: dict - Data organized by category
    category: str - Rating category to visualize
   
    Returns:
    fig: matplotlib figure
    """
   
    print(f"\n📊 Creating {category} comparison charts...")
   
    if category not in all_data:
        print(f"❌ Category '{category}' not found in data")
        return None
   
    category_data = all_data[category]
   
    # Create 4x3 grid layout (3 rows for languages, 4 columns for overall+domains)
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
   
    # Add main title for the entire figure
    fig.suptitle(f'{category} Ratings - Cross-Language Domain Comparison',
                 fontsize=20, fontweight='bold', y=0.96)
   
    # Language configuration
    languages = ["Python", "javascript", "Java"]
    language_domains = {
        "Python": ["Overall", "Machine Learning", "FinTech", "EdTech"],
        "javascript": ["Overall", "Social Networking", "Media", "Streaming"],
        "Java": ["Overall", "E‑Commerce", "CRM", "Hotel"]
    }
   
    language_colors = {
        "Python": "#3776ab",      # Python blue
        "javascript": "#f7df1e",  # JavaScript yellow
        "Java": "#ed8b00"         # Java orange
    }
   
    # Model colors
    model_colors = {
        'Chat GPT o4-mini-high': '#FF6B6B',    # Red
        'Gemini 2.5 pro': '#2ECC71',          # Green
        'Claude Opus 4': '#9B59B6'            # Purple
    }
   
    # Display names for legends (what user wants to see in charts)
    model_display_names = {
        'Chat GPT o4-mini-high': 'o4-mini-high',
        'Gemini 2.5 pro': 'Gemini 2.5 pro',
        'Claude Opus 4': 'Claude Opus 4'
    }
   
    models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
    ratings = [1, 2, 3, 4, 5]
   
    # Bar width and positions
    bar_width = 0.25
    x_positions = np.arange(len(ratings))
   
    # Track legend handles for global legend
    legend_handles = []
    legend_labels = []
    legend_created = False
   
    # Process each language (row)
    for lang_idx, language in enumerate(languages):
        if language not in category_data:
            print(f"⚠️  No data for {language} in {category}")
            continue
           
        lang_data = category_data[language]
        domains = language_domains[language]
       
        # Process each domain (column)
        for dom_idx, domain in enumerate(domains):
            ax = axes[lang_idx, dom_idx]
           
            if domain not in lang_data:
                ax.set_visible(False)
                continue
           
            domain_data = lang_data[domain]
           
            # Calculate rating distribution for each model
            model_counts = {model: [0, 0, 0, 0, 0] for model in models}
           
            for model in models:
                ratings_list = domain_data.get(model, [])
               
                if ratings_list:
                    # Count occurrences of each rating (1-5)
                    for rating in ratings_list:
                        if 1 <= rating <= 5:
                            model_counts[model][rating - 1] += 1
           
            # Create grouped bars for each model
            for j, model in enumerate(models):
                x_offset = x_positions + (j - 1) * bar_width
                counts = model_counts[model]
               
                bars = ax.bar(x_offset, counts,
                             width=bar_width,
                             label=f'{model_display_names[model]} (n={sum(counts)})',
                             color=model_colors[model],
                             alpha=0.8,
                             edgecolor='white',
                             linewidth=1)
               
                # Collect legend info only once
                if not legend_created and lang_idx == 0 and dom_idx == 0:
                    legend_handles.append(bars)
                    legend_labels.append(f'{model_display_names[model]}')
           
            if not legend_created and lang_idx == 0 and dom_idx == 0:
                legend_created = True
           
            # Customize the subplot
            domain_short = domain if len(domain) <= 15 else domain[:12] + "..."
            ax.set_title(f'{domain_short}',
                        fontsize=11, fontweight='bold', pad=10,
                        bbox=dict(boxstyle='round,pad=0.3',
                                facecolor=language_colors[language],
                                alpha=0.3))
           
            # Add language label only on the leftmost graph of each row
            if dom_idx == 0:
                # Add language name closer to graphs to reduce excessive left spacing
                ax.text(-0.18, 0.5, language,
                       transform=ax.transAxes,
                       fontsize=14, fontweight='bold',
                       ha='center', va='center', rotation=90,
                       bbox=dict(boxstyle='round,pad=0.5',
                               facecolor=language_colors[language],
                               alpha=0.7, edgecolor='black'))
               
                # Y-axis labels removed - info moved to legend
                pass
            else:
                # Y-axis labels removed - info moved to legend
                pass
           
            # Set x-axis ticks and labels
            ax.set_xticks(x_positions)
            ax.set_xticklabels(['1', '2', '3', '4', '5'], fontsize=10)
           
            # Enhanced grid styling
            ax.grid(True, axis='y', linestyle='--', alpha=0.4, color='gray')
            ax.set_axisbelow(True)
           
            # Set appropriate y-axis limits
            max_count = max([max(model_counts[model]) for model in models] + [5])
            ax.set_ylim(0, max_count + 2)
           
            # Add Y-axis labels to ALL graphs (not just leftmost)
            # ax.set_ylabel('Number of Tasks', fontsize=9) # This line is now handled by the if/else block
           
            # X-axis labels removed - info moved to legend
           
            # Customize tick labels for better readability
            ax.tick_params(axis='both', which='major', labelsize=8)
           
            # Set appropriate Y-axis ticks based on actual data range
            if max_count <= 10:
                tick_step = 1
            elif max_count <= 20:
                tick_step = 2
            else:
                tick_step = 5
            ax.set_yticks(range(0, max_count + 1, tick_step))
           
            # Print data for verification
            total_tasks = sum([sum(model_counts[model]) for model in models])
            print(f"  {language} - {domain}: {total_tasks} total tasks, max_count: {max_count}")
   
    # Remove redundant global axis labels since each graph has individual labels
    # fig.text(0.5, 0.02, f'{category} Rating Scale: 1 = Poorest, 5 = Best',
    #          fontsize=12, fontweight='bold', ha='center',
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
   
    # fig.text(0.02, 0.5, 'Task Count Distribution',
    #          fontsize=12, fontweight='bold', ha='center', va='center', rotation=90,
    #          bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
   
    # Create separate legends for models and axis information
    if legend_handles:
        # Models legend (centered)
        models_legend = fig.legend(legend_handles, legend_labels,
                                 loc='center', bbox_to_anchor=(0.35, 0.02),
                                 fontsize=12, title='Models', title_fontsize=13,
                                 frameon=True, fancybox=True, shadow=True,
                                 ncol=3)  # Horizontal layout to save space
       
        # Axis information legend (horizontal layout like models)
        axis_legend = fig.legend([Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.7),
                                 Rectangle((0,0),1,1, facecolor='lightgreen', alpha=0.7)],
                               ['X-Axis: Rating (1-5)', 'Y-Axis: Number of Tasks'],
                               loc='center', bbox_to_anchor=(0.65, 0.02),
                               fontsize=11, title='Chart Guide', title_fontsize=12,
                               frameon=True, fancybox=True, shadow=True,
                               ncol=2)  # Horizontal layout
   
    # Remove the comprehensive information box as requested
    # info_text = f"""{category} Analysis Guide:
    # Models: ChatGPT (Red), Gemini (Green), Claude (Purple)
    # • Each row represents a programming language
    # • Each column shows domain performance
    # • Bars show task count by {category.lower()} rating
    # • Compare cross-language domain patterns
    # • Identify model strengths/weaknesses per language"""
   
    # fig.text(0.98, 0.02, info_text,
    #          fontsize=10, ha='right', va='bottom',
    #          bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.9))
   
    # Balanced spacing with adequate bottom margin for legends
    plt.tight_layout(pad=3.0, h_pad=2.0, w_pad=1.5, rect=[0.08, 0.15, 0.95, 0.94])
   
    # Save with category-specific filename
    filename = f"{category.lower().replace(' ', '_')}_cross_language_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
   
    print(f"✅ Saved {filename}")
    return fig


# =============================================================================
# MAIN EXECUTION FUNCTION
# =============================================================================


def run_complete_analysis_corrected(data_filename="RLHF data - Main RLHF.csv", language="Python", missing_strategy="skip"):
    """
    Run the complete corrected analysis for your multi-model dataset with robust missing data handling
   
    Parameters:
    data_filename: str - Name of the data file
    language: str - Programming language to analyze ('Python', 'javascript', 'Java')
    missing_strategy: str - How to handle missing values: 'skip', 'impute_mean', 'impute_median', 'report_only'
    """
   
    print(f"🔄 Starting Multi-Model Analysis for {language} with Robust Data Handling...")
    print("="*60)
    print(f"📋 Missing data strategy: {missing_strategy}")
   
    # 1. Load and process the data
    print("\n1️⃣ Loading and processing data...")
    processed_data = load_and_process_data(data_filename, language, missing_strategy)
   
    if processed_data is None or not processed_data:
        print("❌ Failed to load data or no valid data found.")
        return None, None, None
   
    # Skip visualization for report_only mode
    if missing_strategy == "report_only":
        print("\n📋 Data quality report completed. Use a different strategy for visualization.")
        return processed_data, None, None
   
    # 2. Data overview by domain
    print(f"\n2️⃣ Data Overview for {language}:")
    models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
    # Display names for legends (what user wants to see in charts)
    model_display_names = {
        'Chat GPT o4-mini-high': 'o4-mini-high',
        'Gemini 2.5 pro': 'Gemini 2.5 pro',
        'Claude Opus 4': 'Claude Opus 4'
    }
    rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
   
    for domain_key in processed_data.keys():
        total_ratings = 0
        categories_with_data = 0
       
        print(f"\n   📊 {domain_key}:")
        for model in models:
            model_total = 0
            for category in rating_categories:
                ratings = processed_data[domain_key][category].get(model, [])
                model_total += len(ratings)
            total_ratings += model_total
            print(f"      {model_display_names[model]}: {model_total} ratings")
       
        print(f"      Total for {domain_key}: {total_ratings} ratings")
   
    # 3. Create domain-based rating distribution charts
    print(f"\n3️⃣ Creating {language} domain-based rating distribution charts...")
    create_improved_rating_distribution_charts(processed_data, language)
   
    # 4. Create comprehensive summary table
    print("\n4️⃣ Generating comprehensive summary...")
    summary_df = create_domain_summary_table(processed_data, language)
   
    # 5. Analysis explanation
    print("\n" + "="*70)
    print(f"📝 {language.upper()} MULTI-MODEL DOMAIN ANALYSIS")
    print("="*70)
    print(f"""
🔍 Analysis Structure:
   - Programming Language: {language}
   - Models: ChatGPT, Gemini, Claude
   - Domain-based comparison with Overall aggregation
   - Missing data strategy: {missing_strategy}
   
💡 Visualization Approach:
   - 2×2 grid showing Overall + Top 3 Domains
   - Aggregated ratings across all categories per domain
   - Direct comparison of model performance by domain
   - Consistent 0-30 scale for easy comparison
   - Identifies domain-specific model strengths
    """)
   
    print("\n✅ Analysis Complete!")
    print("📁 Generated files:")
    print(f"   - {language.lower()}_domain_analysis.png")
   
    return processed_data, None, summary_df


def create_domain_summary_table(processed_data, language):
    """Create a comprehensive summary table for domain-based multi-model data"""
   
    summary_data = []
    models = ['Chat GPT o4-mini-high', 'Gemini 2.5 pro', 'Claude Opus 4']
    # Display names for legends (what user wants to see in charts)
    model_display_names = {
        'Chat GPT o4-mini-high': 'o4-mini-high',
        'Gemini 2.5 pro': 'Gemini 2.5 pro',
        'Claude Opus 4': 'Claude Opus 4'
    }
    rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
   
    for domain_key in processed_data.keys():
        for model in models:
            # Aggregate all ratings for this model in this domain
            all_ratings = []
            for category in rating_categories:
                ratings_list = processed_data[domain_key][category].get(model, [])
                all_ratings.extend(ratings_list)
           
            if all_ratings:
                avg_rating = np.mean(all_ratings)
                std_rating = np.std(all_ratings)
                sample_size = len(all_ratings)
               
                # Calculate distribution
                percentages, counts = calculate_rating_percentages(all_ratings)
                mode_rating = percentages.index(max(percentages)) + 1
               
                summary_data.append({
                    'Language': language,
                    'Domain': domain_key,
                    'Model': model_display_names[model],  # Custom display name
                    'Sample Size': sample_size,
                    'Average Rating': f"{avg_rating:.2f}",
                    'Std Deviation': f"{std_rating:.2f}",
                    'Mode Rating': mode_rating
                })
            else:
                summary_data.append({
                    'Language': language,
                    'Domain': domain_key,
                    'Model': model_display_names[model],
                    'Sample Size': 0,
                    'Average Rating': "N/A",
                    'Std Deviation': "N/A",
                    'Mode Rating': "N/A"
                })
   
    summary_df = pd.DataFrame(summary_data)
   
    print("\n" + "="*80)
    print(f"COMPREHENSIVE {language.upper()} DOMAIN SUMMARY TABLE")
    print("="*80)
    print(summary_df.to_string(index=False))
    print("="*80)
   
    return summary_df


def run_complete_cross_language_analysis(data_filename="RLHF data - Main RLHF.csv", missing_strategy="skip"):
    """
    Run complete cross-language analysis generating 7 category-specific comparison pages
   
    Parameters:
    data_filename: str - Name of the data file
    missing_strategy: str - How to handle missing values
    """
   
    print("🔄 Starting Cross-Language Multi-Model Analysis...")
    print("="*70)
    print(f"📋 Missing data strategy: {missing_strategy}")
    print("📊 Generating 7 category-specific comparison pages...")
   
    # 1. Load data for all languages
    print("\n1️⃣ Loading data for all languages...")
    all_data = load_all_languages_data(data_filename, missing_strategy)
   
    if all_data is None:
        print("❌ Failed to load data.")
        return None
   
    # 2. Generate visualization for each category
    rating_categories = ['Coherence', 'Completeness', 'Correctness', 'Creativity', 'Helpfulness', 'Relevance', 'Style Presentation']
   
    print("\n2️⃣ Generating category comparison charts...")
    generated_files = []
   
    for i, category in enumerate(rating_categories, 1):
        print(f"\n📈 Processing {i}/7: {category}")
        fig = create_category_comparison_charts(all_data, category)
       
        if fig is not None:
            filename = f"{category.lower().replace(' ', '_')}_cross_language_analysis.png"
            generated_files.append(filename)
   
    # 3. Summary
    print("\n" + "="*70)
    print("📝 CROSS-LANGUAGE ANALYSIS COMPLETE")
    print("="*70)
    print(f"""
🔍 Analysis Overview:
   - 7 rating categories analyzed
   - 3 programming languages compared
   - 4×3 grid layout per category
   - Models: ChatGPT, Gemini, Claude
   
💡 Visualization Structure:
   - Each page focuses on one rating category
   - Rows: Python, JavaScript, Java
   - Columns: Overall + Top 3 Domains per language
   - Cross-language domain pattern comparison
   
📁 Generated Files ({len(generated_files)} total):""")
   
    for file in generated_files:
        print(f"   - {file}")
   
    print(f"""
🎯 Analysis Benefits:
   - Identify category-specific model strengths
   - Compare domain performance across languages
   - Spot consistent patterns or anomalies
   - Guide model selection for specific use cases
    """)
   
    return all_data, generated_files


# =============================================================================
# USAGE INSTRUCTIONS
# =============================================================================


def print_corrected_usage_instructions():
    """Print usage instructions for domain-based multi-model analysis"""
   
    print("""
🔧 DOMAIN-BASED MULTI-MODEL ANALYSIS INSTRUCTIONS:
==========================================


1. 📁 PLACE YOUR DATA FILE in the same directory as this script
   - Default file: "RLHF data - Main RLHF.csv"
   - Supported: .xlsx, .csv files


2. 🏃 RUN ANALYSIS BY PROGRAMMING LANGUAGE:
   
   # Python Analysis (default)
   data, _, summary = run_complete_analysis_corrected(language="Python")
   
   # JavaScript Analysis
   data, _, summary = run_complete_analysis_corrected(language="javascript")
   
   # Java Analysis  
   data, _, summary = run_complete_analysis_corrected(language="Java")


3. 🎯 ANALYSIS STRUCTURE:
   
   ✅ 2×2 Grid Layout: Overall + 3 Top Domains per Language
   ✅ Domain-Specific Analysis:
      - Python: Machine Learning, FinTech, EdTech
      - JavaScript: Social Networking, Media, Streaming  
      - Java: E-Commerce, CRM, Hotel
   ✅ Model Comparison: ChatGPT, Gemini, Claude
   ✅ Aggregated ratings across all 7 categories per domain


4. 📊 MISSING DATA STRATEGIES:
   
   ✅ 'skip' (default): Remove rows with missing values
   ✅ 'impute_mean': Fill missing values with column mean
   ✅ 'impute_median': Fill missing values with column median  
   ✅ 'report_only': Analyze data quality without processing


5. 📈 WHAT YOU'LL GET:
   
   ✅ Domain-Based Performance Charts
      - Overall performance across all domains
      - Individual domain performance analysis
      - Model comparison within each domain
   
   ✅ Comprehensive Summary Tables
      - Language-specific statistics
      - Domain-wise model performance
      - Sample sizes and distributions
   
   ✅ Data Quality Assessment
      - Missing data detection and handling
      - Robust processing with multiple strategies


6. 📁 OUTPUT FILES:
   - python_domain_analysis.png
   - javascript_domain_analysis.png  
   - java_domain_analysis.png


7. 🔄 BATCH PROCESSING EXAMPLE:
   
   for language in ["Python", "javascript", "Java"]:
       run_complete_analysis_corrected(language=language)
""")


def print_cross_language_usage_instructions():
    """Print usage instructions for cross-language analysis"""
   
    print("""
🔧 CROSS-LANGUAGE MULTI-MODEL ANALYSIS INSTRUCTIONS:
==========================================


1. 📁 DATA REQUIREMENTS:
   - Default file: "RLHF data - Main RLHF.csv"
   - Must contain Python, JavaScript, and Java data
   - 7 rating categories per language/domain


2. 🏃 RUN COMPLETE ANALYSIS:
   
   # Generate all 7 category comparison pages
   data, files = run_complete_cross_language_analysis()
   
   # With custom missing data strategy
   data, files = run_complete_cross_language_analysis(missing_strategy="impute_mean")


3. 🎯 OUTPUT STRUCTURE:
   
   ✅ 7 PNG Files Generated:
      - coherence_cross_language_analysis.png
      - completeness_cross_language_analysis.png
      - correctness_cross_language_analysis.png
      - creativity_cross_language_analysis.png
      - helpfulness_cross_language_analysis.png
      - relevance_cross_language_analysis.png
      - style_presentation_cross_language_analysis.png
   
   ✅ Each Page Contains:
      - 4×3 Grid (12 graphs total)
      - Row 1: Python (Overall + ML + FinTech + EdTech)
      - Row 2: Java (Overall + E-Commerce + CRM + Hotel)
      - Row 3: JavaScript (Overall + Social + Media + Streaming)


4. 📊 ANALYSIS BENEFITS:
   
   ✅ Category-Focused Insights:
      - See which models excel at specific criteria
      - Identify consistent strengths/weaknesses
      - Compare performance across programming contexts
   
   ✅ Cross-Language Patterns:
      - Domain performance comparison
      - Language-specific model preferences
      - Universal vs context-specific strengths


5. 🔍 INTERPRETATION GUIDE:
   - Higher bars = more tasks at that rating level
   - Compare bar heights across languages/domains
   - Look for patterns within categories
   - Identify outliers or consistent performers
""")


# =============================================================================
# RUN THE CORRECTED ANALYSIS
# =============================================================================


if __name__ == "__main__":
    # Print usage instructions
    print_cross_language_usage_instructions()
   
    # Run the complete cross-language analysis
    try:
        all_data, generated_files = run_complete_cross_language_analysis(
            data_filename="RLHF data - Main RLHF.csv",
            missing_strategy="skip"
        )
       
        if all_data and generated_files:
            print(f"\n🎉 Successfully generated {len(generated_files)} analysis pages!")
            print("📊 Cross-language model comparison complete!")
   
    except Exception as e:
        print(f"❌ Error running analysis: {str(e)}")
        import traceback
        print(f"🔍 Full error details:\n{traceback.format_exc()}")
        print("\n💡 Troubleshooting:")
        print("   - Check if data file exists in current directory")
        print("   - Verify data contains all three languages")
        print("   - Ensure all rating categories are present")