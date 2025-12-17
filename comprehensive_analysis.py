import os
import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys
from datetime import datetime
from scipy import stats
from scipy.stats import kruskal, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

# Import Plotly for modern boxplots
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    print("Warning: Plotly not available. Will use matplotlib for plots.")
    PLOTLY_AVAILABLE = False

class Logger:
    """Class to log all print statements to both console and file"""
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
        
        # Write header with timestamp
        header = f"="*80 + "\n"
        header += f"COMPREHENSIVE SLACKLINE DATA ANALYSIS\n"
        header += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        header += f"="*80 + "\n\n"
        
        self.log.write(header)
        self.terminal.write(header)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        footer = f"\n\n" + "="*80 + "\n"
        footer += f"Analysis completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        footer += f"="*80 + "\n"
        
        self.log.write(footer)
        self.terminal.write(footer)
        self.log.close()

def parse_filename(filename):
    """Extract ID, Test and Repetition from filename."""
    match = re.match(r"(ID\d+)-Dwall-(\d+)([a-z]?)", filename)
    if not match:
        return None
    subject_id, test, repetition = match.groups()
    test = int(test)
    repetition_num = ord(repetition) - 96 if repetition else None
    return subject_id, test, repetition_num

def resample_to_30hz(df, cell_cols, cop_cols, original_hz=40, target_hz=30):
    """
    Resample specified columns from original_hz to target_hz using linear interpolation.
    Adds a common time vector (in seconds) starting from 0 at target_hz.
    Also includes all kinematic columns from the original DataFrame.
    """
    n_samples = len(df)
    duration = n_samples / original_hz
    # Original and target time vectors
    t_orig = np.arange(n_samples) / original_hz
    n_target = int(np.round(duration * target_hz))
    t_target = np.arange(n_target) / target_hz
    
    # Resample columns
    resampled = {}
    for col in cell_cols + cop_cols:
        if col in df:
            resampled[col] = np.interp(t_target, t_orig, df[col].values)
    
    # Build new DataFrame with resampled columns and time vector
    resampled_df = pd.DataFrame(resampled)
    resampled_df['t'] = t_target[:len(resampled_df)]
    
    # Add all kinematic columns from original DataFrame
    kinematic_vars = [
        'APT', 'MLT', 'Trunk Rotation', 'HeadFlexExt', 'HeadLatFlex',
        'ShoulderRightFlexExt', 'ShoulderRightAddAbd', 'ShoulderRightIntraExtra',
        'ShoulderLeftFlexExt', 'ShoulderLeftAddAbd', 'ShoulderLeftIntraExtra',
        'ElbowRightFlexExt', 'ElbowLeftFlexExt',
        'HipRightFlexExt', 'HipLeftFlexExt',
        'HipRightAddAbd', 'HipLeftAddAbd',
        'KneeRightFlexExt', 'KneeLeftFlexExt'
    ]
    
    for kin_var in kinematic_vars:
        if kin_var in df.columns:
            # Cut the kinematic column to match the resampled length
            resampled_df[kin_var] = df[kin_var].iloc[:len(resampled_df)].values
    
    # Attach meta columns (if present)
    meta_cols = [c for c in ['Subject_ID', 'Test', 'Repetition', 'Group'] if c in df.columns]
    for col in meta_cols:
        if col in df:
            val = df[col].iloc[0] if len(df[col]) > 0 else None
            resampled_df[col] = val
    
    return resampled_df

def load_test_data(base_dir, test_number, repetitions=None):
    """
    Load data for a specific test number.
    
    Args:
        base_dir: Base directory containing the data
        test_number: Test number to load (1, 2, 3, 4, 5)
        repetitions: List of repetition numbers to include (None for all)
    """
    all_data = []
    
    for group in os.listdir(base_dir):
        group_dir = os.path.join(base_dir, group)
        if os.path.isdir(group_dir):
            for subject in os.listdir(group_dir):
                subject_dir = os.path.join(group_dir, subject)
                if os.path.isdir(subject_dir):
                    for file in os.listdir(subject_dir):
                        if file.endswith(".csv"):
                            parsed = parse_filename(file)
                            if parsed:
                                subject_id, test, repetition = parsed
                                if test == test_number:
                                    if repetitions is None or repetition in repetitions:
                                        filepath = os.path.join(subject_dir, file)
                                        df = pd.read_csv(filepath)
                                        # Identify columns for resampling
                                        cell_cols = [col for col in df.columns if 'cell' in col.lower()]
                                        cop_cols = [col for col in df.columns if 'cop' in col.lower()]
                                        df = resample_to_30hz(df, cell_cols, cop_cols)
                                        
                                        df["Subject_ID"] = subject_id
                                        df["Test"] = test
                                        df["Repetition"] = repetition
                                        df["Group"] = group.lower()
                                        all_data.append(df)
                                    # if repetition == [1]:
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    else:
        return pd.DataFrame()

def load_demographic_data(excel_path):
    """Load demographic data including weight from Excel file."""
    try:
        demo_df = pd.read_excel(excel_path)
        print("Demographic data columns:", demo_df.columns.tolist())
        
        # Try to find weight column
        weight_col = None
        for col in demo_df.columns:
            if any(keyword in col.lower() for keyword in ['weight', 'peso', 'kg', 'mass']):
                weight_col = col
                break
        
        if weight_col is None:
            print("Weight column not found. Available columns:", demo_df.columns.tolist())
            return None
        
        # Try to find subject ID column
        id_col = None
        for col in demo_df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'subject', 'participant', 'soggetto']):
                id_col = col
                break
        
        if id_col is None:
            print("Subject ID column not found. Available columns:", demo_df.columns.tolist())
            return None
        
        # Extract relevant columns
        weight_data = demo_df[[id_col, weight_col]].copy()
        weight_data.columns = ['Subject_ID', 'Weight']
        
        # Ensure Subject_ID format matches
        weight_data['Subject_ID'] = weight_data['Subject_ID'].astype(str)
        if not weight_data['Subject_ID'].str.startswith('ID').any():
            weight_data['Subject_ID'] = 'ID' + weight_data['Subject_ID'].astype(str)
        
        print("Weight data loaded successfully:")
        print(weight_data.head())
        return weight_data
        
    except Exception as e:
        print(f"Error loading demographic data: {e}")
        return None

def preprocess_data(df):
    """
    Apply preprocessing steps: time filtering, duration cutting, and averaging.
    """
    print("Starting data preprocessing...")
    
    # Keep only after the first 5 seconds
    if 't' in df.columns:
        original_shape = df.shape
        df = df[df['t'] > 5].copy()
        print(f"After time filtering (t > 5s): {original_shape} -> {df.shape}")

    # Cut equal length duration
    print("Cutting equal length duration...")
    durations = []
    unique_reps = df['Repetition'].unique()
    for rep in unique_reps:
        rep_data = df[df['Repetition'] == rep]
        if len(rep_data) > 0:
            durations.append(np.max(rep_data['t']))
            print(f"Repetition {rep} duration: {durations[-1]:.2f}s")
    
    durations.append(30)  # Safety limit
    cut_idx = np.min(durations)
    print(f"Cutting all data at: {cut_idx:.2f}s")
    df = df[df['t'] < cut_idx].copy()
    print(f"After cutting: {df.shape}")

    # Average across repetitions
    print("Averaging across repetitions...")
    meta_cols = [c for c in ['Subject_ID', 'Group', 't'] if c in df.columns]
    avg_cols = [c for c in df.columns if c not in meta_cols + ['Test', 'Repetition']]
    df_avg = df.groupby(meta_cols)[avg_cols].mean().reset_index()
    print(f"Averaged DataFrame shape: {df_avg.shape}")
    print(f"Unique subjects: {df_avg['Subject_ID'].nunique()}")
    print(f"Unique groups: {df_avg['Group'].unique()}")
    
    return df_avg

def calculate_comprehensive_metrics(df_avg, weight_data=None):
    """
    Calculate metrics for all variables (COP, normalized cells, kinematics) in one shot.
    """
    print("Calculating comprehensive metrics for all variables...")
    
    # Merge weight data if available
    if weight_data is not None:
        df_work = df_avg.merge(weight_data, on='Subject_ID', how='left')
    else:
        df_work = df_avg.copy()
    
    # Identify all variable types
    cop_cols = [col for col in df_work.columns if 'cop' in col.lower()]
    cell_cols = [col for col in df_work.columns if 'cell' in col.lower() and any(str(i) in col.lower() for i in range(1, 5))]
    kinematic_vars = [
        'APT', 'MLT', 'Trunk Rotation', 'HeadFlexExt', 'HeadLatFlex',
        'ShoulderRightFlexExt', 'ShoulderRightAddAbd', 'ShoulderRightIntraExtra',
        'ShoulderLeftFlexExt', 'ShoulderLeftAddAbd', 'ShoulderLeftIntraExtra',
        'ElbowRightFlexExt', 'ElbowLeftFlexExt',
        'HipRightFlexExt', 'HipLeftFlexExt',
        'HipRightAddAbd', 'HipLeftAddAbd',
        'KneeRightFlexExt', 'KneeLeftFlexExt'
    ]
    kinematic_cols = [var for var in kinematic_vars if var in df_work.columns]
    
    # Normalize cell columns by weight if available
    if weight_data is not None and len(cell_cols) > 0:
        print("Normalizing cell values by subject weight...")
        for cell_col in cell_cols:
            mask = df_work['Weight'].notna()
            df_work.loc[mask, cell_col] = df_work.loc[mask, cell_col] / df_work.loc[mask, 'Weight']
    
    # Combine all variables to analyze
    all_variables = cop_cols + cell_cols + kinematic_cols
    variable_types = (['COP'] * len(cop_cols) + 
                     ['Cell_Normalized'] * len(cell_cols) + 
                     ['Kinematic'] * len(kinematic_cols))
    
    print(f"Found {len(cop_cols)} COP variables: {cop_cols}")
    print(f"Found {len(cell_cols)} Cell variables: {cell_cols}")
    print(f"Found {len(kinematic_cols)} Kinematic variables: {kinematic_cols}")
    
    if not all_variables:
        print("No variables found for analysis!")
        return None
    
    metrics_list = []
    
    # Calculate metrics for each subject and variable
    for (subject_id, group), group_data in df_work.groupby(['Subject_ID', 'Group']):
        # Skip subjects without weight data for cell analysis
        if weight_data is not None and len(cell_cols) > 0 and group_data['Weight'].isna().all():
            print(f"Skipping subject {subject_id} - no weight data available")
            continue
            
        for var, var_type in zip(all_variables, variable_types):
            if var in group_data.columns:
                data = group_data[var].values
                
                # Calculate all metrics
                q05_val = np.percentile(data, 5)
                q95_val = np.percentile(data, 95)
                mean_val = np.mean(data)
                std_val = np.std(data, ddof=1)
                range_val = q95_val - q05_val
                median_val = np.median(data)
                q25 = np.percentile(data, 25)
                q75 = np.percentile(data, 75)
                iqr_val = q75 - q25
                
                metrics_list.append({
                    'Subject_ID': subject_id,
                    'Group': group,
                    'Variable': var,
                    'Variable_Type': var_type,
                    'Q05': q05_val,
                    'Q95': q95_val,
                    'Mean': mean_val,
                    'Std': std_val,
                    'Range': range_val,
                    'Median': median_val,
                    'IQR': iqr_val
                })
    
    if not metrics_list:
        print("No metrics could be calculated!")
        return None
    
    metrics_df = pd.DataFrame(metrics_list)
    print(f"Calculated metrics for {len(all_variables)} variables across {metrics_df['Subject_ID'].nunique()} subjects")
    return metrics_df

def create_individual_boxplots(metrics_df, output_dir):
    """
    Create individual boxplot figures for each variable and each metric using Plotly.
    """
    print("Creating individual boxplots for each variable and metric...")
    
    # Create output directory
    plots_dir = Path(output_dir) / "individual_boxplots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Group mapping and colors - Updated order: Sedentary, Athletes, Slackliners
    group_map = {'sedentari': 'Sedentary', 'sportivi': 'Athletes', 'slackliner': 'Slackliners'}
    colors = ['red', 'blue', 'green']  # Matching the Cardio project style
    
    # Get all unique variables and metrics
    variables = metrics_df['Variable'].unique()
    metrics = ['Q05', 'Q95', 'Mean', 'Std', 'Range', 'Median', 'IQR']
    
    total_plots = 0
    
    for variable in variables:
        print(f"Creating boxplots for {variable}")
        var_data = metrics_df[metrics_df['Variable'] == variable]
        var_type = var_data['Variable_Type'].iloc[0]
        
        for metric in metrics:
            if metric not in var_data.columns:
                continue
                
            if PLOTLY_AVAILABLE:
                # Create Plotly boxplot (preferred)
                fig = make_subplots(rows=1, cols=1,
                                    subplot_titles=[f"{variable} - {metric}"],
                                    horizontal_spacing=0.1)
                
                # Groups in desired order: Sedentary, Athletes, Slackliners
                groups = ['sedentari', 'sportivi', 'slackliner']
                
                for i, group in enumerate(groups):
                    group_data = var_data[var_data['Group'] == group][metric].dropna()
                    if len(group_data) > 0:
                        fig.add_trace(
                            go.Box(y=group_data.values, 
                                   name=group_map[group], 
                                   boxpoints=False,  # Remove outliers/fliers
                                   marker_color=colors[i % len(colors)], 
                                   opacity=0.7,
                                   showlegend=True),
                            row=1, col=1
                        )
                
                # Update layout with custom styling
                fig.update_yaxes(
                    title_text=f"{metric} Value", 
                    row=1, col=1,
                    linecolor='black',
                    linewidth=2,
                    showgrid=False,  # Remove grid lines
                    zeroline=False
                )
                fig.update_xaxes(
                    title_text="",  # Remove "Groups" label
                    row=1, col=1,
                    linecolor='black',
                    linewidth=2,
                    showgrid=False,  # Remove grid lines
                    zeroline=False
                )
                
                fig.update_layout(
                    # title=f"{variable} - {metric} Distribution Across Groups<br><sub>Variable Type: {var_type}</sub>",
                    showlegend=True,
                    legend=dict(
                        x=0.02,  # Position legend inside plot area
                        y=0.98,  # Top-left corner
                        xanchor='left',
                        yanchor='top',
                        bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
                        bordercolor='rgba(0,0,0,0.2)',
                        borderwidth=1
                    ),
                    width=600, 
                    height=500,
                    margin=dict(l=80, r=80, t=100, b=80),
                    plot_bgcolor='white',  # Remove gray background
                    paper_bgcolor='white'  # Remove gray background from paper
                )
                
                # Save files
                clean_var_name = variable.replace(' ', '_').replace('/', '_').replace('\\', '_')
                base_path = plots_dir / f"{clean_var_name}_{metric}_boxplot"
                
                # Save HTML (interactive)
                # fig.write_html(f"{base_path}.html")
                
                # Try to save static images
                try:
                    # fig.write_image(f"{base_path}.png", width=600, height=500, scale=2)
                    fig.write_image(f"{base_path}.svg", width=600, height=500)
                except Exception as e:
                    print(f"    Warning: Could not save static images for {clean_var_name}_{metric}: {e}")
                    print("    Consider installing kaleido: pip install kaleido")
                
                print(f"  Saved: {clean_var_name}_{metric}_boxplot (.html, .png, .svg)")
                
            else:
                print("Plotly not available! Individual boxplots require Plotly. Install with: pip install plotly kaleido")
            total_plots += 1
    
    if PLOTLY_AVAILABLE:
        print(f"Created {total_plots} individual Plotly boxplot figures in {plots_dir}")
        print("Each plot saved as HTML (interactive), PNG, and SVG formats")
    else:
        print(f"Created {total_plots} individual matplotlib boxplot figures in {plots_dir}")
    print(f"Generated boxplots for {len(variables)} variables across {len(metrics)} metrics")

def create_cop_scatterplot_with_ellipses(df_avg, output_dir):
    """
    Create a 2D scatterplot of COP x vs COP y for each group with fitted covariance ellipses.
    Each point represents the mean COP position for one subject.
    Uses Plotly styling to match the boxplots.
    """
    print("Creating COP scatterplot with covariance ellipses...")
    
    # Create output directory
    plots_dir = Path(output_dir) / "cop_analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify COP columns
    copx_col = next((col for col in df_avg.columns if 'cop' in col.lower() and 'x' in col.lower()), None)
    copy_col = next((col for col in df_avg.columns if 'cop' in col.lower() and 'y' in col.lower()), None)
    
    if not copx_col or not copy_col:
        print("COP X or COP Y columns not found! Skipping COP scatterplot.")
        return
    
    print(f"Found COP columns: {copx_col}, {copy_col}")
    
    # Average COP values for each subject (across all time points)
    subject_means = df_avg.groupby(['Subject_ID', 'Group'])[[copx_col, copy_col]].mean().reset_index()
    
    # Print statistics of COP centroids
    print("\n" + "-"*60)
    print("COP CENTROID STATISTICS")
    print("-"*60)
    
    # Overall statistics (across all subjects)
    print("\nOVERALL STATISTICS (all subjects):")
    overall_x_median = subject_means[copx_col].median()
    overall_x_q25 = subject_means[copx_col].quantile(0.25)
    overall_x_q75 = subject_means[copx_col].quantile(0.75)
    overall_x_iqr = overall_x_q75 - overall_x_q25
    
    overall_y_median = subject_means[copy_col].median()
    overall_y_q25 = subject_means[copy_col].quantile(0.25)
    overall_y_q75 = subject_means[copy_col].quantile(0.75)
    overall_y_iqr = overall_y_q75 - overall_y_q25
    
    print(f"  COP X position: Median = {overall_x_median:.3f} mm, IQR = {overall_x_iqr:.3f} mm [{overall_x_q25:.3f}-{overall_x_q75:.3f}]")
    print(f"  COP Y position: Median = {overall_y_median:.3f} mm, IQR = {overall_y_iqr:.3f} mm [{overall_y_q25:.3f}-{overall_y_q75:.3f}]")
    print(f"  Total subjects: {len(subject_means)}")
    
    # Statistics grouped by group
    print("\nSTATISTICS BY GROUP:")
    group_map = {'sedentari': 'Sedentary', 'sportivi': 'Athletes', 'slackliner': 'Slackliners'}
    
    for group in ['sedentari', 'sportivi', 'slackliner']:
        group_data = subject_means[subject_means['Group'] == group]
        if len(group_data) > 0:
            x_median = group_data[copx_col].median()
            x_q25 = group_data[copx_col].quantile(0.25)
            x_q75 = group_data[copx_col].quantile(0.75)
            x_iqr = x_q75 - x_q25
            
            y_median = group_data[copy_col].median()
            y_q25 = group_data[copy_col].quantile(0.25)
            y_q75 = group_data[copy_col].quantile(0.75)
            y_iqr = y_q75 - y_q25
            
            print(f"\n  {group_map[group]} (n={len(group_data)}):")
            print(f"    COP X position: Median = {x_median:.3f} mm, IQR = {x_iqr:.3f} mm [{x_q25:.3f}-{x_q75:.3f}]")
            print(f"    COP Y position: Median = {y_median:.3f} mm, IQR = {y_iqr:.3f} mm [{y_q25:.3f}-{y_q75:.3f}]")
    
    print("-"*60 + "\n")
    
    colors = ['red', 'blue', 'green']  # Same as boxplots
    
    if PLOTLY_AVAILABLE:
        # Create Plotly figure (preferred - matches boxplot style)
        fig = go.Figure()
        
        # Add reference lines at origin
        fig.add_hline(y=0, line=dict(color='grey', dash='dash', width=1), opacity=0.7)
        fig.add_vline(x=0, line=dict(color='grey', dash='dash', width=1), opacity=0.7)
        
        # # Add 'limits' text annotation
        # fig.add_annotation(x=30, y=2, text='35 cm', showarrow=False, 
        #                   font=dict(size=16, color='grey'), opacity=0.9)
        
        # Process groups in the same order as boxplots
        for i, group in enumerate(['sedentari', 'sportivi', 'slackliner']):
            group_data = subject_means[subject_means['Group'] == group]
            
            if len(group_data) > 0:
                x_data = group_data[copx_col].values
                y_data = group_data[copy_col].values
                
                # Different marker symbols for each group
                marker_symbols = ['circle', 'square', 'triangle-up']
                
                # Add scatter plot
                fig.add_trace(go.Scatter(
                    x=x_data, 
                    y=y_data,
                    mode='markers',
                    marker=dict(
                        color=colors[i],
                        symbol=marker_symbols[i],
                        size=10,
                        opacity=0.7,
                        line=dict(color='black', width=1)
                    ),
                    name=group_map[group],
                    showlegend=True
                ))
                
                # Fit covariance ellipse if we have enough points
                if len(x_data) >= 2:
                    # Calculate mean and covariance
                    mean_x, mean_y = np.mean(x_data), np.mean(y_data)
                    cov_matrix = np.cov(x_data, y_data)
                    
                    # Calculate ellipse parameters
                    eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
                    order = eigenvals.argsort()[::-1]
                    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
                    
                    # Calculate ellipse angle
                    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
                    
                    # Calculate ellipse width and height (2 standard deviations)
                    width, height = 2 * np.sqrt(eigenvals)
                    
                    # Generate ellipse points for plotting
                    theta = np.linspace(0, 2*np.pi, 100)
                    ellipse_x = (width/2) * np.cos(theta)
                    ellipse_y = (height/2) * np.sin(theta)
                    
                    # Rotate ellipse
                    cos_angle = np.cos(np.radians(angle))
                    sin_angle = np.sin(np.radians(angle))
                    ellipse_x_rot = ellipse_x * cos_angle - ellipse_y * sin_angle + mean_x
                    ellipse_y_rot = ellipse_x * sin_angle + ellipse_y * cos_angle + mean_y
                    
                    # Add ellipse as a filled shape
                    fig.add_trace(go.Scatter(
                        x=ellipse_x_rot, 
                        y=ellipse_y_rot,
                        mode='lines',
                        fill='toself',
                        fillcolor=colors[i],
                        opacity=0.2,
                        line=dict(color=colors[i], width=2),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
                    
                    # Add center point
                    fig.add_trace(go.Scatter(
                        x=[mean_x], 
                        y=[mean_y],
                        mode='markers',
                        marker=dict(
                            color=colors[i],
                            symbol='x',
                            size=15,
                            line=dict(width=3)
                        ),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
        
        # Update layout to match boxplot style
        fig.update_layout(
            width=600,
            height=600,
            showlegend=True,
            legend=dict(
                x=0.02,  # Position legend inside plot area (same as boxplots)
                y=0.98,  # Top-left corner
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255,255,255,0.8)',  # Semi-transparent white background
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            ),
            plot_bgcolor='white',  # Remove gray background
            paper_bgcolor='white',  # Remove gray background from paper
            margin=dict(l=80, r=80, t=80, b=80)
        )
        
        # Update axes to match boxplot style
        fig.update_xaxes(
            range=[-44, 44],
            showgrid=False,  # Remove grid lines
            zeroline=False,
            linecolor='black',
            linewidth=2,
            title="COP X (mm)"
        )
        
        fig.update_yaxes(
            range=[-44, 44],
            showgrid=False,  # Remove grid lines
            zeroline=False,
            linecolor='black',
            linewidth=2,
            title="COP Y (mm)",
            scaleanchor="x",  # Keep square aspect ratio
            scaleratio=1
        )
        
        # Save files
        cop_plot_path = plots_dir / "cop_scatterplot_with_ellipses"
        
        # Save PNG and SVG
        try:
            fig.write_image(f"{cop_plot_path}.png", width=600, height=600, scale=2)
            fig.write_image(f"{cop_plot_path}.svg", width=600, height=600)
            print(f"Saved COP scatterplot: {cop_plot_path}.png")
            print(f"Saved COP scatterplot SVG: {cop_plot_path}.svg")
        except Exception as e:
            print(f"Warning: Could not save static images: {e}")
            print("Consider installing kaleido: pip install kaleido")
        
        # Save HTML (interactive)
        fig.write_html(f"{cop_plot_path}.html")
        print(f"Saved COP scatterplot HTML: {cop_plot_path}.html")
    
    else:
        print("Plotly not available! COP scatterplot requires Plotly. Install with: pip install plotly kaleido")

def perform_comprehensive_statistical_analysis(metrics_df):
    """
    Perform Kruskal-Wallis tests and post-hoc comparisons for all variables and all metrics.
    Returns a summary DataFrame with all statistics.
    """
    print("Performing comprehensive statistical analysis for all metrics...")
    
    # Group mapping - Updated order: Sedentary, Athletes, Slackliners
    group_map = {'sedentari': 'Sedentary', 'sportivi': 'Athletes', 'slackliner': 'Slackliners'}
    
    results_list = []
    variables = metrics_df['Variable'].unique()
    metrics = ['Q05', 'Q95', 'Mean', 'Std', 'Range', 'Median', 'IQR']
    
    for variable in variables:
        var_data = metrics_df[metrics_df['Variable'] == variable]
        var_type = var_data['Variable_Type'].iloc[0]
        
        print(f"\nAnalyzing {variable} ({var_type})...")
        
        # Analyze each metric for this variable
        for metric in metrics:
            if metric not in var_data.columns:
                continue
                
            print(f"  Processing {metric}...")
            
            # Calculate descriptive statistics for each group - same order
            group_stats = {}
            for group in ['sedentari', 'sportivi', 'slackliner']:
                group_values = var_data[var_data['Group'] == group][metric].dropna()
                if len(group_values) > 0:
                    median_val = np.median(group_values)
                    q25 = np.percentile(group_values, 25)
                    q75 = np.percentile(group_values, 75)
                    group_stats[group] = f"{median_val:.3f} [{q25:.3f}-{q75:.3f}]"
                else:
                    group_stats[group] = "N/A"
            
            # Prepare data for Kruskal-Wallis test - same order
            groups = ['sedentari', 'sportivi', 'slackliner']
            group_data = []
            group_labels = []
            
            for group in groups:
                data = var_data[var_data['Group'] == group][metric].dropna()
                if len(data) > 0:
                    group_data.append(data.values)
                    group_labels.append(group_map[group])
            
            # Initialize result dictionary with English column names
            result = {
                'Variable': variable,
                'Variable_Type': var_type,
                'Metric': metric,
                'Sedentary_Median_IQR': group_stats.get('sedentari', 'N/A'),
                'Athletes_Median_IQR': group_stats.get('sportivi', 'N/A'),
                'Slackliners_Median_IQR': group_stats.get('slackliner', 'N/A'),
                'KW_Statistic': np.nan,
                'KW_p_value': np.nan,
                'PostHoc_p_Sed_vs_Ath': np.nan,
                'PostHoc_p_Sed_vs_Slack': np.nan,
                'PostHoc_p_Ath_vs_Slack': np.nan
            }
            
            # Perform Kruskal-Wallis test if we have at least 2 groups
            if len(group_data) >= 2:
                try:
                    h_stat, p_value = kruskal(*group_data)
                    result['KW_Statistic'] = h_stat
                    result['KW_p_value'] = p_value
                    
                    print(f"    {metric} - Kruskal-Wallis: H={h_stat:.4f}, p={p_value:.6f}")
                    
                    # If significant, perform post-hoc tests
                    if p_value < 0.05 and len(group_data) == 3:
                        print(f"    {metric} - Performing post-hoc comparisons...")
                        
                        # Post-hoc comparisons with Bonferroni correction
                        # Order: [0]=Sedentary, [1]=Athletes, [2]=Slackliners
                        n_comparisons = 3
                        
                        # Sedentary vs Athletes
                        if len(group_data[0]) > 0 and len(group_data[1]) > 0:
                            _, p_val = mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
                            result['PostHoc_p_Sed_vs_Ath'] = min(p_val * n_comparisons, 1.0)
                        
                        # Sedentary vs Slackliners
                        if len(group_data[0]) > 0 and len(group_data[2]) > 0:
                            _, p_val = mannwhitneyu(group_data[0], group_data[2], alternative='two-sided')
                            result['PostHoc_p_Sed_vs_Slack'] = min(p_val * n_comparisons, 1.0)
                        
                        # Athletes vs Slackliners
                        if len(group_data[1]) > 0 and len(group_data[2]) > 0:
                            _, p_val = mannwhitneyu(group_data[1], group_data[2], alternative='two-sided')
                            result['PostHoc_p_Ath_vs_Slack'] = min(p_val * n_comparisons, 1.0)
                        
                        print(f"      Sedentary vs Athletes: p={result['PostHoc_p_Sed_vs_Ath']:.6f}")
                        print(f"      Sedentary vs Slackliners: p={result['PostHoc_p_Sed_vs_Slack']:.6f}")
                        print(f"      Athletes vs Slackliners: p={result['PostHoc_p_Ath_vs_Slack']:.6f}")
                
                except Exception as e:
                    print(f"    Error in statistical test for {metric}: {e}")
            
            results_list.append(result)
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(results_list)
    
    # Sort by variable type, variable name, and then by metric
    summary_df = summary_df.sort_values(['Variable_Type', 'Variable', 'Metric']).reset_index(drop=True)
    
    total_tests = len(summary_df)
    significant_tests = summary_df[summary_df['KW_p_value'] < 0.05]['KW_p_value'].count()
    
    print(f"\nCompleted statistical analysis:")
    print(f"  Variables analyzed: {len(variables)}")
    print(f"  Total statistical tests: {total_tests}")
    print(f"  Significant tests (p < 0.05): {significant_tests} ({significant_tests/total_tests*100:.1f}%)")
    
    return summary_df

def save_excel_summary(summary_df, output_dir, test_name):
    """
    Save the statistical summary to an Excel file with multiple sheets.
    """
    print("Saving Excel summary...")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save to Excel
    excel_path = output_path / f"{test_name}_statistical_summary.xlsx"
    
    # Create Excel writer with formatting
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Save complete summary
        summary_df.to_excel(writer, sheet_name='Complete_Analysis', index=False)
        
        # Create a summary sheet with only significant results
        significant_df = summary_df[summary_df['KW_p_value'] < 0.05].copy()
        if not significant_df.empty:
            significant_df.to_excel(writer, sheet_name='Significant_Results', index=False)
        
        # Create a pivot-like summary by variable type
        variable_summary = []
        for var_type in summary_df['Variable_Type'].unique():
            type_data = summary_df[summary_df['Variable_Type'] == var_type]
            total_tests = len(type_data)
            significant_tests = len(type_data[type_data['KW_p_value'] < 0.05])
            unique_vars = type_data['Variable'].nunique()
            vars_with_sig = type_data[type_data['KW_p_value'] < 0.05]['Variable'].nunique()
            
            variable_summary.append({
                'Variable_Type': var_type,
                'Total_Variables': unique_vars,
                'Variables_with_Significant_Results': vars_with_sig,
                'Total_Statistical_Tests': total_tests,
                'Significant_Tests': significant_tests,
                'Percentage_Significant': f"{significant_tests/total_tests*100:.1f}%" if total_tests > 0 else "0%"
            })
        
        summary_by_type = pd.DataFrame(variable_summary)
        summary_by_type.to_excel(writer, sheet_name='Summary_by_Type', index=False)
        
        # Auto-adjust column widths for all sheets
        for sheet_name in writer.sheets:
            worksheet = writer.sheets[sheet_name]
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)
                worksheet.column_dimensions[column_letter].width = adjusted_width
    
    print(f"Saved Excel summary with multiple sheets: {excel_path}")
    print(f"  - Complete_Analysis: All {len(summary_df)} statistical tests")
    if not significant_df.empty:
        print(f"  - Significant_Results: {len(significant_df)} significant tests")
    print(f"  - Summary_by_Type: Overview by variable type")
    
    return excel_path

def main_analysis(test_number, test_name, repetitions=None, base_dir="data"):
    """
    Main function to run the comprehensive analysis.
    """
    # Set up logging
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{test_name}_comprehensive_analysis_log_{timestamp}.txt"
    logger = Logger(log_filename)
    
    # Redirect stdout to logger
    sys.stdout = logger
    
    try:
        print(f"Starting comprehensive analysis for {test_name}...")
        print(f"Test number: {test_number}")
        if repetitions:
            print(f"Repetitions: {repetitions}")
        print("-" * 50)
        
        # Create output directory
        output_dir = f"comprehensive_analysis_{test_name}_{timestamp}"
        Path(output_dir).mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
        
        # 1. Load and preprocess data
        print("\n" + "="*50)
        print("STEP 1: LOADING AND PREPROCESSING DATA")
        print("="*50)
        
        base_data_dir = Path(__file__).resolve().parent.parent / base_dir
        df_raw = load_test_data(base_data_dir, test_number, repetitions)
        
        if df_raw.empty:
            print("No data loaded! Check test number and repetitions.")
            return
        
        print(f"Raw data shape: {df_raw.shape}")
        
        # Load demographic data
        excel_path = Path(__file__).resolve().parent.parent / "demo_data.xlsx"
        weight_data = load_demographic_data(excel_path)
        
        # Preprocess data
        df_avg = preprocess_data(df_raw)
        
        # 2. Calculate comprehensive metrics
        print("\n" + "="*50)
        print("STEP 2: CALCULATING COMPREHENSIVE METRICS")
        print("="*50)
        
        metrics_df = calculate_comprehensive_metrics(df_avg, weight_data)
        
        if metrics_df is None:
            print("No metrics calculated!")
            return
        
        print(f"Metrics calculated for {metrics_df['Variable'].nunique()} variables")
        print(f"Total metric records: {len(metrics_df)}")
        
        # # 3. Create individual boxplots
        # print("\n" + "="*50)
        # print("STEP 3: CREATING INDIVIDUAL BOXPLOTS")
        # print("="*50)
        # create_individual_boxplots(metrics_df, output_dir)
        
        # 4. Create COP scatterplot with ellipses
        print("\n" + "="*50)
        print("STEP 4: CREATING COP SCATTERPLOT WITH ELLIPSES")
        print("="*50)
        create_cop_scatterplot_with_ellipses(df_avg, output_dir)
        
        # # 5. Perform statistical analysis
        # print("\n" + "="*50)
        # print("STEP 5: STATISTICAL ANALYSIS")
        # print("="*50)
        # summary_df = perform_comprehensive_statistical_analysis(metrics_df)
        
        # # 6. Save Excel summary
        # print("\n" + "="*50)
        # print("STEP 6: SAVING EXCEL SUMMARY")
        # print("="*50)
        # excel_path = save_excel_summary(summary_df, output_dir, test_name)
        
        # print("\n" + "="*80)
        # print("COMPREHENSIVE ANALYSIS COMPLETED SUCCESSFULLY!")
        # print("="*80)
        # print(f"Results saved in: {output_dir}")
        # print(f"Excel summary: {excel_path}")
        # print(f"Log file: {log_filename}")
        
        # # Print summary statistics
        # significant_tests = summary_df[summary_df['KW_p_value'] < 0.05]['KW_p_value'].count()
        # total_tests = len(summary_df)
        # unique_vars = summary_df['Variable'].nunique()
        # print(f"Significant tests: {significant_tests}/{total_tests} ({significant_tests/total_tests*100:.1f}%)")
        # print(f"Variables with at least one significant metric: {summary_df[summary_df['KW_p_value'] < 0.05]['Variable'].nunique()}/{unique_vars}")
        
    except Exception as e:
        print(f"\nERROR OCCURRED: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())
        
    finally:
        # Restore stdout and close logger
        sys.stdout = logger.terminal
        logger.close()
        logger.terminal.write(f"\nLog file saved: {log_filename}\n")

if __name__ == "__main__":
    # Example usage - modify as needed
    
    # Test 1
    main_analysis(test_number=1, test_name="Test1", repetitions=[1,2,3])

    # # Test 2
    main_analysis(test_number=2, test_name="Test2", repetitions=[1,2,3])

    # # Test 3
    # main_analysis(test_number=3, test_name="Test3", repetitions=None)
    
    # # Test 4 Right (repetitions 1-3)
    # main_analysis(test_number=4, test_name="Test4_Right", repetitions=[1, 3, 5])
    
    # # Test 4 Left (repetitions 4-6)
    # main_analysis(test_number=4, test_name="Test4_Left", repetitions=[2, 4, 6])

    # # Test 5
    # main_analysis(test_number=5, test_name="Test5", repetitions=None)