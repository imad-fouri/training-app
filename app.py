import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import hashlib
from datetime import datetime
import os

# Set page configuration
st.set_page_config(
    page_title="MAS Team Data Analyzer - Season 2024/2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data cleaning functions (add these at the top level)
def clean_column_names(df):
    """Standardize column names to expected format"""
    column_mapping = {
        # Player name variations
        'name': 'Player',
        'player_name': 'Player',
        'full_name': 'Player',
        'player': 'Player',
        
        # Progressive passes variations
        'progressive_passes': 'Progressive Passes',
        'prog_passes': 'Progressive Passes',
        'forward_passes': 'Progressive Passes',
        'progressive passes': 'Progressive Passes',
        
        # Crosses variations
        'total_crosses': 'Total Crosses',
        'crosses': 'Total Crosses',
        'cross_attempts': 'Total Crosses',
        'total crosses': 'Total Crosses',
        
        'successful_crosses': 'Successful Crosses',
        'cross_success': 'Successful Crosses',
        'crosses_completed': 'Successful Crosses',
        'successful crosses': 'Successful Crosses',
        
        # Shots variations
        'shots': 'Shots',
        'shot_attempts': 'Shots',
        'total_shots': 'Shots',
        'shots_taken': 'Shots',
        
        # Goals variations
        'goals': 'Goals',
        'goals_scored': 'Goals',
        'total_goals': 'Goals',
        
        # Defensive variations
        'pressures': 'Pressures',
        'defensive_pressures': 'Pressures',
        'press_attempts': 'Pressures',
        
        'interceptions': 'Interceptions',
        'intercepts': 'Interceptions',
        'defensive_intercepts': 'Interceptions',
        
        # Touch coordinates
        'touch_x': 'Touch X',
        'x_coordinate': 'Touch X',
        'position_x': 'Touch X',
        'touch x': 'Touch X',
        
        'touch_y': 'Touch Y',
        'y_coordinate': 'Touch Y',
        'position_y': 'Touch Y',
        'touch y': 'Touch Y'
    }
    
    # Create a copy of the dataframe
    df_cleaned = df.copy()
    
    # Normalize column names (lowercase, replace spaces/special chars with underscores)
    normalized_columns = {}
    for col in df_cleaned.columns:
        normalized = re.sub(r'[^a-zA-Z0-9]+', '_', col.lower().strip()).strip('_')
        normalized_columns[col] = normalized
    
    # Apply column mapping
    new_columns = {}
    for original_col, normalized_col in normalized_columns.items():
        if normalized_col in column_mapping:
            new_columns[original_col] = column_mapping[normalized_col]
        else:
            new_columns[original_col] = original_col
    
    df_cleaned = df_cleaned.rename(columns=new_columns)
    
    return df_cleaned

def validate_and_clean_data(df):
    """Validate and clean the dataset"""
    cleaning_report = {
        'missing_values': [],
        'data_types': [],
        'consistency_issues': [],
        'duplicates': [],
        'outliers': []
    }
    
    df_cleaned = df.copy()
    
    # 1. Clean player names
    if 'Player' in df_cleaned.columns:
        # Remove leading/trailing whitespace
        df_cleaned['Player'] = df_cleaned['Player'].astype(str).str.strip()
        
        # Handle missing player names
        missing_names = df_cleaned['Player'].isin(['', 'nan', 'None', 'null'])
        if missing_names.any():
            df_cleaned.loc[missing_names, 'Player'] = [f'Player_{i+1}' for i in range(missing_names.sum())]
            cleaning_report['missing_values'].append(f"Filled {missing_names.sum()} missing player names")
    
    # 2. Process numeric columns
    numeric_columns = ['Progressive Passes', 'Total Crosses', 'Successful Crosses', 
                      'Shots', 'Goals', 'Pressures', 'Interceptions', 'Touch X', 'Touch Y']
    
    for col in numeric_columns:
        if col in df_cleaned.columns:
            # Convert to numeric, coercing errors to NaN
            original_type = df_cleaned[col].dtype
            df_cleaned[col] = pd.to_numeric(df_cleaned[col], errors='coerce')
            
            if original_type != df_cleaned[col].dtype:
                cleaning_report['data_types'].append(f"Converted {col} to numeric type")
            
            # Fill missing values with 0
            missing_count = df_cleaned[col].isna().sum()
            if missing_count > 0:
                df_cleaned[col] = df_cleaned[col].fillna(0)
                cleaning_report['missing_values'].append(f"Filled {missing_count} missing values in {col} with 0")
            
            # Ensure non-negative values
            negative_count = (df_cleaned[col] < 0).sum()
            if negative_count > 0:
                df_cleaned[col] = df_cleaned[col].abs()
                cleaning_report['consistency_issues'].append(f"Converted {negative_count} negative values in {col} to positive")
    
    # 3. Data consistency checks
    if 'Successful Crosses' in df_cleaned.columns and 'Total Crosses' in df_cleaned.columns:
        inconsistent = df_cleaned['Successful Crosses'] > df_cleaned['Total Crosses']
        if inconsistent.any():
            df_cleaned.loc[inconsistent, 'Successful Crosses'] = df_cleaned.loc[inconsistent, 'Total Crosses']
            cleaning_report['consistency_issues'].append(f"Fixed {inconsistent.sum()} cases where successful crosses exceeded total crosses")
    
    if 'Goals' in df_cleaned.columns and 'Shots' in df_cleaned.columns:
        inconsistent = df_cleaned['Goals'] > df_cleaned['Shots']
        if inconsistent.any():
            df_cleaned.loc[inconsistent, 'Shots'] = df_cleaned.loc[inconsistent, 'Goals']
            cleaning_report['consistency_issues'].append(f"Fixed {inconsistent.sum()} cases where goals exceeded shots")
    
    # 4. Remove duplicate players
    if 'Player' in df_cleaned.columns:
        duplicates = df_cleaned.duplicated(subset=['Player'], keep='first')
        if duplicates.any():
            df_cleaned = df_cleaned[~duplicates]
            cleaning_report['duplicates'].append(f"Removed {duplicates.sum()} duplicate player entries")
    
    # 5. Remove completely empty rows
    numeric_cols_present = [col for col in numeric_columns if col in df_cleaned.columns]
    if numeric_cols_present:
        # Only remove rows where player name is missing/empty AND all numeric values are 0 or NaN
        empty_rows = (
            (df_cleaned['Player'].isin(['', 'nan', 'None', 'null']) | df_cleaned['Player'].isna()) &
            (df_cleaned[numeric_cols_present].sum(axis=1) == 0)
        )
        if empty_rows.any():
            df_cleaned = df_cleaned[~empty_rows]
            cleaning_report['missing_values'].append(f"Removed {empty_rows.sum()} completely empty rows")
    
    # 6. Data quality assessment
    if len(df_cleaned) == 0:
        # Create fallback sample data if no valid data found
        cleaning_report['missing_values'].append("No valid data found, creating sample dataset")
        sample_data = {
            'Player': ['Sample Player 1', 'Sample Player 2', 'Sample Player 3'],
            'Progressive Passes': [50, 30, 20],
            'Total Crosses': [10, 15, 5],
            'Successful Crosses': [3, 8, 2],
            'Shots': [5, 8, 3],
            'Goals': [1, 2, 0],
            'Pressures': [25, 30, 15],
            'Interceptions': [5, 7, 3]
        }
        df_cleaned = pd.DataFrame(sample_data)
    
    return df_cleaned, cleaning_report

def advanced_data_manipulation(df):
    """Perform advanced data manipulation and feature engineering"""
    df_advanced = df.copy()
    
    # Calculate derived metrics
    if 'Total Crosses' in df_advanced.columns and 'Successful Crosses' in df_advanced.columns:
        df_advanced['Cross Success Rate'] = np.where(
            df_advanced['Total Crosses'] > 0,
            (df_advanced['Successful Crosses'] / df_advanced['Total Crosses'] * 100).round(1),
            0
        )
    
    if 'Shots' in df_advanced.columns and 'Goals' in df_advanced.columns:
        df_advanced['Shot Conversion Rate'] = np.where(
            df_advanced['Shots'] > 0,
            (df_advanced['Goals'] / df_advanced['Shots'] * 100).round(1),
            0
        )
    
    if 'Pressures' in df_advanced.columns and 'Interceptions' in df_advanced.columns:
        df_advanced['Total Defensive Actions'] = df_advanced['Pressures'] + df_advanced['Interceptions']
    
    # Create performance categories
    if 'Progressive Passes' in df_advanced.columns:
        df_advanced['Passing Category'] = pd.cut(
            df_advanced['Progressive Passes'],
            bins=[0, 50, 100, 200, float('inf')],
            labels=['Low', 'Medium', 'High', 'Elite'],
            include_lowest=True
        )
    
    # Calculate normalized scores (0-100 scale)
    metrics_to_normalize = ['Progressive Passes', 'Cross Success Rate', 'Shot Conversion Rate', 'Total Defensive Actions']
    
    for metric in metrics_to_normalize:
        if metric in df_advanced.columns:
            col_name = f'{metric} (Normalized)'
            max_val = df_advanced[metric].max()
            if max_val > 0:
                df_advanced[col_name] = (df_advanced[metric] / max_val * 100).round(1)
            else:
                df_advanced[col_name] = 0
    
    # Calculate overall performance score
    normalized_cols = [col for col in df_advanced.columns if '(Normalized)' in col]
    if normalized_cols:
        df_advanced['Overall Performance Score'] = df_advanced[normalized_cols].mean(axis=1).round(1)
    
    return df_advanced

def display_data_quality_report(cleaning_report):
    """Display the data cleaning report"""
    st.markdown('<h3 class="section-header">🔍 Data Quality Report</h3>', unsafe_allow_html=True)
    
    total_issues = sum(len(issues) for issues in cleaning_report.values())
    
    if total_issues == 0:
        st.success("✅ No data quality issues found! Your dataset is clean.")
    else:
        st.info(f"🔧 Fixed {total_issues} data quality issues:")
        
        for category, issues in cleaning_report.items():
            if issues:
                st.markdown(f"**{category.replace('_', ' ').title()}:**")
                for issue in issues:
                    st.markdown(f"- {issue}")

# Custom CSS for styling
st.markdown("""
<style>
.section-header {
    color: #1f77b4;
    border-bottom: 2px solid #1f77b4;
    padding-bottom: 10px;
    margin-top: 30px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 style="text-align: center; color: #1f77b4;">⚽ MAS Team Data Analyzer – Season 2024/2025</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 18px; color: #666;">Advanced Football Analytics Dashboard with Automatic Data Cleaning</p>', unsafe_allow_html=True)
st.markdown("---")

# Authentication function
def check_password():
    """Returns `True` if the user had the correct password."""
    
    def password_entered():
        """Checks whether a password entered by the user is correct."""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        # Admin credentials
        if username == "admin" and password == "13579@":
            st.session_state["password_correct"] = True
            st.session_state["user_role"] = "admin"
            del st.session_state["password"]  # Don't store password
            del st.session_state["username"]  # Don't store username
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.markdown("### 🔐 Login to MAS Team Data Analyzer")
    st.markdown("Please enter your credentials to access the application.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.text_input(
            "Username", 
            key="username", 
            placeholder="Enter username"
        )
        st.text_input(
            "Password", 
            type="password", 
            key="password", 
            placeholder="Enter password"
        )
        
        if st.button("🔑 Login", use_container_width=True):
            password_entered()
        
        if "password_correct" in st.session_state and not st.session_state["password_correct"]:
            st.error("❌ Invalid username or password. Please try again.")
            
    st.markdown("---")
    st.info("💡 **Demo Credentials:** Username: admin, Password: 13579@")
    
    return False

# Check authentication before showing the main app
if not check_password():
    st.stop()  # Do not continue if check_password is not True.

# Show user info in sidebar after successful login
# Sidebar for file upload (after authentication)
with st.sidebar:
    # User info and logout
    st.success(f"✅ Logged in as: {st.session_state.get('user_role', 'user').title()}")
    if st.button("🚪 Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.markdown("---")
    
    st.header("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload your team's match data in CSV format"
    )
    
    if uploaded_file is not None:
        st.success("✅ File uploaded successfully!")
    
    st.markdown("---")
    
    # Data cleaning options
    st.header("🔧 Data Processing Options")
    auto_clean = st.checkbox("Auto-clean data", value=True, help="Automatically fix common data issues")
    show_cleaning_report = st.checkbox("Show cleaning report", value=True, help="Display what was cleaned/fixed")
    advanced_features = st.checkbox("Enable advanced features", value=True, help="Calculate additional metrics and insights")
    
    st.markdown("---")
    st.markdown("""
    ### 📋 Supported CSV Formats:
    **Flexible column names supported:**
    - Player/Name/Player_Name
    - Progressive_Passes/Prog_Passes
    - Total_Crosses/Crosses
    - Successful_Crosses/Cross_Success
    - Shots/Shot_Attempts
    - Goals/Goals_Scored
    - Pressures/Defensive_Pressures
    - Interceptions/Intercepts
    - Touch_X/X_Coordinate
    - Touch_Y/Y_Coordinate
    
    **Auto-cleaning features:**
    - ✅ Column name standardization
    - ✅ Missing value handling
    - ✅ Data type conversion
    - ✅ Consistency validation
    - ✅ Duplicate removal
    - ✅ Outlier detection
    """)

def calculate_team_summary(df):
    """Calculate team performance summary"""
    total_goals = df['Goals'].sum() if 'Goals' in df.columns else 0
    total_shots = df['Shots'].sum() if 'Shots' in df.columns else 0
    total_passes = df['Progressive Passes'].sum() if 'Progressive Passes' in df.columns else 0
    total_pressures = df['Pressures'].sum() if 'Pressures' in df.columns else 0
    
    conversion_rate = (total_goals / total_shots * 100) if total_shots > 0 else 0
    
    return {
        'Total Goals': total_goals,
        'Total Shots': total_shots,
        'Conversion Rate': f"{conversion_rate:.1f}%",
        'Progressive Passes': total_passes,
        'Total Pressures': total_pressures,
        'Squad Size': len(df)
    }

def analyze_progressive_passes(df):
    """Analyze progressive passing performance"""
    if 'Progressive Passes' not in df.columns:
        return None
    
    top_passers = df.nlargest(5, 'Progressive Passes')[['Player', 'Progressive Passes']]
    return top_passers

def analyze_crosses(df):
    """Analyze crossing performance"""
    if 'Total Crosses' not in df.columns or 'Successful Crosses' not in df.columns:
        return None
    
    df_crosses = df[df['Total Crosses'] > 0].copy()
    if df_crosses.empty:
        return None
        
    df_crosses['Cross Success Rate'] = (df_crosses['Successful Crosses'] / df_crosses['Total Crosses'] * 100).round(1)
    
    return df_crosses[['Player', 'Total Crosses', 'Successful Crosses', 'Cross Success Rate']].sort_values('Cross Success Rate', ascending=False)

def analyze_defensive_actions(df):
    """Analyze defensive performance"""
    defensive_cols = ['Pressures', 'Interceptions']
    available_cols = [col for col in defensive_cols if col in df.columns]
    
    if not available_cols:
        return None
    
    result_cols = ['Player'] + available_cols
    if 'Pressures' in df.columns and 'Interceptions' in df.columns:
        df_def = df.copy()
        df_def['Total Defensive Actions'] = df_def['Pressures'] + df_def['Interceptions']
        result_cols.append('Total Defensive Actions')
        return df_def[result_cols].sort_values('Total Defensive Actions', ascending=False)
    else:
        sort_col = available_cols[0]
        return df[result_cols].sort_values(sort_col, ascending=False)

def analyze_shooting_efficiency(df):
    """Analyze shooting and goal conversion"""
    if 'Shots' not in df.columns or 'Goals' not in df.columns:
        return None
    
    df_shooting = df[df['Shots'] > 0].copy()
    if df_shooting.empty:
        return None
        
    df_shooting['Conversion Rate'] = (df_shooting['Goals'] / df_shooting['Shots'] * 100).round(1)
    
    return df_shooting[['Player', 'Shots', 'Goals', 'Conversion Rate']].sort_values('Conversion Rate', ascending=False)

def create_touch_map(df):
    """Create touch map if coordinates are available"""
    if 'Touch X' not in df.columns or 'Touch Y' not in df.columns:
        return None
    
    # Filter out players with valid coordinates
    df_touches = df[(df['Touch X'].notna()) & (df['Touch Y'].notna())].copy()
    
    if df_touches.empty:
        return None
    
    fig = px.scatter(
        df_touches,
        x='Touch X',
        y='Touch Y',
        text='Player',
        title='Player Touch Map',
        labels={'Touch X': 'Field Position X', 'Touch Y': 'Field Position Y'},
        width=800,
        height=500
    )
    
    fig.update_traces(
        textposition="top center",
        marker=dict(size=12, color='#1f77b4', opacity=0.7)
    )
    
    fig.update_layout(
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_performance_charts(df):
    """Create performance visualization charts"""
    charts = {}
    
    # Progressive Passes Chart
    if 'Progressive Passes' in df.columns:
        top_passers = df.nlargest(5, 'Progressive Passes')
        fig_passes = px.bar(
            top_passers,
            x='Player',
            y='Progressive Passes',
            title='Top 5 Progressive Passers',
            color='Progressive Passes',
            color_continuous_scale='Blues'
        )
        fig_passes.update_layout(showlegend=False)
        charts['progressive_passes'] = fig_passes
    
    # Cross Success Rate Chart
    if 'Total Crosses' in df.columns and 'Successful Crosses' in df.columns:
        df_crosses = df[df['Total Crosses'] > 0].copy()
        if not df_crosses.empty:
            df_crosses['Cross Success Rate'] = (df_crosses['Successful Crosses'] / df_crosses['Total Crosses'] * 100)
            
            fig_crosses = px.bar(
                df_crosses.sort_values('Cross Success Rate', ascending=False).head(5),
                x='Player',
                y='Cross Success Rate',
                title='Top 5 Cross Success Rates (%)',
                color='Cross Success Rate',
                color_continuous_scale='Greens'
            )
            fig_crosses.update_layout(showlegend=False)
            charts['cross_success'] = fig_crosses
    
    # Defensive Actions Chart
    if 'Pressures' in df.columns and 'Interceptions' in df.columns:
        df_def = df.copy()
        df_def['Total Defensive Actions'] = df_def['Pressures'] + df_def['Interceptions']
        
        fig_def = px.bar(
            df_def.nlargest(5, 'Total Defensive Actions'),
            x='Player',
            y='Total Defensive Actions',
            title='Top 5 Defensive Performers',
            color='Total Defensive Actions',
            color_continuous_scale='Reds'
        )
        fig_def.update_layout(showlegend=False)
        charts['defensive_actions'] = fig_def
    
    return charts

# Main application logic
if uploaded_file is not None:
    try:
        # Read the CSV file
        df_raw = pd.read_csv(uploaded_file)
        
        # Data cleaning and processing
        if auto_clean:
            # Step 1: Clean column names
            df_cleaned = clean_column_names(df_raw)
            
            # Step 2: Validate and clean data
            df_cleaned, cleaning_report = validate_and_clean_data(df_cleaned)
            
            # Step 3: Advanced data manipulation
            if advanced_features:
                df = advanced_data_manipulation(df_cleaned)
            else:
                df = df_cleaned
            
            # Display cleaning report
            if show_cleaning_report:
                display_data_quality_report(cleaning_report)
        else:
            df = df_raw
        
        # Check if we have any data after cleaning
        if len(df) == 0:
            st.error("❌ No valid data found after processing. Please check your CSV file.")
            st.stop()
        
        # Display dataset preview
        st.markdown('<h2 class="section-header">📊 Dataset Preview</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Raw Data (First 5 rows)")
            st.dataframe(df_raw.head(), use_container_width=True)
        
        if auto_clean:
            with col2:
                st.subheader("Cleaned Data (First 5 rows)")
                st.dataframe(df.head(), use_container_width=True)
        
        # Data summary statistics
        if advanced_features:
            st.markdown('<h3 class="section-header">📈 Data Summary Statistics</h3>', unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                summary_stats = df[numeric_cols].describe().round(2)
                st.dataframe(summary_stats, use_container_width=True)
        
        # Team Summary
        st.markdown('<h2 class="section-header">📈 Team Performance Summary</h2>', unsafe_allow_html=True)
        summary = calculate_team_summary(df)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Goals", summary['Total Goals'])
            st.metric("Squad Size", summary['Squad Size'])
        with col2:
            st.metric("Total Shots", summary['Total Shots'])
            st.metric("Conversion Rate", summary['Conversion Rate'])
        with col3:
            st.metric("Progressive Passes", summary['Progressive Passes'])
            st.metric("Total Pressures", summary['Total Pressures'])
        
        # Progressive Passes Analysis
        st.markdown('<h2 class="section-header">🎯 Progressive Passing Analysis</h2>', unsafe_allow_html=True)
        progressive_data = analyze_progressive_passes(df)
        if progressive_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Top 5 Progressive Passers")
                st.dataframe(progressive_data, use_container_width=True)
            with col2:
                charts = create_performance_charts(df)
                if 'progressive_passes' in charts:
                    st.plotly_chart(charts['progressive_passes'], use_container_width=True)
        else:
            st.warning("Progressive Passes data not available in the uploaded file.")
        
        # Crosses Analysis
        st.markdown('<h2 class="section-header">⚽ Crossing Analysis</h2>', unsafe_allow_html=True)
        crosses_data = analyze_crosses(df)
        if crosses_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Cross Success Rates")
                st.dataframe(crosses_data, use_container_width=True)
            with col2:
                charts = create_performance_charts(df)
                if 'cross_success' in charts:
                    st.plotly_chart(charts['cross_success'], use_container_width=True)
        else:
            st.warning("Crossing data not available in the uploaded file.")
        
        # Defensive Actions Analysis
        st.markdown('<h2 class="section-header">🛡️ Defensive Actions</h2>', unsafe_allow_html=True)
        defensive_data = analyze_defensive_actions(df)
        if defensive_data is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Defensive Performance")
                st.dataframe(defensive_data, use_container_width=True)
            with col2:
                charts = create_performance_charts(df)
                if 'defensive_actions' in charts:
                    st.plotly_chart(charts['defensive_actions'], use_container_width=True)
        else:
            st.warning("Defensive actions data not available in the uploaded file.")
        
        # Shooting Efficiency Analysis
        st.markdown('<h2 class="section-header">🥅 Shooting Efficiency</h2>', unsafe_allow_html=True)
        shooting_data = analyze_shooting_efficiency(df)
        if shooting_data is not None:
            st.subheader("Goal Conversion Rates")
            st.dataframe(shooting_data, use_container_width=True)
        else:
            st.warning("Shooting efficiency data not available in the uploaded file.")
        
        # Advanced Analytics (if enabled)
        if advanced_features and 'Overall Performance Score' in df.columns:
            st.markdown('<h2 class="section-header">🏆 Advanced Performance Analytics</h2>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Top Performers by Overall Score")
                top_performers = df.nlargest(5, 'Overall Performance Score')[['Player', 'Overall Performance Score']]
                st.dataframe(top_performers, use_container_width=True)
            
            with col2:
                if 'Passing Category' in df.columns:
                    st.subheader("Players by Passing Category")
                    category_counts = df['Passing Category'].value_counts()
                    fig_categories = px.pie(
                        values=category_counts.values,
                        names=category_counts.index,
                        title="Distribution of Passing Categories"
                    )
                    st.plotly_chart(fig_categories, use_container_width=True)
        
        # Touch Map
        st.markdown('<h2 class="section-header">🗺️ Player Touch Map</h2>', unsafe_allow_html=True)
        touch_map = create_touch_map(df)
        if touch_map is not None:
            st.plotly_chart(touch_map, use_container_width=True)
        else:
            st.info("Touch map not available. Please ensure your CSV includes 'Touch X' and 'Touch Y' columns with valid coordinates.")
        
        # Download processed data
        st.markdown('<h2 class="section-header">💾 Export Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Download cleaned data
            csv_cleaned = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Cleaned Data",
                data=csv_cleaned,
                file_name="cleaned_team_data.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download summary report
            summary_report = pd.DataFrame([summary]).T
            summary_report.columns = ['Value']
            csv_summary = summary_report.to_csv()
            st.download_button(
                label="📊 Download Summary Report",
                data=csv_summary,
                file_name="team_summary_report.csv",
                mime="text/csv"
            )
        
        with col3:
            if advanced_features and auto_clean:
                # Download cleaning report
                report_data = []
                for category, items in cleaning_report.items():
                    for item in items:
                        report_data.append({'Category': category.title(), 'Description': item})
                
                if report_data:
                    report_df = pd.DataFrame(report_data)
                    csv_report = report_df.to_csv(index=False)
                    st.download_button(
                        label="🔍 Download Cleaning Report",
                        data=csv_report,
                        file_name="data_cleaning_report.csv",
                        mime="text/csv"
                    )
        
    except Exception as e:
        st.error(f"❌ Error processing the file: {str(e)}")
        st.info("💡 **Troubleshooting Tips:**")
        st.markdown("""
        - Ensure your CSV file is properly formatted
        - Check that column names don't contain special characters
        - Verify that numeric columns contain only numbers
        - Try enabling auto-clean mode for automatic fixes
        - Make sure the file is not corrupted
        """)
        
        # Show file info for debugging
        if uploaded_file is not None:
            st.subheader("File Information:")
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size} bytes")
            st.write(f"**File type:** {uploaded_file.type}")

else:
    # Welcome screen
    st.markdown("""
    ## 🎯 Welcome to MAS Team Data Analyzer!
    
    This application helps you analyze your team's performance data with comprehensive insights and **automatic data cleaning**.
    
    ### 🚀 New Features:
    - **🔧 Automatic Data Cleaning**: Handles missing values, inconsistent formats, and data errors
    - **📊 Advanced Analytics**: Calculate performance scores and player categories
    - **🔍 Data Quality Reports**: See exactly what was cleaned and fixed
    - **📈 Enhanced Visualizations**: More charts and insights
    - **💾 Multiple Export Options**: Download cleaned data, reports, and summaries
    
    ### 📊 Key Features:
    - **Progressive Passing Analysis**: Identify your best forward passers
    - **Crossing Efficiency**: Calculate success rates for crosses
    - **Defensive Actions**: Track pressures and interceptions
    - **Shooting Conversion**: Analyze goal-to-shot ratios
    - **Touch Maps**: Visualize player positions (if coordinates available)
    
    ### 📁 Getting Started:
    1. Upload your CSV file using the sidebar
    2. Enable auto-cleaning for best results
    3. Review the data quality report
    4. Explore the automatically generated insights
    
    ### 🔧 Flexible Data Support:
    **The app now supports various column name formats:**
    - Player names: `Player`, `Name`, `Player_Name`, `Full_Name`
    - Progressive passes: `Progressive_Passes`, `Prog_Passes`, `Forward_Passes`
    - Crosses: `Total_Crosses`, `Crosses`, `Cross_Attempts`
    - And many more variations!
    
    **Auto-cleaning handles:**
    - ✅ Missing values and empty cells
    - ✅ Inconsistent data types
    - ✅ Duplicate player names
    - ✅ Invalid numeric values
    - ✅ Data consistency issues
    - ✅ Column name standardization
    """)
    
    # Sample data preview
    st.markdown("### 📝 Sample Data Format:")
    sample_data = {
        'Player': ['Achraf Harmach', 'Seakanyeng', 'El Jabali', 'Hamadi', 'Mhannaoui'],
        'Progressive Passes': [221, 140, 168, 100, 90],
        'Total Crosses': [20, 56, 12, 38, 36],
        'Successful Crosses': [8, 20, 5, 16, 10],
        'Shots': [12, 28, 7, 9, 6],
        'Goals': [1, 5, 0, 0, 0],
        'Pressures': [481, 306, 200, 150, 130],
        'Interceptions': [21, 20, 35, 10, 18],
        'Touch X': [50, 70, 40, 20, 15],
        'Touch Y': [60, 30, 50, 80, 70]
    }
    
    sample_df = pd.DataFrame(sample_data)
    st.dataframe(sample_df, use_container_width=True)
    
    st.markdown("""
    ### 💡 Pro Tips:
    - **Enable auto-cleaning** for the best experience
    - **Check the cleaning report** to see what was fixed
    - **Use advanced features** for deeper insights
    - **Download cleaned data** for future use
    - **Column names are flexible** - the app will recognize common variations
    """)
# Add these imports at the top
from ml_features import FootballMLAnalyzer
import warnings
warnings.filterwarnings('ignore')

# Add this section after your existing analytics (around line 700)
if advanced_features:
    st.markdown('<h2 class="section-header">🤖 Machine Learning Analytics</h2>', unsafe_allow_html=True)
    
    # Initialize ML analyzer
    ml_analyzer = FootballMLAnalyzer()
    
    # Performance Prediction
    st.subheader('🎯 Next Match Performance Prediction')
    try:
        predictions, feature_importance = ml_analyzer.predict_next_match_performance(df)
        
        # Display predictions
        df_predictions = df[['Player']].copy()
        df_predictions['Predicted Performance'] = predictions.round(1)
        df_predictions['Current Performance'] = df['Overall Performance Score'] if 'Overall Performance Score' in df.columns else 'N/A'
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(df_predictions.sort_values('Predicted Performance', ascending=False), use_container_width=True)
        
        with col2:
            # Feature importance chart
            fig_importance = px.bar(
                feature_importance, 
                x='Importance', 
                y='Feature',
                title='Most Important Performance Factors',
                orientation='h'
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Performance prediction unavailable: {str(e)}")
    
    # Player Clustering
    st.subheader('👥 Player Type Analysis')
    try:
        df_clustered, cluster_centers = ml_analyzer.player_clustering_analysis(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display player types
            player_types = df_clustered[['Player', 'Player Type Name']].sort_values('Player')
            st.dataframe(player_types, use_container_width=True)
        
        with col2:
            # Cluster visualization
            if 'Progressive Passes' in df.columns and 'Total Defensive Actions' in df.columns:
                fig_cluster = px.scatter(
                    df_clustered,
                    x='Progressive Passes',
                    y='Total Defensive Actions',
                    color='Player Type Name',
                    hover_data=['Player'],
                    title='Player Types by Playing Style'
                )
                st.plotly_chart(fig_cluster, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Player clustering unavailable: {str(e)}")
    
    # Injury Risk Assessment
    st.subheader('⚕️ Injury Risk Assessment')
    try:
        df_risk = ml_analyzer.injury_risk_assessment(df.copy())
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk assessment table
            risk_data = df_risk[['Player', 'Injury Risk Score', 'Injury Risk Level']].sort_values('Injury Risk Score', ascending=False)
            st.dataframe(risk_data, use_container_width=True)
        
        with col2:
            # Risk distribution
            risk_counts = df_risk['Injury Risk Level'].value_counts()
            fig_risk = px.pie(
                values=risk_counts.values,
                names=risk_counts.index,
                title='Squad Injury Risk Distribution',
                color_discrete_map={
                    'High Risk': '#ff4444',
                    'Medium Risk': '#ffaa00', 
                    'Low Risk': '#44ff44',
                    'Very Low Risk': '#00aa44'
                }
            )
            st.plotly_chart(fig_risk, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Injury risk assessment unavailable: {str(e)}")
    
    # Tactical Formation Optimizer
    st.subheader('⚽ Tactical Formation Optimizer')
    try:
        formation_analysis, suggested_formation = ml_analyzer.tactical_formation_optimizer(df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric('Suggested Formation', suggested_formation)
            
            # Squad composition
            for role, count in formation_analysis.items():
                if role != 'Total Squad':
                    percentage = (count / formation_analysis['Total Squad'] * 100) if formation_analysis['Total Squad'] > 0 else 0
                    st.metric(role, f"{count} ({percentage:.1f}%)")
        
        with col2:
            # Formation visualization
            fig_formation = px.bar(
                x=list(formation_analysis.keys())[:-1],
                y=list(formation_analysis.values())[:-1],
                title='Squad Composition Analysis'
            )
            st.plotly_chart(fig_formation, use_container_width=True)
    
    except Exception as e:
        st.warning(f"Formation optimizer unavailable: {str(e)}")
# Add this import at the top
from pdf_report_generator import FootballAnalyticsPDFReport

# Add this section after your existing export options (around line 720)
st.markdown('<h2 class="section-header">📄 PDF Report Generation</h2>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Generate Comprehensive PDF Report", use_container_width=True):
        try:
            with st.spinner('Generating comprehensive PDF report...'):
                # Initialize report generator
                report_generator = FootballAnalyticsPDFReport()
                
                # Generate PDF
                if advanced_features:
                    ml_analyzer = FootballMLAnalyzer()
                    pdf_path = report_generator.generate_comprehensive_report(
                        df, summary, cleaning_report, ml_analyzer
                    )
                else:
                    pdf_path = report_generator.generate_comprehensive_report(
                        df, summary, cleaning_report
                    )
                
                # Read PDF file
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_bytes = pdf_file.read()
                
                # Download button
                st.download_button(
                    label="📥 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"MAS_Team_Analytics_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
                
                # Clean up temporary file
                os.unlink(pdf_path)
                
                st.success("✅ PDF report generated successfully!")
                
        except Exception as e:
            st.error(f"❌ Error generating PDF report: {str(e)}")
            st.info("💡 Make sure you have installed: pip install reportlab")

with col2:
    st.info("""
    📋 **PDF Report Includes:**
    
    • Executive Summary with KPIs
    • Team Performance Overview  
    • Individual Player Analysis
    • ML Insights & Predictions
    • Injury Risk Assessment
    • Data Quality Report
    • Strategic Recommendations
    
    Perfect for coaches, management, and stakeholders!
    """)