import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

class FootballMLAnalyzer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.performance_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.position_classifier = GradientBoostingClassifier(random_state=42)
        self.kmeans = KMeans(n_clusters=4, random_state=42)
        
    def predict_next_match_performance(self, df):
        """Predict player performance for next match based on historical data"""
        features = ['Progressive Passes', 'Total Crosses', 'Successful Crosses', 
                   'Shots', 'Goals', 'Pressures', 'Interceptions']
        
        # Create features for prediction
        X = df[features].fillna(0)
        
        # Simulate historical performance (in real app, you'd have multiple matches)
        # For demo, we'll use current stats to predict "next match" performance
        y = df['Overall Performance Score'] if 'Overall Performance Score' in df.columns else X.mean(axis=1)
        
        # Train model
        self.performance_model.fit(X, y)
        
        # Predict next match performance
        predictions = self.performance_model.predict(X)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': self.performance_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return predictions, feature_importance
    
    def player_clustering_analysis(self, df):
        """Cluster players based on playing style"""
        features = ['Progressive Passes', 'Cross Success Rate', 'Shot Conversion Rate', 
                   'Total Defensive Actions']
        
        # Prepare data
        X = df[features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster labels
        df_clustered = df.copy()
        df_clustered['Player Type'] = clusters
        
        # Define cluster names based on characteristics
        cluster_names = {
            0: 'Defensive Specialist',
            1: 'Creative Midfielder', 
            2: 'Goal Scorer',
            3: 'All-Rounder'
        }
        
        df_clustered['Player Type Name'] = df_clustered['Player Type'].map(cluster_names)
        
        return df_clustered, self.kmeans.cluster_centers_
    
    def injury_risk_assessment(self, df):
        """Assess injury risk based on workload metrics"""
        # Calculate workload score
        workload_features = ['Pressures', 'Total Defensive Actions', 'Progressive Passes']
        
        workload_score = df[workload_features].sum(axis=1)
        
        # Normalize to 0-100 scale
        max_workload = workload_score.max()
        normalized_workload = (workload_score / max_workload * 100) if max_workload > 0 else 0
        
        # Define risk categories
        def get_risk_level(score):
            if score >= 80:
                return 'High Risk'
            elif score >= 60:
                return 'Medium Risk'
            elif score >= 40:
                return 'Low Risk'
            else:
                return 'Very Low Risk'
        
        df['Injury Risk Score'] = normalized_workload
        df['Injury Risk Level'] = normalized_workload.apply(get_risk_level)
        
        return df
    
    def tactical_formation_optimizer(self, df):
        """Suggest optimal formation based on player strengths"""
        # Analyze player strengths
        attacking_players = df[df['Goals'] + df['Shots'] > df['Goals'].median() + df['Shots'].median()]
        defensive_players = df[df['Total Defensive Actions'] > df['Total Defensive Actions'].median()]
        creative_players = df[df['Progressive Passes'] > df['Progressive Passes'].median()]
        
        formation_analysis = {
            'Attacking Players': len(attacking_players),
            'Defensive Players': len(defensive_players),
            'Creative Players': len(creative_players),
            'Total Squad': len(df)
        }
        
        # Suggest formation based on player distribution
        if len(attacking_players) >= 3:
            suggested_formation = '4-3-3 (Attacking)'
        elif len(defensive_players) >= 6:
            suggested_formation = '5-4-1 (Defensive)'
        elif len(creative_players) >= 4:
            suggested_formation = '4-4-2 (Balanced)'
        else:
            suggested_formation = '4-5-1 (Midfield Control)'
        
        return formation_analysis, suggested_formation