import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.piecharts import Pie
from datetime import datetime
import io
import base64
import tempfile
import os

class FootballAnalyticsPDFReport:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self.setup_custom_styles()
        
    def setup_custom_styles(self):
        """Setup custom styles for the PDF report"""
        # Title style
        self.title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=colors.HexColor('#1f77b4')
        )
        
        # Section header style
        self.section_style = ParagraphStyle(
            'SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=20,
            textColor=colors.HexColor('#1f77b4'),
            borderWidth=1,
            borderColor=colors.HexColor('#1f77b4'),
            borderPadding=5
        )
        
        # Metric style
        self.metric_style = ParagraphStyle(
            'MetricStyle',
            parent=self.styles['Normal'],
            fontSize=12,
            spaceAfter=6,
            leftIndent=20
        )
        
    def create_plotly_image(self, fig, width=800, height=500):
        """Convert Plotly figure to image for PDF"""
        img_bytes = pio.to_image(fig, format='png', width=width, height=height, scale=2)
        img_buffer = io.BytesIO(img_bytes)
        return img_buffer
        
    def generate_comprehensive_report(self, df, summary, cleaning_report, ml_analyzer=None):
        """Generate a comprehensive PDF report"""
        # Create temporary file for PDF
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        doc = SimpleDocTemplate(temp_file.name, pagesize=A4, topMargin=0.5*inch)
        
        story = []
        
        # Title Page
        story.extend(self.create_title_page(df))
        story.append(PageBreak())
        
        # Executive Summary
        story.extend(self.create_executive_summary(summary))
        story.append(PageBreak())
        
        # Team Performance Overview
        story.extend(self.create_team_overview(df, summary))
        story.append(PageBreak())
        
        # Individual Player Analysis
        story.extend(self.create_player_analysis(df))
        story.append(PageBreak())
        
        # Machine Learning Insights (if available)
        if ml_analyzer:
            story.extend(self.create_ml_insights(df, ml_analyzer))
            story.append(PageBreak())
        
        # Data Quality Report
        story.extend(self.create_data_quality_section(cleaning_report))
        story.append(PageBreak())
        
        # Recommendations
        story.extend(self.create_recommendations(df, summary))
        
        # Build PDF
        doc.build(story)
        
        return temp_file.name
    
    def create_title_page(self, df):
        """Create title page"""
        story = []
        
        # Main title
        title = Paragraph("⚽ MAS Team Analytics Report", self.title_style)
        story.append(title)
        story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = Paragraph(f"Season 2024/2025 - Generated on {datetime.now().strftime('%B %d, %Y')}", 
                            self.styles['Heading3'])
        story.append(subtitle)
        story.append(Spacer(1, 0.5*inch))
        
        # Team summary box
        team_data = [
            ['Total Players Analyzed', str(len(df))],
            ['Report Generation Date', datetime.now().strftime('%Y-%m-%d %H:%M')],
            ['Analysis Type', 'Comprehensive Performance Review'],
            ['Data Quality', 'Automatically Cleaned & Validated']
        ]
        
        team_table = Table(team_data, colWidths=[3*inch, 2*inch])
        team_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(team_table)
        story.append(Spacer(1, 1*inch))
        
        # Key highlights
        story.append(Paragraph("📊 Report Contents", self.section_style))
        highlights = [
            "• Executive Summary with Key Performance Indicators",
            "• Individual Player Performance Analysis",
            "• Team Tactical Insights and Formation Recommendations",
            "• Machine Learning Predictions and Player Clustering",
            "• Injury Risk Assessment and Workload Management",
            "• Data Quality Report and Cleaning Summary",
            "• Strategic Recommendations for Team Improvement"
        ]
        
        for highlight in highlights:
            story.append(Paragraph(highlight, self.metric_style))
        
        return story
    
    def create_executive_summary(self, summary):
        """Create executive summary section"""
        story = []
        
        story.append(Paragraph("📈 Executive Summary", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Key metrics table
        metrics_data = [
            ['Metric', 'Value', 'Status'],
            ['Total Goals', str(summary.get('total_goals', 0)), '⚽'],
            ['Total Shots', str(summary.get('total_shots', 0)), '🎯'],
            ['Conversion Rate', f"{summary.get('conversion_rate', 0):.1f}%", '📊'],
            ['Progressive Passes', str(summary.get('total_passes', 0)), '⚡'],
            ['Defensive Pressures', str(summary.get('total_pressures', 0)), '🛡️']
        ]
        
        metrics_table = Table(metrics_data, colWidths=[2*inch, 1.5*inch, 1*inch])
        metrics_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(metrics_table)
        story.append(Spacer(1, 0.5*inch))
        
        return story
    
    def create_team_overview(self, df, summary):
        """Create team performance overview"""
        story = []
        
        story.append(Paragraph("🏆 Team Performance Overview", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Top performers table
        if 'Overall Performance Score' in df.columns:
            top_performers = df.nlargest(5, 'Overall Performance Score')[['Player', 'Overall Performance Score', 'Goals', 'Progressive Passes']]
        else:
            # Fallback if no overall score
            top_performers = df.nlargest(5, 'Goals')[['Player', 'Goals', 'Progressive Passes']]
        
        # Convert to list for table
        table_data = [['Player', 'Performance Score', 'Goals', 'Progressive Passes']]
        for _, row in top_performers.iterrows():
            if 'Overall Performance Score' in df.columns:
                table_data.append([row['Player'], f"{row['Overall Performance Score']:.1f}", 
                                 str(row['Goals']), str(row['Progressive Passes'])])
            else:
                table_data.append([row['Player'], 'N/A', str(row['Goals']), str(row['Progressive Passes'])])
        
        performers_table = Table(table_data, colWidths=[2*inch, 1.5*inch, 1*inch, 1.5*inch])
        performers_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        
        story.append(Paragraph("🌟 Top 5 Performers", self.section_style))
        story.append(performers_table)
        story.append(Spacer(1, 0.5*inch))
        
        return story
    
    def create_player_analysis(self, df):
        """Create individual player analysis"""
        story = []
        
        story.append(Paragraph("👤 Individual Player Analysis", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        # Player statistics table
        player_cols = ['Player', 'Goals', 'Shots', 'Progressive Passes', 'Total Crosses', 'Pressures']
        available_cols = [col for col in player_cols if col in df.columns]
        
        if len(available_cols) > 1:
            player_data = [available_cols]
            for _, row in df.iterrows():
                player_row = [str(row[col]) if col in row else 'N/A' for col in available_cols]
                player_data.append(player_row)
            
            # Calculate column widths dynamically
            col_width = 6.5 * inch / len(available_cols)
            col_widths = [col_width] * len(available_cols)
            
            player_table = Table(player_data, colWidths=col_widths)
            player_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 9),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('FONTSIZE', (0, 1), (-1, -1), 8)
            ]))
            
            story.append(player_table)
        
        return story
    
    def create_ml_insights(self, df, ml_analyzer):
        """Create ML insights section"""
        story = []
        
        story.append(Paragraph("🤖 Machine Learning Insights", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        try:
            # Player clustering
            df_clustered, _ = ml_analyzer.player_clustering_analysis(df)
            
            story.append(Paragraph("👥 Player Type Distribution", self.section_style))
            
            if 'Player Type Name' in df_clustered.columns:
                type_counts = df_clustered['Player Type Name'].value_counts()
                type_data = [['Player Type', 'Count', 'Percentage']]
                
                for ptype, count in type_counts.items():
                    percentage = (count / len(df_clustered) * 100)
                    type_data.append([ptype, str(count), f"{percentage:.1f}%"])
                
                type_table = Table(type_data, colWidths=[2.5*inch, 1*inch, 1.5*inch])
                type_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(type_table)
                story.append(Spacer(1, 0.3*inch))
            
            # Injury risk assessment
            df_risk = ml_analyzer.injury_risk_assessment(df.copy())
            
            story.append(Paragraph("⚕️ Injury Risk Assessment", self.section_style))
            
            if 'Injury Risk Level' in df_risk.columns:
                risk_counts = df_risk['Injury Risk Level'].value_counts()
                risk_data = [['Risk Level', 'Players', 'Percentage']]
                
                for risk, count in risk_counts.items():
                    percentage = (count / len(df_risk) * 100)
                    risk_data.append([risk, str(count), f"{percentage:.1f}%"])
                
                risk_table = Table(risk_data, colWidths=[2*inch, 1*inch, 1.5*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff6b6b')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.lightgrey),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(risk_table)
            
        except Exception as e:
            story.append(Paragraph(f"ML analysis unavailable: {str(e)}", self.styles['Normal']))
        
        return story
    
    def create_data_quality_section(self, cleaning_report):
        """Create data quality section"""
        story = []
        
        story.append(Paragraph("🔍 Data Quality Report", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        total_issues = sum(len(issues) for issues in cleaning_report.values())
        
        if total_issues == 0:
            story.append(Paragraph("✅ No data quality issues found! Dataset was clean.", 
                                 self.styles['Normal']))
        else:
            story.append(Paragraph(f"🔧 Fixed {total_issues} data quality issues:", 
                                 self.styles['Normal']))
            story.append(Spacer(1, 0.2*inch))
            
            for category, issues in cleaning_report.items():
                if issues:
                    story.append(Paragraph(f"<b>{category.replace('_', ' ').title()}:</b>", 
                                         self.styles['Normal']))
                    for issue in issues:
                        story.append(Paragraph(f"• {issue}", self.metric_style))
                    story.append(Spacer(1, 0.1*inch))
        
        return story
    
    def create_recommendations(self, df, summary):
        """Create recommendations section"""
        story = []
        
        story.append(Paragraph("💡 Strategic Recommendations", self.title_style))
        story.append(Spacer(1, 0.3*inch))
        
        recommendations = []
        
        # Analyze conversion rate
        conversion_rate = summary.get('conversion_rate', 0)
        if conversion_rate < 15:
            recommendations.append("🎯 Focus on shooting accuracy training - conversion rate is below average")
        elif conversion_rate > 25:
            recommendations.append("⭐ Excellent shooting efficiency - maintain current training methods")
        
        # Analyze defensive pressure
        total_pressures = summary.get('total_pressures', 0)
        if total_pressures < 100:
            recommendations.append("🛡️ Increase defensive intensity and pressing drills")
        
        # Analyze progressive passing
        total_passes = summary.get('total_passes', 0)
        if total_passes < 500:
            recommendations.append("⚡ Work on progressive passing and ball circulation")
        
        # General recommendations
        recommendations.extend([
            "📊 Continue regular performance monitoring using this analytics platform",
            "🔄 Rotate high-risk injury players to prevent fatigue",
            "🎯 Set individual performance targets based on player type analysis",
            "📈 Track progress over multiple matches for trend analysis"
        ])
        
        for rec in recommendations:
            story.append(Paragraph(rec, self.metric_style))
            story.append(Spacer(1, 0.1*inch))
        
        return story