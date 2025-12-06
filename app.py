# app.py
"""
🏗️ ULTIMATE CONSTRUCTION INSPECTION DASHBOARD
Advanced Analytics with Keyword Extraction & Topic Modeling

Business Intelligence Project - Premium Edition
"""

import dash
from dash import dcc, html, dash_table, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime
import json

# Import custom modules
from data_processor import ConstructionInspectionAnalyzer, analyzer

# ============================================
# INITIALIZE APPLICATION
# ============================================

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
        'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap'
    ],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": "Advanced Construction Inspection Analytics Dashboard"}
    ],
    title="🏗️ Construction Inspection Dashboard",
    suppress_callback_exceptions=True
)

server = app.server

# ============================================
# GENERATE AND PROCESS DATA
# ============================================

print("🔄 Initializing dashboard and generating sample data...")

# Generate sample data
df = analyzer.generate_sample_data(n_reports=500)

# Extract keywords
keywords_df, tfidf_matrix = analyzer.extract_keywords(df)

# Perform topic modeling
topics, topic_distribution, df = analyzer.perform_topic_modeling(df, n_topics=8)

# Calculate statistics
stats = analyzer.calculate_statistics(df)

print("✅ Data processing complete!")

# ============================================
# COLOR SCHEMES
# ============================================

COLORS = {
    'primary': '#667eea',
    'secondary': '#764ba2',
    'accent': '#4facfe',
    'success': '#38ef7d',
    'warning': '#ffd200',
    'danger': '#f45c43',
    'background': '#0a0a0f',
    'card': 'rgba(20, 20, 35, 0.8)',
    'text': '#ffffff',
    'text_secondary': 'rgba(255, 255, 255, 0.7)',
    'gradient_colors': ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe', '#38ef7d', '#11998e']
}

TOPIC_COLORS = {
    'Safety & Compliance': '#eb3349',
    'Structural Work': '#667eea',
    'Electrical Systems': '#ffd200',
    'Plumbing & Water': '#4facfe',
    'Quality Control': '#38ef7d',
    'Progress Tracking': '#f093fb',
    'Material & Resources': '#764ba2',
    'Environmental & HVAC': '#00f2fe'
}

# ============================================
# CHART TEMPLATE
# ============================================

chart_template = {
    'layout': {
        'paper_bgcolor': 'rgba(0,0,0,0)',
        'plot_bgcolor': 'rgba(0,0,0,0)',
        'font': {'family': 'Inter, sans-serif', 'color': '#ffffff'},
        'title': {'font': {'size': 18, 'color': '#ffffff'}},
        'xaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.1)',
            'tickfont': {'color': 'rgba(255,255,255,0.7)'}
        },
        'yaxis': {
            'gridcolor': 'rgba(255,255,255,0.1)',
            'zerolinecolor': 'rgba(255,255,255,0.1)',
            'tickfont': {'color': 'rgba(255,255,255,0.7)'}
        },
        'legend': {'font': {'color': 'rgba(255,255,255,0.7)'}}
    }
}

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_stat_card(icon, value, label, color_class="primary"):
    """Create a beautiful stat card component"""
    gradient_map = {
        'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'success': 'linear-gradient(135deg, #11998e 0%, #38ef7d 100%)',
        'warning': 'linear-gradient(135deg, #f7971e 0%, #ffd200 100%)',
        'danger': 'linear-gradient(135deg, #eb3349 0%, #f45c43 100%)',
        'accent': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'secondary': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    }
    
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={
                'fontSize': '2.5rem',
                'background': gradient_map.get(color_class, gradient_map['primary']),
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'marginBottom': '1rem'
            }),
            html.Div(str(value), style={
                'fontSize': '2.75rem',
                'fontWeight': '800',
                'background': gradient_map.get(color_class, gradient_map['primary']),
                'WebkitBackgroundClip': 'text',
                'WebkitTextFillColor': 'transparent',
                'lineHeight': '1'
            }),
            html.Div(label, style={
                'fontSize': '0.875rem',
                'color': 'rgba(255,255,255,0.6)',
                'textTransform': 'uppercase',
                'letterSpacing': '0.1em',
                'marginTop': '0.5rem',
                'fontWeight': '500'
            })
        ], style={'textAlign': 'center'})
    ], className='stat-card pulse', style={
        'background': 'rgba(255,255,255,0.03)',
        'backdropFilter': 'blur(20px)',
        'border': '1px solid rgba(255,255,255,0.1)',
        'borderRadius': '20px',
        'padding': '2rem',
        'height': '100%',
        'transition': 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
    })

def create_chart_container(title, chart, icon="fa-chart-bar"):
    """Wrap chart in styled container"""
    return html.Div([
        html.Div([
            html.I(className=f"fas {icon}", style={
                'color': COLORS['primary'],
                'marginRight': '10px',
                'fontSize': '1.25rem'
            }),
            html.Span(title, style={
                'fontSize': '1.25rem',
                'fontWeight': '700',
                'color': COLORS['text']
            })
        ], style={
            'display': 'flex',
            'alignItems': 'center',
            'marginBottom': '1.5rem',
            'paddingBottom': '1rem',
            'borderBottom': '1px solid rgba(255,255,255,0.1)'
        }),
        chart
    ], className='chart-container', style={
        'background': 'rgba(255,255,255,0.03)',
        'backdropFilter': 'blur(20px)',
        'border': '1px solid rgba(255,255,255,0.1)',
        'borderRadius': '24px',
        'padding': '1.5rem',
        'marginBottom': '1.5rem',
        'transition': 'all 0.4s ease'
    })

# ============================================
# CREATE VISUALIZATIONS
# ============================================

def create_keywords_chart():
    """Create beautiful keyword importance chart"""
    top_keywords = keywords_df.head(15)
    
    fig = go.Figure()
    
    # Add bars with gradient-like effect
    colors = px.colors.sample_colorscale('Viridis', 
        [i/len(top_keywords) for i in range(len(top_keywords))])
    colors = colors[::-1]  # Reverse for better visual
    
    fig.add_trace(go.Bar(
        y=top_keywords['keyword'],
        x=top_keywords['tfidf_score'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>TF-IDF Score: %{x:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="TF-IDF Score",
        yaxis_title="",
        showlegend=False,
        hoverlabel=dict(
            bgcolor="rgba(20,20,35,0.95)",
            font_size=14,
            font_family="Inter"
        )
    )
    
    return fig

def create_topic_distribution_chart():
    """Create topic distribution pie chart"""
    topic_counts = df['topic_name'].value_counts()
    
    fig = go.Figure(data=[go.Pie(
        labels=topic_counts.index,
        values=topic_counts.values,
        hole=0.5,
        marker=dict(
            colors=[TOPIC_COLORS.get(t, COLORS['primary']) for t in topic_counts.index],
            line=dict(color='rgba(255,255,255,0.2)', width=2)
        ),
        textinfo='percent+label',
        textposition='outside',
        textfont=dict(size=11, color='rgba(255,255,255,0.8)'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        template=chart_template,
        height=450,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        ),
        annotations=[dict(
            text='Topics',
            x=0.5, y=0.5,
            font_size=20,
            font_color='rgba(255,255,255,0.8)',
            showarrow=False
        )]
    )
    
    return fig

def create_topic_words_chart():
    """Create topic words heatmap"""
    # Prepare data for heatmap
    topic_data = []
    for topic in topics:
        for word, weight in zip(topic['words'][:8], topic['weights'][:8]):
            topic_data.append({
                'topic': topic['topic_name'],
                'word': word,
                'weight': weight
            })
    
    topic_words_df = pd.DataFrame(topic_data)
    pivot_df = topic_words_df.pivot(index='word', columns='topic', values='weight').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_df.values,
        x=pivot_df.columns,
        y=pivot_df.index,
        colorscale=[
            [0, 'rgba(10,10,15,0.8)'],
            [0.5, 'rgba(102,126,234,0.6)'],
            [1, 'rgba(118,75,162,1)']
        ],
        hovertemplate='Topic: %{x}<br>Word: %{y}<br>Weight: %{z:.4f}<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=500,
        margin=dict(l=100, r=20, t=20, b=100),
        xaxis=dict(tickangle=45),
        coloraxis_colorbar=dict(
            title="Weight",
            titlefont=dict(color='rgba(255,255,255,0.7)'),
            tickfont=dict(color='rgba(255,255,255,0.7)')
        )
    )
    
    return fig

def create_timeline_chart():
    """Create inspection timeline chart"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    daily_counts = df_copy.groupby(df_copy['date'].dt.to_period('W')).agg({
        'id': 'count',
        'compliance_score': 'mean',
        'issues_count': 'sum'
    }).reset_index()
    daily_counts['date'] = daily_counts['date'].astype(str)
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Inspection count bars
    fig.add_trace(
        go.Bar(
            x=daily_counts['date'],
            y=daily_counts['id'],
            name='Inspections',
            marker=dict(
                color='rgba(102,126,234,0.7)',
                line=dict(color='rgba(102,126,234,1)', width=1)
            ),
            hovertemplate='Week: %{x}<br>Inspections: %{y}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Compliance score line
    fig.add_trace(
        go.Scatter(
            x=daily_counts['date'],
            y=daily_counts['compliance_score'],
            name='Avg Compliance',
            mode='lines+markers',
            line=dict(color='#38ef7d', width=3),
            marker=dict(size=8, symbol='circle'),
            hovertemplate='Week: %{x}<br>Compliance: %{y:.1f}%<extra></extra>'
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        template=chart_template,
        height=400,
        margin=dict(l=20, r=20, t=20, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        hovermode='x unified'
    )
    
    fig.update_yaxes(title_text="Number of Inspections", secondary_y=False)
    fig.update_yaxes(title_text="Compliance Score (%)", secondary_y=True)
    
    return fig

def create_category_chart():
    """Create category distribution chart"""
    category_stats = df.groupby('category').agg({
        'id': 'count',
        'compliance_score': 'mean',
        'issues_count': 'sum'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=category_stats['category'],
        y=category_stats['id'],
        name='Inspections',
        marker=dict(
            color=COLORS['gradient_colors'][:len(category_stats)],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Inspections: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=400,
        margin=dict(l=20, r=20, t=20, b=60),
        xaxis=dict(tickangle=45),
        showlegend=False
    )
    
    return fig

def create_severity_radar():
    """Create severity radar chart by category"""
    severity_by_cat = df.groupby('category')['severity'].mean().reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=severity_by_cat['severity'],
        theta=severity_by_cat['category'],
        fill='toself',
        fillcolor='rgba(102,126,234,0.3)',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#764ba2'),
        hovertemplate='<b>%{theta}</b><br>Avg Severity: %{r:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=400,
        polar=dict(
            bgcolor='rgba(0,0,0,0)',
            radialaxis=dict(
                visible=True,
                range=[0, 5],
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='rgba(255,255,255,0.5)')
            ),
            angularaxis=dict(
                gridcolor='rgba(255,255,255,0.1)',
                tickfont=dict(color='rgba(255,255,255,0.7)')
            )
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=40, b=40)
    )
    
    return fig

def create_project_type_chart():
    """Create project type sunburst chart"""
    project_category = df.groupby(['project_type', 'category']).size().reset_index(name='count')
    
    fig = px.sunburst(
        project_category,
        path=['project_type', 'category'],
        values='count',
        color='count',
        color_continuous_scale=['#667eea', '#764ba2', '#f093fb', '#f5576c']
    )
    
    fig.update_layout(
        template=chart_template,
        height=500,
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    fig.update_traces(
        textfont=dict(color='white'),
        hovertemplate='<b>%{label}</b><br>Count: %{value}<extra></extra>'
    )
    
    return fig

def create_inspector_performance():
    """Create inspector performance chart"""
    inspector_stats = df.groupby('inspector').agg({
        'id': 'count',
        'compliance_score': 'mean',
        'issues_count': 'mean'
    }).reset_index()
    inspector_stats.columns = ['inspector', 'inspections', 'avg_compliance', 'avg_issues']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=inspector_stats['inspections'],
        y=inspector_stats['avg_compliance'],
        mode='markers+text',
        text=inspector_stats['inspector'].apply(lambda x: x.split()[0]),
        textposition='top center',
        textfont=dict(color='rgba(255,255,255,0.8)', size=10),
        marker=dict(
            size=inspector_stats['avg_issues'] * 5 + 20,
            color=inspector_stats['avg_compliance'],
            colorscale=[[0, '#f45c43'], [0.5, '#ffd200'], [1, '#38ef7d']],
            line=dict(color='rgba(255,255,255,0.3)', width=2),
            opacity=0.8
        ),
        hovertemplate='<b>%{text}</b><br>Inspections: %{x}<br>Avg Compliance: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=400,
        margin=dict(l=20, r=20, t=40, b=60),
        xaxis_title="Number of Inspections",
        yaxis_title="Average Compliance Score (%)",
        showlegend=False
    )
    
    return fig

def create_status_gauge():
    """Create compliance gauge chart"""
    avg_compliance = df['compliance_score'].mean()
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=avg_compliance,
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 85, 'increasing': {'color': '#38ef7d'}, 'decreasing': {'color': '#f45c43'}},
        number={'font': {'size': 50, 'color': '#ffffff'}, 'suffix': '%'},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "rgba(255,255,255,0.3)"},
            'bar': {'color': "#667eea"},
            'bgcolor': "rgba(255,255,255,0.05)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.1)",
            'steps': [
                {'range': [0, 60], 'color': 'rgba(244,92,67,0.3)'},
                {'range': [60, 80], 'color': 'rgba(255,210,0,0.3)'},
                {'range': [80, 100], 'color': 'rgba(56,239,125,0.3)'}
            ],
            'threshold': {
                'line': {'color': "#f093fb", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        template=chart_template,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

def create_location_map():
    """Create ultimate location performance visualization"""
    location_stats = df.groupby('location').agg({
        'id': 'count',
        'compliance_score': 'mean',
        'issues_count': ['sum', 'mean'],
        'resolution_time': 'mean',
        'severity': 'mean'
    }).reset_index()
    location_stats.columns = ['location', 'inspections', 'avg_compliance', 'total_issues', 'avg_issues', 'avg_resolution', 'avg_severity']
    
    # Normalize values for radar chart (0-100 scale)
    location_stats['norm_inspections'] = (location_stats['inspections'] / location_stats['inspections'].max() * 100).round(1)
    location_stats['norm_issues'] = (100 - (location_stats['total_issues'] / location_stats['total_issues'].max() * 100)).round(1)  # Inverse - lower is better
    location_stats['norm_resolution'] = (100 - (location_stats['avg_resolution'] / location_stats['avg_resolution'].max() * 100)).round(1)  # Inverse
    location_stats['norm_severity'] = (100 - (location_stats['avg_severity'] / 5 * 100)).round(1)  # Inverse
    
    # Sort by compliance
    location_stats = location_stats.sort_values('avg_compliance', ascending=False)
    
    # Create figure
    fig = go.Figure()
    
    # Add horizontal bar chart showing overall performance score
    location_stats['overall_score'] = (
        location_stats['avg_compliance'] * 0.4 + 
        location_stats['norm_issues'] * 0.25 + 
        location_stats['norm_resolution'] * 0.2 + 
        location_stats['norm_severity'] * 0.15
    ).round(1)
    
    # Color based on overall score
    colors = []
    for score in location_stats['overall_score']:
        if score >= 80:
            colors.append('#38ef7d')
        elif score >= 65:
            colors.append('#4facfe')
        elif score >= 50:
            colors.append('#ffd200')
        else:
            colors.append('#f45c43')
    
    fig.add_trace(go.Bar(
        y=location_stats['location'],
        x=location_stats['overall_score'],
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='rgba(255,255,255,0.3)', width=1)
        ),
        customdata=np.column_stack([
            location_stats['inspections'],
            location_stats['avg_compliance'].round(1),
            location_stats['total_issues'],
            location_stats['avg_resolution'].round(1),
            location_stats['avg_severity'].round(2)
        ]),
        text=location_stats['overall_score'].apply(lambda x: f'{x:.0f}'),
        textposition='inside',
        textfont=dict(color='white', size=12, family='Inter'),
        hovertemplate='<b>%{y}</b><br><br>' +
                      '🏆 Overall Score: %{x:.1f}/100<br>' +
                      '━━━━━━━━━━━━━━━━━<br>' +
                      '📋 Total Inspections: %{customdata[0]}<br>' +
                      '✅ Avg Compliance: %{customdata[1]}%<br>' +
                      '⚠️ Total Issues: %{customdata[2]}<br>' +
                      '⏱️ Avg Resolution: %{customdata[3]} days<br>' +
                      '🔴 Avg Severity: %{customdata[4]}/5<extra></extra>'
    ))
    
    # Add compliance markers
    fig.add_trace(go.Scatter(
        y=location_stats['location'],
        x=location_stats['avg_compliance'],
        mode='markers',
        name='Compliance %',
        marker=dict(
            size=12,
            color='#667eea',
            symbol='diamond',
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>%{y}</b><br>Compliance: %{x:.1f}%<extra></extra>'
    ))
    
    # Add vertical lines for reference
    fig.add_vline(x=80, line_dash="dash", line_color="rgba(56,239,125,0.5)", 
                  annotation_text="Target: 80", annotation_position="top")
    fig.add_vline(x=50, line_dash="dash", line_color="rgba(244,92,67,0.5)",
                  annotation_text="Warning: 50", annotation_position="top")
    
    fig.update_layout(
        template=chart_template,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            title='Performance Score',
            range=[0, 105],
            ticksuffix='',
        ),
        yaxis=dict(title=''),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        annotations=[
            dict(
                text="🏆 Overall Performance Score (Weighted: Compliance 40%, Issues 25%, Resolution 20%, Severity 15%)",
                xref="paper", yref="paper",
                x=0, y=-0.12,
                showarrow=False,
                font=dict(size=10, color='rgba(255,255,255,0.5)')
            )
        ]
    )
    
    return fig

def create_topic_network():
    """Create topic relationship network"""
    # Create node positions in a circle
    n_topics = len(topics)
    angles = np.linspace(0, 2*np.pi, n_topics, endpoint=False)
    
    nodes_x = np.cos(angles) * 2
    nodes_y = np.sin(angles) * 2
    
    # Create edges based on shared words
    edge_traces = []
    for i, topic1 in enumerate(topics):
        for j, topic2 in enumerate(topics):
            if i < j:
                shared = len(set(topic1['words']) & set(topic2['words']))
                if shared > 0:
                    edge_traces.append(go.Scatter(
                        x=[nodes_x[i], nodes_x[j]],
                        y=[nodes_y[i], nodes_y[j]],
                        mode='lines',
                        line=dict(
                            width=shared * 0.5,
                            color='rgba(102,126,234,0.3)'
                        ),
                        hoverinfo='none'
                    ))
    
    # Create node trace
    node_trace = go.Scatter(
        x=nodes_x,
        y=nodes_y,
        mode='markers+text',
        text=[t['topic_name'] for t in topics],
        textposition='top center',
        textfont=dict(size=10, color='rgba(255,255,255,0.9)'),
        marker=dict(
            size=40,
            color=[TOPIC_COLORS.get(t['topic_name'], COLORS['primary']) for t in topics],
            line=dict(color='rgba(255,255,255,0.3)', width=2)
        ),
        hovertemplate='<b>%{text}</b><extra></extra>'
    )
    
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        template=chart_template,
        height=450,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    
    return fig

def create_3d_scatter():
    """Create 3D scatter plot"""
    fig = go.Figure(data=[go.Scatter3d(
        x=df['compliance_score'],
        y=df['issues_count'],
        z=df['resolution_time'],
        mode='markers',
        marker=dict(
            size=5,
            color=df['severity'],
            colorscale=[[0, '#38ef7d'], [0.5, '#ffd200'], [1, '#f45c43']],
            opacity=0.8,
            line=dict(color='rgba(255,255,255,0.2)', width=0.5)
        ),
        text=df['project_type'],
        hovertemplate='<b>%{text}</b><br>Compliance: %{x}%<br>Issues: %{y}<br>Resolution: %{z} days<extra></extra>'
    )])
    
    fig.update_layout(
        template=chart_template,
        height=500,
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(
            xaxis=dict(
                title='Compliance Score',
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True
            ),
            yaxis=dict(
                title='Issues Count',
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True
            ),
            zaxis=dict(
                title='Resolution Time',
                backgroundcolor='rgba(0,0,0,0)',
                gridcolor='rgba(255,255,255,0.1)',
                showbackground=True
            ),
            bgcolor='rgba(0,0,0,0)'
        )
    )
    
    return fig

def create_status_distribution():
    """Create status distribution bar chart"""
    status_counts = df['status'].value_counts().reset_index()
    status_counts.columns = ['status', 'count']
    
    status_colors = {
        'Approved': '#38ef7d',
        'Pending': '#ffd200',
        'Requires Follow-up': '#4facfe',
        'Critical': '#f45c43'
    }
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=status_counts['status'],
        y=status_counts['count'],
        marker=dict(
            color=[status_colors.get(s, '#667eea') for s in status_counts['status']],
            line=dict(color='rgba(255,255,255,0.2)', width=1)
        ),
        hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=350,
        margin=dict(l=20, r=20, t=20, b=60),
        showlegend=False
    )
    
    return fig

def create_monthly_trend():
    """Create monthly trend line chart"""
    df_copy = df.copy()
    df_copy['date'] = pd.to_datetime(df_copy['date'])
    df_copy['month'] = df_copy['date'].dt.to_period('M')
    
    monthly_data = df_copy.groupby('month').agg({
        'id': 'count',
        'compliance_score': 'mean',
        'issues_count': 'sum'
    }).reset_index()
    monthly_data['month'] = monthly_data['month'].astype(str)
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=monthly_data['month'],
        y=monthly_data['id'],
        mode='lines+markers',
        name='Inspections',
        line=dict(color='#667eea', width=3),
        marker=dict(size=10, color='#667eea'),
        fill='tozeroy',
        fillcolor='rgba(102,126,234,0.2)'
    ))
    
    fig.update_layout(
        template=chart_template,
        height=350,
        margin=dict(l=20, r=20, t=20, b=60),
        xaxis_title="Month",
        yaxis_title="Number of Inspections",
        hovermode='x unified'
    )
    
    return fig

# ============================================
# LAYOUT
# ============================================

app.layout = html.Div([
    # Background Animation
    html.Div(className='background-animation'),
    
    # Header
html.Div([
    dbc.Container([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.I(className="fas fa-hard-hat", style={
                        'fontSize': '3rem',
                        'marginRight': '1rem',
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent'
                    }),
                    html.Span("Construction Inspection Analytics", style={
                        'fontSize': '2.25rem',
                        'fontWeight': '800',
                        'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                        'WebkitBackgroundClip': 'text',
                        'WebkitTextFillColor': 'transparent',
                        'letterSpacing': '-0.02em'
                    })
                ], style={'display': 'flex', 'alignItems': 'center'}),  # <-- THIS WAS MISSING
                html.P([
                    html.I(className="fas fa-brain", style={'marginRight': '8px'}),
                    "Advanced Keyword Extraction & Topic Modeling Dashboard | ",
                    html.Span("Business Intelligence Project", style={'color': '#f093fb'}),
                    " | ",
                    html.Span("Group 14", style={'color': '#38ef7d', 'fontWeight': '600'})
                ], style={
                    'color': 'rgba(255,255,255,0.6)',
                    'marginTop': '0.5rem',
                    'fontSize': '0.95rem'
                })
            ], md=8),
            dbc.Col([
                html.Div([
                    html.Div([
                        html.I(className="fas fa-circle pulse", style={
                            'color': '#38ef7d',
                            'fontSize': '0.5rem',
                            'marginRight': '8px'
                        }),
                        html.Span("Live Dashboard", style={
                            'color': 'rgba(255,255,255,0.7)',
                            'fontSize': '0.875rem'
                        })
                    ], style={'marginBottom': '0.5rem'}),
                    html.Div([
                        html.I(className="fas fa-clock", style={
                            'color': 'rgba(255,255,255,0.5)',
                            'marginRight': '8px'
                        }),
                        html.Span(
                            datetime.now().strftime("%B %d, %Y | %H:%M"),
                            style={'color': 'rgba(255,255,255,0.7)', 'fontSize': '0.875rem'}
                        )
                    ])
                ], style={'textAlign': 'right'})
            ], md=4)
        ])
    ], fluid=True)
], style={
    'background': 'linear-gradient(135deg, rgba(102,126,234,0.15) 0%, rgba(118,75,162,0.15) 100%)',
    'backdropFilter': 'blur(20px)',
    'borderBottom': '1px solid rgba(255,255,255,0.1)',
    'padding': '1.5rem 2rem',
    'marginBottom': '2rem'
}),
    
    # Main Content
    dbc.Container([
        
        # Stats Row
        dbc.Row([
            dbc.Col([
                create_stat_card("fa-file-alt", stats['total_reports'], "Total Inspections", "primary")
            ], lg=2, md=4, sm=6, className='mb-4'),
            dbc.Col([
                create_stat_card("fa-chart-line", f"{stats['avg_compliance']:.1f}%", "Avg Compliance", "success")
            ], lg=2, md=4, sm=6, className='mb-4'),
            dbc.Col([
                create_stat_card("fa-exclamation-triangle", stats['total_issues'], "Total Issues", "warning")
            ], lg=2, md=4, sm=6, className='mb-4'),
            dbc.Col([
                create_stat_card("fa-clock", f"{stats['avg_resolution_time']:.1f}d", "Avg Resolution", "accent")
            ], lg=2, md=4, sm=6, className='mb-4'),
            dbc.Col([
                create_stat_card("fa-times-circle", stats['critical_count'], "Critical Issues", "danger")
            ], lg=2, md=4, sm=6, className='mb-4'),
            dbc.Col([
                create_stat_card("fa-check-circle", stats['approved_count'], "Approved", "success")
            ], lg=2, md=4, sm=6, className='mb-4'),
        ], className='mb-4'),
        
        # Keyword Extraction & Topic Modeling Section
        html.Div([
            html.H2([
                html.I(className="fas fa-key", style={'marginRight': '15px', 'color': COLORS['primary']}),
                "Keyword Extraction & Topic Modeling"
            ], style={
                'color': COLORS['text'],
                'fontWeight': '700',
                'marginBottom': '1.5rem',
                'paddingBottom': '1rem',
                'borderBottom': '2px solid rgba(102,126,234,0.3)'
            })
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Top Keywords by TF-IDF Score",
                    dcc.Graph(figure=create_keywords_chart(), config={'displayModeBar': False}),
                    "fa-tags"
                )
            ], lg=6, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Topic Distribution",
                    dcc.Graph(figure=create_topic_distribution_chart(), config={'displayModeBar': False}),
                    "fa-pie-chart"
                )
            ], lg=6, className='mb-4'),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Topic-Keyword Relationship Heatmap",
                    dcc.Graph(figure=create_topic_words_chart(), config={'displayModeBar': False}),
                    "fa-th"
                )
            ], lg=8, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Topic Network",
                    dcc.Graph(figure=create_topic_network(), config={'displayModeBar': False}),
                    "fa-project-diagram"
                )
            ], lg=4, className='mb-4'),
        ]),
        
        # Inspection Analytics Section
        html.Div([
            html.H2([
                html.I(className="fas fa-chart-bar", style={'marginRight': '15px', 'color': COLORS['accent']}),
                "Inspection Analytics"
            ], style={
                'color': COLORS['text'],
                'fontWeight': '700',
                'marginBottom': '1.5rem',
                'marginTop': '2rem',
                'paddingBottom': '1rem',
                'borderBottom': '2px solid rgba(79,172,254,0.3)'
            })
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Inspection Timeline & Compliance Trend",
                    dcc.Graph(figure=create_timeline_chart(), config={'displayModeBar': False}),
                    "fa-calendar-alt"
                )
            ], lg=8, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Overall Compliance Score",
                    dcc.Graph(figure=create_status_gauge(), config={'displayModeBar': False}),
                    "fa-tachometer-alt"
                )
            ], lg=4, className='mb-4'),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Category Distribution",
                    dcc.Graph(figure=create_category_chart(), config={'displayModeBar': False}),
                    "fa-layer-group"
                )
            ], lg=4, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Severity by Category",
                    dcc.Graph(figure=create_severity_radar(), config={'displayModeBar': False}),
                    "fa-crosshairs"
                )
            ], lg=4, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Inspector Performance",
                    dcc.Graph(figure=create_inspector_performance(), config={'displayModeBar': False}),
                    "fa-user-tie"
                )
            ], lg=4, className='mb-4'),
        ]),
        
        # Advanced Visualizations
        html.Div([
            html.H2([
                html.I(className="fas fa-cube", style={'marginRight': '15px', 'color': '#f093fb'}),
                "Advanced Analytics"
            ], style={
                'color': COLORS['text'],
                'fontWeight': '700',
                'marginBottom': '1.5rem',
                'marginTop': '2rem',
                'paddingBottom': '1rem',
                'borderBottom': '2px solid rgba(240,147,251,0.3)'
            })
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Project Type Hierarchy",
                    dcc.Graph(figure=create_project_type_chart(), config={'displayModeBar': False}),
                    "fa-building"
                )
            ], lg=6, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Location Performance",
                    dcc.Graph(figure=create_location_map(), config={'displayModeBar': False}),
                    "fa-map-marked-alt"
                )
            ], lg=6, className='mb-4'),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "3D Analysis: Compliance vs Issues vs Resolution Time",
                    dcc.Graph(figure=create_3d_scatter(), config={'displayModeBar': True}),
                    "fa-cubes"
                )
            ], lg=8, className='mb-4'),
            dbc.Col([
                create_chart_container(
                    "Status Distribution",
                    dcc.Graph(figure=create_status_distribution(), config={'displayModeBar': False}),
                    "fa-tasks"
                )
            ], lg=4, className='mb-4'),
        ]),
        
        dbc.Row([
            dbc.Col([
                create_chart_container(
                    "Monthly Inspection Trend",
                    dcc.Graph(figure=create_monthly_trend(), config={'displayModeBar': False}),
                    "fa-chart-area"
                )
            ], className='mb-4'),
        ]),
        
        # Data Table Section
        html.Div([
            html.H2([
                html.I(className="fas fa-table", style={'marginRight': '15px', 'color': COLORS['success']}),
                "Inspection Records"
            ], style={
                'color': COLORS['text'],
                'fontWeight': '700',
                'marginBottom': '1.5rem',
                'marginTop': '2rem',
                'paddingBottom': '1rem',
                'borderBottom': '2px solid rgba(56,239,125,0.3)'
            })
        ]),
        
        html.Div([
            # Filters
            dbc.Row([
                dbc.Col([
                    html.Label("Category", style={'color': 'rgba(255,255,255,0.7)', 'marginBottom': '0.5rem', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='category-filter',
                        options=[{'label': c, 'value': c} for c in df['category'].unique()],
                        multi=True,
                        placeholder="All Categories",
                        style={'background': 'transparent'}
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Status", style={'color': 'rgba(255,255,255,0.7)', 'marginBottom': '0.5rem', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='status-filter',
                        options=[{'label': s, 'value': s} for s in df['status'].unique()],
                        multi=True,
                        placeholder="All Statuses"
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Project Type", style={'color': 'rgba(255,255,255,0.7)', 'marginBottom': '0.5rem', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='project-filter',
                        options=[{'label': p, 'value': p} for p in df['project_type'].unique()],
                        multi=True,
                        placeholder="All Projects"
                    )
                ], md=3),
                dbc.Col([
                    html.Label("Topic", style={'color': 'rgba(255,255,255,0.7)', 'marginBottom': '0.5rem', 'fontSize': '0.875rem'}),
                    dcc.Dropdown(
                        id='topic-filter',
                        options=[{'label': t, 'value': t} for t in df['topic_name'].unique()],
                        multi=True,
                        placeholder="All Topics"
                    )
                ], md=3),
            ], style={
                'background': 'rgba(255,255,255,0.03)',
                'padding': '1.5rem',
                'borderRadius': '16px',
                'marginBottom': '1.5rem',
                'border': '1px solid rgba(255,255,255,0.1)'
            }),
            
            # Data Table
            html.Div([
                dash_table.DataTable(
                    id='inspection-table',
                    columns=[
                        {'name': 'ID', 'id': 'id'},
                        {'name': 'Date', 'id': 'date'},
                        {'name': 'Project Type', 'id': 'project_type'},
                        {'name': 'Category', 'id': 'category'},
                        {'name': 'Topic', 'id': 'topic_name'},
                        {'name': 'Status', 'id': 'status'},
                        {'name': 'Compliance', 'id': 'compliance_score'},
                        {'name': 'Issues', 'id': 'issues_count'}
                    ],
                    data=df.head(50).to_dict('records'),
                    page_size=10,
                    style_table={'overflowX': 'auto'},
                    style_cell={
                        'backgroundColor': 'rgba(20,20,35,0.8)',
                        'color': 'rgba(255,255,255,0.8)',
                        'border': '1px solid rgba(255,255,255,0.1)',
                        'textAlign': 'left',
                        'padding': '12px 15px',
                        'fontFamily': 'Inter, sans-serif'
                    },
                    style_header={
                        'backgroundColor': 'rgba(102,126,234,0.3)',
                        'color': 'white',
                        'fontWeight': '600',
                        'border': '1px solid rgba(255,255,255,0.1)'
                    },
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgba(255,255,255,0.02)'
                        },
                        {
                            'if': {
                                'filter_query': '{status} = "Critical"',
                                'column_id': 'status'
                            },
                            'backgroundColor': 'rgba(235,51,73,0.3)',
                            'color': '#f45c43'
                        },
                        {
                            'if': {
                                'filter_query': '{status} = "Approved"',
                                'column_id': 'status'
                            },
                            'backgroundColor': 'rgba(56,239,125,0.2)',
                            'color': '#38ef7d'
                        },
                        {
                            'if': {
                                'filter_query': '{compliance_score} < 70',
                                'column_id': 'compliance_score'
                            },
                            'color': '#f45c43'
                        },
                        {
                            'if': {
                                'filter_query': '{compliance_score} >= 90',
                                'column_id': 'compliance_score'
                            },
                            'color': '#38ef7d'
                        }
                    ],
                    filter_action='native',
                    sort_action='native',
                    page_action='native'
                )
            ], style={
                'background': 'rgba(255,255,255,0.03)',
                'padding': '1.5rem',
                'borderRadius': '24px',
                'border': '1px solid rgba(255,255,255,0.1)'
            })
        ], className='mb-4'),
        
        # Footer
        html.Div([
            html.Div([
                html.Div([
                    html.I(className="fas fa-hard-hat", style={'marginRight': '10px', 'color': COLORS['primary']}),
                    html.Span("Construction Inspection Analytics Dashboard", style={'fontWeight': '600'})
                ], style={'marginBottom': '0.5rem'}),
                html.Div([
                    html.Span("Powered by ", style={'color': 'rgba(255,255,255,0.5)'}),
                    html.Span("Dash & Plotly", style={'color': COLORS['primary'], 'fontWeight': '500'}),
                    html.Span(" | Business Intelligence Project | ", style={'color': 'rgba(255,255,255,0.5)'}),
                    html.Span(datetime.now().strftime("%Y"), style={'color': 'rgba(255,255,255,0.5)'})
                ], style={'fontSize': '0.875rem'})
            ], style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.7)'})
        ], style={
            'background': 'rgba(255,255,255,0.03)',
            'backdropFilter': 'blur(20px)',
            'borderTop': '1px solid rgba(255,255,255,0.1)',
            'padding': '2rem',
            'marginTop': '3rem',
            'borderRadius': '24px 24px 0 0'
        })
        
    ], fluid=True, style={'maxWidth': '1800px'})
    
], style={
    'minHeight': '100vh',
    'background': 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0a0a0f 100%)'
})

# ============================================
# CALLBACKS
# ============================================

@app.callback(
    Output('inspection-table', 'data'),
    [
        Input('category-filter', 'value'),
        Input('status-filter', 'value'),
        Input('project-filter', 'value'),
        Input('topic-filter', 'value')
    ]
)
def update_table(categories, statuses, projects, topics_selected):
    """Filter table based on selections"""
    filtered_df = df.copy()
    
    if categories:
        filtered_df = filtered_df[filtered_df['category'].isin(categories)]
    if statuses:
        filtered_df = filtered_df[filtered_df['status'].isin(statuses)]
    if projects:
        filtered_df = filtered_df[filtered_df['project_type'].isin(projects)]
    if topics_selected:
        filtered_df = filtered_df[filtered_df['topic_name'].isin(topics_selected)]
    
    return filtered_df.head(100).to_dict('records')

# ============================================
# RUN APPLICATION
# ============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🏗️  CONSTRUCTION INSPECTION ANALYTICS DASHBOARD")
    print("="*60)
    print("\n📊 Dashboard is starting...")
    print("🌐 Open your browser and navigate to: http://127.0.0.1:8050")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    app.run_server(debug=True, port=8050)