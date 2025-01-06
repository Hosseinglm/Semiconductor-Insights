import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict

class Visualizer:
    def __init__(self):
        self.color_scheme = px.colors.qualitative.Set3

    def create_time_series(self, data: pd.DataFrame, metric: str) -> go.Figure:
        """Create interactive time series plot with enhanced visualization"""

        # Calculate daily averages for all metrics
        df_daily = data.groupby(['date'])[metric].agg(['mean', 'std']).reset_index()
        df_daily.columns = ['date', 'mean', 'std']

        # Create main figure with average line
        fig = go.Figure()

        # Add average line with confidence interval
        fig.add_trace(go.Scatter(
            x=df_daily['date'],
            y=df_daily['mean'],
            name='Overall Average',
            line=dict(color='rgba(31, 119, 180, 1)', width=2),
            mode='lines'
        ))

        # Add confidence interval
        fig.add_trace(go.Scatter(
            x=df_daily['date'].tolist() + df_daily['date'].tolist()[::-1],
            y=(df_daily['mean'] + df_daily['std']).tolist() +
              (df_daily['mean'] - df_daily['std']).tolist()[::-1],
            fill='toself',
            fillcolor='rgba(31, 119, 180, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1 Standard Deviation',
            showlegend=True
        ))

        # Add individual machine lines with lower opacity
        unique_machines = data['machine_id'].unique()
        for machine in unique_machines[:5]:  # Show only top 5 machines by default
            machine_data = data[data['machine_id'] == machine]
            fig.add_trace(go.Scatter(
                x=machine_data['date'],
                y=machine_data[metric],
                name=f'{machine}',
                mode='lines',
                line=dict(width=1, dash='dot'),
                opacity=0.5,
                visible='legendonly'  # Hidden by default
            ))

        # Update layout
        title_text = f"{metric.replace('_', ' ').title()} Over Time"
        if metric == 'machine_downtime':
            title_text = 'Machine Downtime Over Time (hours)'
            y_axis_title = 'Downtime (hours)'
        else:
            y_axis_title = metric.replace('_', ' ').title()

        fig.update_layout(
            title=title_text,
            xaxis_title="Date",
            yaxis_title=y_axis_title,
            hovermode='x unified',
            plot_bgcolor='white',
            xaxis=dict(
                gridcolor='lightgray',
                rangeslider=dict(visible=True)
            ),
            yaxis=dict(gridcolor='lightgray'),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=1.05
            )
        )

        # Update hover template
        hover_template = (
            "Date: %{x}<br>"
            f"{y_axis_title}: %{{y:.2f}}"
            "<extra></extra>"
        )
        fig.update_traces(hovertemplate=hover_template)

        return fig

    def create_correlation_heatmap(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive correlation heatmap"""
        corr_matrix = data[[
            'production_yield', 'defect_rate', 'machine_downtime',
            'temperature', 'humidity'
        ]].corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1,
            zmax=1,
            hoverongaps=False,
            hovertemplate="Variable 1: %{x}<br>Variable 2: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))

        fig.update_layout(
            title='Feature Correlation Analysis',
            xaxis_title="Features",
            yaxis_title="Features",
            xaxis=dict(tickangle=45),
            plot_bgcolor='white'
        )
        return fig

    def create_machine_comparison(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive machine performance comparison"""
        avg_performance = data.groupby('machine_id').agg({
            'production_yield': 'mean',
            'defect_rate': 'mean',
            'downtime_ratio': 'mean',
            'temperature': 'mean',
            'humidity': 'mean'
        }).reset_index()

        fig = go.Figure()
        metrics = ['production_yield', 'defect_rate', 'downtime_ratio']

        for i, metric in enumerate(metrics):
            hover_text = (
                f"Machine: %{{x}}<br>"
                f"{metric.replace('_', ' ').title()}: %{{y:.2f}}<br>"
                f"Avg Temperature: %{{customdata[0]:.1f}}°C<br>"
                f"Avg Humidity: %{{customdata[1]:.1f}}%"
                f"<extra></extra>"
            )

            fig.add_trace(go.Bar(
                name=metric.replace("_", " ").title(),
                x=avg_performance['machine_id'],
                y=avg_performance[metric],
                marker_color=self.color_scheme[i],
                hovertemplate=hover_text,
                customdata=avg_performance[['temperature', 'humidity']].values
            ))

        fig.update_layout(
            barmode='group',
            title='Machine Performance Analysis',
            xaxis_title="Machine ID",
            yaxis_title="Value",
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            showlegend=True,
            legend_title="Metrics",
            hovermode='closest'
        )
        return fig

    def create_material_analysis(self, data: pd.DataFrame) -> go.Figure:
        """Create interactive material type analysis"""
        material_stats = data.groupby('material_type').agg({
            'production_yield': ['mean', 'std'],
            'defect_rate': ['mean', 'std'],
            'machine_downtime': 'mean'
        }).reset_index()

        material_stats.columns = ['material_type', 'yield_mean', 'yield_std',
                                'defect_mean', 'defect_std', 'downtime_mean']

        fig = px.scatter(
            material_stats,
            x='yield_mean',
            y='defect_mean',
            color='material_type',
            size='yield_mean',
            error_x='yield_std',
            error_y='defect_std',
            title='Material Performance Analysis',
            labels={
                'yield_mean': 'Average Production Yield (%)',
                'defect_mean': 'Average Defect Rate (%)',
                'material_type': 'Material Type'
            }
        )

        hover_text = (
            f"Material: %{{customdata[0]}}<br>"
            f"Yield: %{{x:.2f}}% ± %{{customdata[1]:.2f}}%<br>"
            f"Defect Rate: %{{y:.2f}}% ± %{{customdata[2]:.2f}}%<br>"
            f"Avg Downtime: %{{customdata[3]:.1f}} hours"
            f"<extra></extra>"
        )

        fig.update_traces(
            hovertemplate=hover_text,
            customdata=material_stats[['material_type', 'yield_std',
                                     'defect_std', 'downtime_mean']].values
        )

        fig.update_layout(
            xaxis_title="Average Production Yield (%)",
            yaxis_title="Average Defect Rate (%)",
            plot_bgcolor='white',
            xaxis=dict(gridcolor='lightgray'),
            yaxis=dict(gridcolor='lightgray'),
            showlegend=True,
            legend_title="Material Type"
        )
        return fig