"""
グラフ生成関数
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np


def create_grouped_bar_chart(data, x, y, color=None, title="", xaxis_title="", yaxis_title=""):
    """グループ棒グラフを作成"""
    fig = px.bar(
        data, x=x, y=y, color=color,
        barmode='group',
        title=title,
        labels={x: xaxis_title, y: yaxis_title}
    )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        legend_title_text=color if color else "",
        template="plotly_white"
    )
    return fig


def create_histogram(data, x, title="", xaxis_title="", nbins=10, color=None):
    """ヒストグラムを作成"""
    fig = px.histogram(
        data, x=x,
        nbins=nbins,
        title=title,
        color=color,
        labels={x: xaxis_title}
    )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title="度数",
        template="plotly_white"
    )
    return fig


def create_scatter_with_trendline(data, x, y, title="", xaxis_title="", yaxis_title="", color=None):
    """回帰線付き散布図を作成"""
    fig = px.scatter(
        data, x=x, y=y,
        trendline="ols",
        title=title,
        color=color,
        labels={x: xaxis_title, y: yaxis_title}
    )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white"
    )
    return fig


def create_heatmap(data, title="", xaxis_title="", yaxis_title="", text_auto=True):
    """ヒートマップを作成"""
    fig = px.imshow(
        data,
        title=title,
        text_auto=text_auto,
        color_continuous_scale="Blues",
        aspect="auto"
    )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white"
    )
    return fig


def create_confusion_matrix(matrix, labels, title="混同行列"):
    """混同行列ヒートマップを作成"""
    fig = px.imshow(
        matrix,
        labels=dict(x="予測", y="実際", color="件数"),
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title=title
    )
    fig.update_layout(template="plotly_white")
    return fig


def create_pie_chart(values, names, title="", colors=None):
    """円グラフを作成"""
    fig = px.pie(
        values=values,
        names=names,
        title=title,
        color_discrete_sequence=colors
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(template="plotly_white")
    return fig


def create_box_plot(data, x, y, title="", xaxis_title="", yaxis_title="", color=None):
    """箱ひげ図を作成"""
    fig = px.box(
        data, x=x, y=y,
        title=title,
        color=color,
        labels={x: xaxis_title, y: yaxis_title}
    )
    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        template="plotly_white"
    )
    return fig


def create_metrics_comparison_chart(metrics_df, metric_name, title=""):
    """評価指標比較棒グラフを作成"""
    fig = px.bar(
        metrics_df,
        x='モデル',
        y=metric_name,
        color='モデル',
        title=title,
        text=metric_name
    )
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        showlegend=False,
        template="plotly_white",
        yaxis_title=metric_name
    )
    return fig


def create_dual_histogram(data1, data2, x1, x2, label1, label2, title="", nbins=10):
    """2つのヒストグラムを並べて表示"""
    fig = make_subplots(rows=1, cols=2, subplot_titles=(label1, label2))
    
    fig.add_trace(
        go.Histogram(x=data1[x1], nbinsx=nbins, name=label1, marker_color='#636EFA'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=data2[x2], nbinsx=nbins, name=label2, marker_color='#EF553B'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text=title,
        showlegend=False,
        template="plotly_white"
    )
    return fig


def create_residual_plots(y_true, y_pred, title_prefix=""):
    """残差診断プロットを作成（4パネル）"""
    residuals = y_true - y_pred
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            f'{title_prefix}残差ヒストグラム',
            f'{title_prefix}残差 vs 予測値',
            f'{title_prefix}Q-Qプロット',
            f'{title_prefix}残差の箱ひげ図'
        ]
    )
    
    # 残差ヒストグラム
    fig.add_trace(
        go.Histogram(x=residuals, name='残差', marker_color='#636EFA'),
        row=1, col=1
    )
    
    # 残差 vs 予測値
    fig.add_trace(
        go.Scatter(x=y_pred, y=residuals, mode='markers', name='残差', marker=dict(color='#636EFA')),
        row=1, col=2
    )
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
    
    # Q-Qプロット（簡易版）
    from scipy import stats
    sorted_residuals = np.sort(residuals)
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(sorted_residuals)))
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sorted_residuals, mode='markers', name='Q-Q', marker=dict(color='#636EFA')),
        row=2, col=1
    )
    # 対角線を追加
    min_val = min(theoretical_quantiles.min(), sorted_residuals.min())
    max_val = max(theoretical_quantiles.max(), sorted_residuals.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', line=dict(color='red', dash='dash'), showlegend=False),
        row=2, col=1
    )
    
    # 残差の箱ひげ図
    fig.add_trace(
        go.Box(y=residuals, name='残差', marker_color='#636EFA'),
        row=2, col=2
    )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        template="plotly_white"
    )
    
    return fig


def create_gauge_chart(value, title, max_value=100, suffix="%"):
    """ゲージチャートを作成"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        number={'suffix': suffix},
        gauge={
            'axis': {'range': [0, max_value]},
            'bar': {'color': "#636EFA"},
            'steps': [
                {'range': [0, max_value*0.33], 'color': "#ffcccc"},
                {'range': [max_value*0.33, max_value*0.66], 'color': "#ffffcc"},
                {'range': [max_value*0.66, max_value], 'color': "#ccffcc"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=250, template="plotly_white")
    return fig
