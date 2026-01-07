"""
📈 分布の比較ページ
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_analysis_data

st.set_page_config(page_title="分布の比較", page_icon="📈", layout="wide")

st.title("📈 分布の比較")
st.markdown("人間の自己評価とAI予測の分布を比較します。")
st.markdown("---")

df = load_analysis_data()

if df is not None:
    # 指標選択
    metric = st.selectbox(
        "表示する指標",
        ['GAP（自信度-挑戦度）', '自信度', '挑戦度'],
        index=0
    )
    
    metric_map = {
        'GAP（自信度-挑戦度）': ('human_gap', 'ai_gap', 'GAP'),
        '自信度': ('confidence', 'ai_predicted_confidence', '自信度'),
        '挑戦度': ('challenge', 'ai_predicted_challenge', '挑戦度')
    }
    
    human_col, ai_col, label = metric_map[metric]
    
    st.markdown("---")
    
    # 並列ヒストグラム
    st.header(f"📊 {label}の分布比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("人間の評価")
        fig1 = px.histogram(
            df, x=human_col,
            nbins=13 if 'gap' in human_col.lower() else 7,
            title=f"人間の{label}分布",
            labels={human_col: label},
            color_discrete_sequence=['#636EFA']
        )
        fig1.update_layout(
            xaxis_title=label,
            yaxis_title="度数",
            bargap=0.1
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # 統計量
        st.metric(f"平均", f"{df[human_col].mean():.2f}")
        st.metric(f"標準偏差", f"{df[human_col].std():.2f}")
        st.metric(f"分散", f"{df[human_col].var():.2f}")
    
    with col2:
        st.subheader("AIの予測")
        fig2 = px.histogram(
            df, x=ai_col,
            nbins=13 if 'gap' in ai_col.lower() else 7,
            title=f"AIの{label}予測分布",
            labels={ai_col: label},
            color_discrete_sequence=['#EF553B']
        )
        fig2.update_layout(
            xaxis_title=label,
            yaxis_title="度数",
            bargap=0.1
        )
        st.plotly_chart(fig2, use_container_width=True)
        
        # 統計量
        st.metric(f"平均", f"{df[ai_col].mean():.2f}")
        st.metric(f"標準偏差", f"{df[ai_col].std():.2f}")
        st.metric(f"分散", f"{df[ai_col].var():.2f}")
    
    st.markdown("---")
    
    # 分散の比較（メトリクス表示）
    st.header("📐 分散の比較")
    
    var_human = df[human_col].var()
    var_ai = df[ai_col].var()
    var_diff = var_ai - var_human
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(f"人間の{label}分散", f"{var_human:.2f}")
    
    with col2:
        st.metric(f"AIの{label}分散", f"{var_ai:.2f}")
    
    with col3:
        delta_color = "inverse" if var_diff > 0 else "normal"
        st.metric(
            "差（AI - 人間）", 
            f"{var_diff:.2f}",
            delta=f"{'増加' if var_diff > 0 else '減少'}",
            delta_color=delta_color
        )
    
    st.markdown("---")
    
    # オーバーレイヒストグラム
    st.header("📊 重ね合わせ分布")
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df[human_col],
        name='人間',
        opacity=0.6,
        marker_color='#636EFA',
        nbinsx=13 if 'gap' in human_col.lower() else 7
    ))
    
    fig.add_trace(go.Histogram(
        x=df[ai_col],
        name='AI',
        opacity=0.6,
        marker_color='#EF553B',
        nbinsx=13 if 'gap' in ai_col.lower() else 7
    ))
    
    fig.update_layout(
        title=f'{label}の分布比較（人間 vs AI）',
        xaxis_title=label,
        yaxis_title='度数',
        barmode='overlay',
        template='plotly_white',
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 箱ひげ図比較
    st.header("📦 箱ひげ図比較")
    
    # データを整形
    box_df = pd.DataFrame({
        '値': list(df[human_col]) + list(df[ai_col]),
        '種類': ['人間'] * len(df) + ['AI'] * len(df)
    })
    
    fig = px.box(
        box_df, 
        x='種類', 
        y='値',
        color='種類',
        title=f'{label}の分布比較',
        labels={'値': label}
    )
    fig.update_layout(
        showlegend=False,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 各指標の分散比較まとめ
    st.markdown("---")
    st.header("📋 全指標の分散比較まとめ")
    
    variance_summary = pd.DataFrame({
        '指標': ['自信度', '挑戦度', 'GAP'],
        '人間の分散': [
            df['confidence'].var(),
            df['challenge'].var(),
            df['human_gap'].var()
        ],
        'AIの分散': [
            df['ai_predicted_confidence'].var(),
            df['ai_predicted_challenge'].var(),
            df['ai_gap'].var()
        ]
    })
    variance_summary['差（AI-人間）'] = variance_summary['AIの分散'] - variance_summary['人間の分散']
    
    st.dataframe(
        variance_summary.style.format({
            '人間の分散': '{:.3f}',
            'AIの分散': '{:.3f}',
            '差（AI-人間）': '{:.3f}'
        }).background_gradient(subset=['差（AI-人間）'], cmap='RdYlGn_r'),
        use_container_width=True
    )

else:
    st.error("データの読み込みに失敗しました。")
