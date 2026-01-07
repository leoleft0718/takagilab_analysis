"""
🎯 推薦システム評価ページ
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
from utils import load_analysis_data, create_confusion_matrix, create_gauge_chart

st.set_page_config(page_title="推薦システム評価", page_icon="🎯", layout="wide")

st.title("🎯 推薦システム評価")
st.markdown("問題推薦の精度を評価します。")
st.markdown("---")

df = load_analysis_data()

if df is not None:
    # 推薦ロジックの説明
    st.header("📋 推薦ロジック")
    
    with st.expander("推薦ロジックの詳細", expanded=False):
        st.markdown("""
        **適切な問題の定義:**
        
        GAP（自信度 - 挑戦度）が0に近い問題を「適切」と判定します。
        
        - **適切**: |GAP| ≤ しきい値
        - **不適切**: |GAP| > しきい値
        
        これにより、学習者にとって「適度に挑戦的」な問題を推薦します。
        """)
    
    # しきい値の設定
    threshold = st.slider(
        "適切/不適切のしきい値（|GAP| ≤ この値なら適切）",
        min_value=0, max_value=5, value=2, step=1
    )
    
    st.markdown("---")
    
    # 適切/不適切の判定
    df['human_appropriate'] = (np.abs(df['human_gap']) <= threshold).astype(int)
    df['ai_appropriate'] = (np.abs(df['ai_gap']) <= threshold).astype(int)
    
    # 混同行列の計算
    tp = ((df['ai_appropriate'] == 1) & (df['human_appropriate'] == 1)).sum()
    fp = ((df['ai_appropriate'] == 1) & (df['human_appropriate'] == 0)).sum()
    fn = ((df['ai_appropriate'] == 0) & (df['human_appropriate'] == 1)).sum()
    tn = ((df['ai_appropriate'] == 0) & (df['human_appropriate'] == 0)).sum()
    
    # 評価指標の計算
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn)
    
    st.header("📊 混同行列")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # 混同行列ヒートマップ
        confusion_matrix = [[tp, fp], [fn, tn]]
        fig = px.imshow(
            confusion_matrix,
            labels=dict(x="AI予測", y="実際（人間）", color="件数"),
            x=['適切と予測', '不適切と予測'],
            y=['実際に適切', '実際に不適切'],
            text_auto=True,
            color_continuous_scale="Blues",
            title="混同行列"
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 混同行列の解釈")
        st.markdown(f"""
        - **TP (True Positive)**: {tp}
          - AIが適切と予測し、実際に適切
        - **FP (False Positive)**: {fp}
          - AIが適切と予測したが、実際は不適切
        - **FN (False Negative)**: {fn}
          - AIが不適切と予測したが、実際は適切
        - **TN (True Negative)**: {tn}
          - AIが不適切と予測し、実際に不適切
        """)
    
    st.markdown("---")
    
    # 評価指標
    st.header("📈 評価指標")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "適合率 (Precision)", 
            f"{precision*100:.1f}%",
            help="AIが適切と予測したもののうち、実際に適切だった割合"
        )
    
    with col2:
        st.metric(
            "再現率 (Recall)", 
            f"{recall*100:.1f}%",
            help="実際に適切なもののうち、AIが適切と予測できた割合"
        )
    
    with col3:
        st.metric(
            "F1スコア", 
            f"{f1*100:.1f}%",
            help="適合率と再現率の調和平均"
        )
    
    with col4:
        st.metric(
            "正解率 (Accuracy)", 
            f"{accuracy*100:.1f}%",
            help="全体の正解率"
        )
    
    # ゲージチャート
    st.subheader("📊 評価指標ゲージ")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=precision * 100,
            title={'text': "適合率"},
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#636EFA"},
                   'steps': [
                       {'range': [0, 33], 'color': "#ffcccc"},
                       {'range': [33, 66], 'color': "#ffffcc"},
                       {'range': [66, 100], 'color': "#ccffcc"}
                   ]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=recall * 100,
            title={'text': "再現率"},
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#EF553B"},
                   'steps': [
                       {'range': [0, 33], 'color': "#ffcccc"},
                       {'range': [33, 66], 'color': "#ffffcc"},
                       {'range': [66, 100], 'color': "#ccffcc"}
                   ]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=f1 * 100,
            title={'text': "F1スコア"},
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#00CC96"},
                   'steps': [
                       {'range': [0, 33], 'color': "#ffcccc"},
                       {'range': [33, 66], 'color': "#ffffcc"},
                       {'range': [66, 100], 'color': "#ccffcc"}
                   ]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=accuracy * 100,
            title={'text': "正解率"},
            number={'suffix': "%"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#AB63FA"},
                   'steps': [
                       {'range': [0, 33], 'color': "#ffcccc"},
                       {'range': [33, 66], 'color': "#ffffcc"},
                       {'range': [66, 100], 'color': "#ccffcc"}
                   ]}
        ))
        fig.update_layout(height=200)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # GAP分布のヒートマップ（3x3マトリクス）
    st.header("📊 AI予測 vs 人間GAPヒートマップ")
    
    # GAPをカテゴリ化（低/中/高）
    def categorize_gap(gap):
        if gap <= -2:
            return '低 (≤-2)'
        elif gap >= 2:
            return '高 (≥2)'
        else:
            return '中 (-1~1)'
    
    df['human_gap_cat'] = df['human_gap'].apply(categorize_gap)
    df['ai_gap_cat'] = df['ai_gap'].apply(categorize_gap)
    
    # クロス集計
    cross_tab = pd.crosstab(df['human_gap_cat'], df['ai_gap_cat'])
    
    # 順序を指定
    order = ['低 (≤-2)', '中 (-1~1)', '高 (≥2)']
    cross_tab = cross_tab.reindex(index=order, columns=order, fill_value=0)
    
    fig = px.imshow(
        cross_tab.values,
        labels=dict(x="AI予測GAP", y="人間GAP", color="件数"),
        x=order,
        y=order,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title="GAP分類の一致度（3×3マトリクス）"
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # 対角成分（一致）の割合
    diagonal_sum = sum(cross_tab.iloc[i, i] for i in range(len(order)))
    agreement_rate = diagonal_sum / len(df) * 100
    
    st.metric(
        "カテゴリ一致率", 
        f"{agreement_rate:.1f}%",
        help="AI予測GAPと人間GAPが同じカテゴリに分類された割合"
    )
    
    st.markdown("---")
    
    # 分布比較
    st.header("📊 適切/不適切の分布比較")
    
    col1, col2 = st.columns(2)
    
    with col1:
        human_counts = df['human_appropriate'].value_counts().sort_index()
        fig = px.pie(
            values=human_counts.values,
            names=['不適切', '適切'],
            title='人間の評価分布',
            color_discrete_sequence=['#EF553B', '#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        ai_counts = df['ai_appropriate'].value_counts().sort_index()
        fig = px.pie(
            values=ai_counts.values,
            names=['不適切', '適切'],
            title='AI予測分布',
            color_discrete_sequence=['#EF553B', '#00CC96']
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.error("データの読み込みに失敗しました。")
