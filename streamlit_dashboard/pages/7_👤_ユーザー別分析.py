"""
👤 ユーザー別分析ページ
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_analysis_data, calculate_metrics

st.set_page_config(page_title="ユーザー別分析", page_icon="👤", layout="wide")

st.title("👤 ユーザー別分析")
st.markdown("個々のユーザーごとの分析結果を確認できます。")
st.markdown("---")

df = load_analysis_data()

if df is not None:
    # しきい値設定
    threshold = st.sidebar.slider(
        "適切/不適切のしきい値",
        min_value=0, max_value=5, value=2, step=1
    )
    
    # ユーザーごとの評価指標を計算
    df['human_appropriate'] = (np.abs(df['human_gap']) <= threshold).astype(int)
    df['ai_appropriate'] = (np.abs(df['ai_gap']) <= threshold).astype(int)
    
    user_metrics = []
    
    for user_id in df['user_id'].unique():
        user_df = df[df['user_id'] == user_id]
        
        # 混同行列
        tp = ((user_df['ai_appropriate'] == 1) & (user_df['human_appropriate'] == 1)).sum()
        fp = ((user_df['ai_appropriate'] == 1) & (user_df['human_appropriate'] == 0)).sum()
        fn = ((user_df['ai_appropriate'] == 0) & (user_df['human_appropriate'] == 1)).sum()
        tn = ((user_df['ai_appropriate'] == 0) & (user_df['human_appropriate'] == 0)).sum()
        
        # 評価指標
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / len(user_df) if len(user_df) > 0 else 0
        
        # GAP予測の評価指標
        gap_metrics = calculate_metrics(user_df['human_gap'], user_df['ai_gap'])
        
        user_metrics.append({
            'ユーザー': user_id,
            'サンプル数': len(user_df),
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'TN': tn,
            '適合率': precision * 100,
            '再現率': recall * 100,
            'F1': f1 * 100,
            '正解率': accuracy * 100,
            'GAP_MAE': gap_metrics.get('MAE', np.nan),
            'GAP_相関': gap_metrics.get('相関係数', np.nan),
            '自信度平均': user_df['confidence'].mean(),
            '挑戦度平均': user_df['challenge'].mean(),
            'GAP平均': user_df['human_gap'].mean()
        })
    
    user_df_summary = pd.DataFrame(user_metrics)
    
    # ユーザー別F1スコア棒グラフ
    st.header("📊 ユーザー別F1スコア")
    
    fig = px.bar(
        user_df_summary.sort_values('F1', ascending=False),
        x='ユーザー', y='F1',
        color='F1',
        color_continuous_scale='RdYlGn',
        title='ユーザー別F1スコア（降順）',
        labels={'F1': 'F1スコア (%)'}
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)
    
    # 統計サマリー
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("F1スコア平均", f"{user_df_summary['F1'].mean():.1f}%")
    with col2:
        st.metric("F1スコア標準偏差", f"{user_df_summary['F1'].std():.1f}%")
    with col3:
        st.metric("F1スコア最高", f"{user_df_summary['F1'].max():.1f}%")
    
    st.markdown("---")
    
    # ユーザー詳細テーブル
    st.header("📋 ユーザー別詳細テーブル")
    
    st.dataframe(
        user_df_summary.style.format({
            '適合率': '{:.1f}',
            '再現率': '{:.1f}',
            'F1': '{:.1f}',
            '正解率': '{:.1f}',
            'GAP_MAE': '{:.3f}',
            'GAP_相関': '{:.3f}',
            '自信度平均': '{:.2f}',
            '挑戦度平均': '{:.2f}',
            'GAP平均': '{:.2f}'
        }).background_gradient(subset=['F1'], cmap='RdYlGn'),
        use_container_width=True
    )
    
    st.markdown("---")
    
    # 個別ユーザー分析
    st.header("🔍 個別ユーザー分析")
    
    selected_user = st.selectbox(
        "ユーザーを選択",
        user_df_summary.sort_values('F1', ascending=False)['ユーザー'].tolist()
    )
    
    if selected_user:
        user_data = df[df['user_id'] == selected_user]
        user_stats = user_df_summary[user_df_summary['ユーザー'] == selected_user].iloc[0]
        
        st.subheader(f"📊 {selected_user} の詳細")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("サンプル数", int(user_stats['サンプル数']))
            st.metric("F1スコア", f"{user_stats['F1']:.1f}%")
        
        with col2:
            st.metric("適合率", f"{user_stats['適合率']:.1f}%")
            st.metric("再現率", f"{user_stats['再現率']:.1f}%")
        
        with col3:
            st.metric("GAP MAE", f"{user_stats['GAP_MAE']:.3f}")
            st.metric("GAP 相関", f"{user_stats['GAP_相関']:.3f}")
        
        with col4:
            st.metric("自信度平均", f"{user_stats['自信度平均']:.2f}")
            st.metric("挑戦度平均", f"{user_stats['挑戦度平均']:.2f}")
        
        # 個人の散布図
        st.subheader("📈 自信度 vs 挑戦度")
        
        fig = px.scatter(
            user_data, 
            x='confidence', y='challenge',
            color='human_gap',
            color_continuous_scale='RdYlGn',
            title=f'{selected_user}の自信度 vs 挑戦度',
            labels={'confidence': '自信度', 'challenge': '挑戦度', 'human_gap': 'GAP'},
            hover_data=['problem_id']
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # AI予測との比較
        st.subheader("📊 人間GAP vs AI予測GAP")
        
        fig = px.scatter(
            user_data,
            x='human_gap', y='ai_gap',
            title=f'{selected_user}の人間GAP vs AI予測GAP',
            labels={'human_gap': '人間GAP', 'ai_gap': 'AI予測GAP'},
            trendline='ols'
        )
        # 対角線を追加
        min_val = min(user_data['human_gap'].min(), user_data['ai_gap'].min())
        max_val = max(user_data['human_gap'].max(), user_data['ai_gap'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode='lines', line=dict(dash='dash', color='gray'),
            name='y=x（完全一致）'
        ))
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
        
        # データテーブル
        with st.expander("📋 詳細データ", expanded=False):
            display_cols = ['problem_id', 'confidence', 'challenge', 'human_gap',
                          'ai_predicted_confidence', 'ai_predicted_challenge', 'ai_gap']
            st.dataframe(user_data[display_cols], use_container_width=True)
    
    st.markdown("---")
    
    # ユーザー間の比較
    st.header("📊 ユーザー間の比較")
    
    tab1, tab2, tab3 = st.tabs(["F1スコア分布", "GAP MAE分布", "相関係数分布"])
    
    with tab1:
        fig = px.histogram(
            user_df_summary, x='F1',
            nbins=10,
            title='ユーザー別F1スコアの分布',
            labels={'F1': 'F1スコア (%)'}
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        fig = px.histogram(
            user_df_summary, x='GAP_MAE',
            nbins=10,
            title='ユーザー別GAP MAEの分布',
            labels={'GAP_MAE': 'MAE'}
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = px.histogram(
            user_df_summary, x='GAP_相関',
            nbins=10,
            title='ユーザー別GAP相関係数の分布',
            labels={'GAP_相関': '相関係数'}
        )
        fig.update_layout(template='plotly_white')
        st.plotly_chart(fig, use_container_width=True)
    
    # ユーザー特性とAI精度の関係
    st.header("🔗 ユーザー特性とAI精度の関係")
    
    fig = px.scatter(
        user_df_summary,
        x='GAP平均', y='F1',
        size='サンプル数',
        color='GAP_相関',
        color_continuous_scale='RdYlGn',
        hover_name='ユーザー',
        title='ユーザーの平均GAPとF1スコアの関係',
        labels={'GAP平均': '平均GAP', 'F1': 'F1スコア (%)', 'GAP_相関': 'GAP相関'}
    )
    fig.update_layout(template='plotly_white')
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("データの読み込みに失敗しました。")
