"""
📊 ベースライン比較ページ
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    load_analysis_data, 
    calculate_baseline_predictions, 
    calculate_metrics,
    create_grouped_bar_chart
)

st.set_page_config(page_title="ベースライン比較", page_icon="📊", layout="wide")

st.title("📊 ベースライン比較")
st.markdown("LLM予測と各種ベースラインの性能を比較します。")
st.markdown("---")

# データ読み込み
df = load_analysis_data()

if df is not None:
    # ベースライン予測を追加
    df = calculate_baseline_predictions(df)
    
    # タブで切り替え
    tab1, tab2, tab3 = st.tabs(["🎯 自信度", "💪 挑戦度", "📐 GAP"])
    
    # 評価対象の設定
    models = {
        'LLM': {
            'confidence': 'ai_predicted_confidence',
            'challenge': 'ai_predicted_challenge', 
            'gap': 'ai_gap'
        },
        '全体平均': {
            'confidence': 'baseline_mean_confidence',
            'challenge': 'baseline_mean_challenge',
            'gap': 'baseline_mean_gap'
        },
        'ランダム': {
            'confidence': 'baseline_random_confidence',
            'challenge': 'baseline_random_challenge',
            'gap': 'baseline_random_gap'
        },
        '中央値(4)': {
            'confidence': 'baseline_median_confidence',
            'challenge': 'baseline_median_challenge',
            'gap': 'baseline_median_gap'
        },
        'ユーザー平均': {
            'confidence': 'baseline_user_mean_confidence',
            'challenge': 'baseline_user_mean_challenge',
            'gap': 'baseline_user_mean_gap'
        }
    }
    
    def create_comparison_table(target_type):
        """比較テーブルを作成"""
        human_col = 'confidence' if target_type == 'confidence' else ('challenge' if target_type == 'challenge' else 'human_gap')
        
        results = []
        for model_name, cols in models.items():
            pred_col = cols[target_type]
            metrics = calculate_metrics(df[human_col], df[pred_col])
            metrics['モデル'] = model_name
            results.append(metrics)
        
        results_df = pd.DataFrame(results)
        # モデル列を先頭に
        cols = ['モデル'] + [c for c in results_df.columns if c != 'モデル']
        return results_df[cols]
    
    with tab1:
        st.subheader("自信度予測の比較")
        
        conf_df = create_comparison_table('confidence')
        
        # テーブル表示
        st.dataframe(
            conf_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                        .highlight_max(subset=['相関係数', 'R²', '完全一致率(%)', '±1以内(%)'], color='lightgreen')
                        .format({
                            'MAE': '{:.3f}',
                            'RMSE': '{:.3f}',
                            '相関係数': '{:.3f}',
                            'p値': '{:.2e}',
                            'R²': '{:.3f}',
                            '完全一致率(%)': '{:.1f}',
                            '±1以内(%)': '{:.1f}'
                        }),
            use_container_width=True
        )
        
        # LLMが勝利しているか確認
        llm_row = conf_df[conf_df['モデル'] == 'LLM'].iloc[0]
        wins = []
        for _, row in conf_df.iterrows():
            if row['モデル'] != 'LLM':
                if llm_row['MAE'] < row['MAE']:
                    wins.append(f"vs {row['モデル']} (MAE)")
        
        if len(wins) == len(models) - 1:
            st.success("✅ LLMは全ベースラインに対してMAEで勝利しています！")
        
        # グラフ表示
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(conf_df, x='モデル', y='MAE', color='モデル', title='MAE比較（低いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(conf_df, x='モデル', y='相関係数', color='モデル', title='相関係数比較（高いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("挑戦度予測の比較")
        
        chal_df = create_comparison_table('challenge')
        
        st.dataframe(
            chal_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                        .highlight_max(subset=['相関係数', 'R²', '完全一致率(%)', '±1以内(%)'], color='lightgreen')
                        .format({
                            'MAE': '{:.3f}',
                            'RMSE': '{:.3f}',
                            '相関係数': '{:.3f}',
                            'p値': '{:.2e}',
                            'R²': '{:.3f}',
                            '完全一致率(%)': '{:.1f}',
                            '±1以内(%)': '{:.1f}'
                        }),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(chal_df, x='モデル', y='MAE', color='モデル', title='MAE比較（低いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(chal_df, x='モデル', y='相関係数', color='モデル', title='相関係数比較（高いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("GAP予測の比較")
        
        gap_df = create_comparison_table('gap')
        
        st.dataframe(
            gap_df.style.highlight_min(subset=['MAE', 'RMSE'], color='lightgreen')
                       .highlight_max(subset=['相関係数', 'R²', '完全一致率(%)', '±1以内(%)'], color='lightgreen')
                       .format({
                           'MAE': '{:.3f}',
                           'RMSE': '{:.3f}',
                           '相関係数': '{:.3f}',
                           'p値': '{:.2e}',
                           'R²': '{:.3f}',
                           '完全一致率(%)': '{:.1f}',
                           '±1以内(%)': '{:.1f}'
                       }),
            use_container_width=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(gap_df, x='モデル', y='MAE', color='モデル', title='MAE比較（低いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(gap_df, x='モデル', y='相関係数', color='モデル', title='相関係数比較（高いほど良い）')
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # 勝敗ヒートマップ
    st.header("🏆 勝敗ヒートマップ")
    
    metric_choice = st.selectbox("比較指標", ['MAE', '相関係数', 'R²'])
    target_choice = st.selectbox("対象", ['自信度', '挑戦度', 'GAP'])
    
    target_map = {'自信度': 'confidence', '挑戦度': 'challenge', 'GAP': 'gap'}
    comparison_df = create_comparison_table(target_map[target_choice])
    
    model_names = comparison_df['モデル'].tolist()
    n_models = len(model_names)
    win_matrix = np.zeros((n_models, n_models))
    
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if i != j:
                val_i = comparison_df[comparison_df['モデル'] == model_i][metric_choice].values[0]
                val_j = comparison_df[comparison_df['モデル'] == model_j][metric_choice].values[0]
                
                if metric_choice == 'MAE':
                    win_matrix[i, j] = 1 if val_i < val_j else (-1 if val_i > val_j else 0)
                else:
                    win_matrix[i, j] = 1 if val_i > val_j else (-1 if val_i < val_j else 0)
    
    fig = px.imshow(
        win_matrix,
        x=model_names,
        y=model_names,
        color_continuous_scale='RdYlGn',
        zmin=-1, zmax=1,
        labels={'color': '勝敗'},
        title=f'{target_choice}の{metric_choice}による勝敗（行が列に勝つ: 緑）'
    )
    fig.update_traces(text=[['+' if v > 0 else ('-' if v < 0 else '=') for v in row] for row in win_matrix], texttemplate="%{text}")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("データの読み込みに失敗しました。")
