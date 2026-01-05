"""
挑戦度分析ページ
AIが予測した挑戦度と人間の実際の挑戦度を比較分析
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import sys
from pathlib import Path

# 親ディレクトリをパスに追加
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import load_raw_data, prepare_analysis_data, calculate_metrics

# ページ設定
st.set_page_config(
    page_title="挑戦度分析",
    page_icon="🎯",
    layout="wide"
)


def render_overview_section(df):
    """全体概要セクション"""
    st.subheader("📈 全体概要")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("総回答数", len(df))
    with col2:
        st.metric("ユーザー数", df['session_id'].nunique())
    with col3:
        st.metric("問題数", df['problem_id'].nunique())
    with col4:
        mae = df['value_difference'].mean()
        st.metric("MAE（AI vs 人間）", f"{mae:.2f}")
    
    # 基本統計
    col_stat1, col_stat2 = st.columns(2)
    with col_stat1:
        st.markdown("#### 人間の挑戦度")
        st.write(f"- 平均: {df['human_value'].mean():.2f}")
        st.write(f"- 標準偏差: {df['human_value'].std():.2f}")
        st.write(f"- 範囲: {df['human_value'].min():.0f} 〜 {df['human_value'].max():.0f}")
    
    with col_stat2:
        st.markdown("#### AIの予測挑戦度")
        st.write(f"- 平均: {df['ai_value'].mean():.2f}")
        st.write(f"- 標準偏差: {df['ai_value'].std():.2f}")
        st.write(f"- 範囲: {df['ai_value'].min():.0f} 〜 {df['ai_value'].max():.0f}")


def render_scatter_section(df):
    """散布図セクション"""
    st.subheader("📊 AI予測 vs 人間の挑戦度")
    
    # 相関係数の計算
    metrics = calculate_metrics(df['human_value'].values, df['ai_value'].values)
    
    col_scatter, col_metrics = st.columns([2, 1])
    
    with col_scatter:
        fig = px.scatter(df, x='ai_value', y='human_value',
                         hover_data=['problem_id', 'session_id'],
                         title=f'AI予測 vs 人間の挑戦度 (r={metrics["r"]:.3f})')
        
        # 対角線を追加
        min_val = min(df['ai_value'].min(), df['human_value'].min())
        max_val = max(df['ai_value'].max(), df['human_value'].max())
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', name='完全一致線',
                                 line=dict(dash='dash', color='red')))
        
        fig.update_layout(
            xaxis_title='AI予測挑戦度',
            yaxis_title='人間の挑戦度',
            height=450
        )
        st.plotly_chart(fig, key="scatter_challenge")
    
    with col_metrics:
        st.markdown("#### 評価指標")
        st.metric("相関係数 (r)", f"{metrics['r']:.3f}")
        st.metric("決定係数 (R²)", f"{metrics['r2']:.3f}")
        st.metric("MAE", f"{metrics['mae']:.2f}")
        st.metric("RMSE", f"{metrics['rmse']:.2f}")
        st.metric("p値", f"{metrics['p_value']:.4f}")


def render_baseline_section(df):
    """ベースライン比較セクション"""
    st.subheader("🎯 ベースライン比較")
    
    with st.expander("📐 ベースラインの計算方法"):
        st.markdown("""
### ベースラインの定義（Leave-One-Out方式）

⚠️ 予測対象のデータを除外して平均を計算し、公正な比較を実現しています。

#### 全体平均ベースライン
全回答の挑戦度平均値を予測値として使用

#### 問題平均ベースライン
同じ問題の「自分以外」の回答者の挑戦度平均値を予測値として使用

#### ユーザー平均ベースライン
同じユーザーの「この問題以外」の挑戦度平均値を予測値として使用
        """)
    
    # 各ベースラインのメトリクス計算
    baselines = {
        'LLM予測': df['ai_value'].values,
        '全体平均': np.full(len(df), df['human_value'].mean()),
        '問題平均': df['problem_mean'].values,
        'ユーザー平均': df['user_mean'].values
    }
    
    baseline_results = []
    for name, pred in baselines.items():
        m = calculate_metrics(df['human_value'].values, pred)
        exact_match = (df['human_value'].values == np.round(pred)).sum() / len(df) * 100
        within1_match = (np.abs(df['human_value'].values - pred) <= 1).sum() / len(df) * 100
        baseline_results.append({
            'ベースライン': name,
            'MAE': m['mae'],
            '相関係数 (r)': m['r'],
            'R²': m['r2'],
            '完全一致率 (%)': exact_match,
            '±1以内 (%)': within1_match
        })
    
    baseline_df = pd.DataFrame(baseline_results)
    st.dataframe(baseline_df.round(4), hide_index=True)
    
    # MAE比較の棒グラフ
    fig_baseline = px.bar(baseline_df, x='ベースライン', y='MAE', 
                          title='ベースライン別MAE比較（低いほど良い）',
                          color='ベースライン', text='MAE')
    fig_baseline.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    fig_baseline.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig_baseline, key="baseline_mae_chal")


def render_distribution_section(df):
    """分布セクション"""
    st.subheader("📊 値の分布")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_human = px.histogram(df, x='human_value', nbins=10,
                                 title='人間の挑戦度分布',
                                 labels={'human_value': '挑戦度'})
        fig_human.update_layout(height=300)
        st.plotly_chart(fig_human, key="hist_human_chal")
    
    with col2:
        fig_ai = px.histogram(df, x='ai_value', nbins=10,
                              title='AI予測挑戦度分布',
                              labels={'ai_value': '予測挑戦度'})
        fig_ai.update_layout(height=300)
        st.plotly_chart(fig_ai, key="hist_ai_chal")
    
    # 差分の分布
    df['diff'] = df['ai_value'] - df['human_value']
    fig_diff = px.histogram(df, x='diff', nbins=20,
                            title='AI予測 - 人間の挑戦度（差分分布）',
                            labels={'diff': '差分（AI - 人間）'})
    fig_diff.add_vline(x=0, line_dash="dash", line_color="red")
    fig_diff.update_layout(height=300)
    st.plotly_chart(fig_diff, key="hist_diff_chal")
    
    # 差分の統計
    col_diff1, col_diff2, col_diff3 = st.columns(3)
    with col_diff1:
        st.metric("平均差分", f"{df['diff'].mean():.2f}", 
                  help="正: AIが過大評価、負: AIが過小評価")
    with col_diff2:
        over = (df['diff'] > 0).sum()
        st.metric("AI過大評価", f"{over}件 ({over/len(df)*100:.1f}%)")
    with col_diff3:
        under = (df['diff'] < 0).sum()
        st.metric("AI過小評価", f"{under}件 ({under/len(df)*100:.1f}%)")


def render_problem_section(df, problems_df):
    """問題別分析セクション"""
    st.subheader("📝 問題別分析")
    
    problem_stats = df.groupby('problem_id').agg({
        'human_value': ['mean', 'std', 'count'],
        'ai_value': 'mean',
        'value_difference': 'mean',
        'knowledge_component': 'first'
    }).round(2)
    problem_stats.columns = ['人間平均', '人間標準偏差', '回答数', 'AI平均', 'MAE', '知識要素']
    problem_stats = problem_stats.reset_index()
    
    # 問題別MAEでソート
    problem_stats = problem_stats.sort_values('MAE', ascending=False)
    
    # 上位・下位の問題
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### MAEが高い問題（予測が困難）")
        st.dataframe(problem_stats.head(5)[['problem_id', '知識要素', 'MAE', '人間平均', 'AI平均']], hide_index=True)
    
    with col2:
        st.markdown("#### MAEが低い問題（予測が容易）")
        st.dataframe(problem_stats.tail(5)[['problem_id', '知識要素', 'MAE', '人間平均', 'AI平均']], hide_index=True)
    
    # 問題別MAEのグラフ
    fig = px.bar(problem_stats.sort_values('MAE'), 
                 x='knowledge_component' if 'knowledge_component' in problem_stats.columns else 'problem_id',
                 y='MAE',
                 title='問題別MAE',
                 color='MAE',
                 color_continuous_scale='Reds')
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, key="problem_mae_chal")


def render_user_section(df):
    """ユーザー別分析セクション"""
    st.subheader("👤 ユーザー別分析")
    
    user_stats = df.groupby('session_id').agg({
        'human_value': 'mean',
        'ai_value': 'mean',
        'value_difference': 'mean',
        'problem_id': 'count'
    }).round(2)
    user_stats.columns = ['人間平均', 'AI平均', 'MAE', '回答数']
    user_stats = user_stats.reset_index()
    user_stats['ユーザー番号'] = range(1, len(user_stats) + 1)
    
    # ユーザー別MAE
    fig = px.bar(user_stats.sort_values('MAE'), 
                 x='ユーザー番号',
                 y='MAE',
                 title='ユーザー別MAE',
                 color='MAE',
                 color_continuous_scale='Blues')
    fig.update_layout(height=350)
    st.plotly_chart(fig, key="user_mae_chal")
    
    # ユーザーごとの散布図
    fig_scatter = px.scatter(user_stats, x='人間平均', y='AI平均',
                             hover_data=['ユーザー番号', 'MAE'],
                             title='ユーザー別: 人間平均 vs AI平均')
    # 対角線
    min_val = min(user_stats['人間平均'].min(), user_stats['AI平均'].min())
    max_val = max(user_stats['人間平均'].max(), user_stats['AI平均'].max())
    fig_scatter.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                     mode='lines', name='完全一致線',
                                     line=dict(dash='dash', color='red')))
    fig_scatter.update_layout(height=350)
    st.plotly_chart(fig_scatter, key="user_scatter_chal")


def render_difficulty_insight(df):
    """挑戦度に関する洞察セクション"""
    st.subheader("💡 挑戦度の洞察")
    
    st.markdown("""
    挑戦度は「問題に対してどれだけ挑戦的に取り組もうとしているか」を示します。
    - **高い挑戦度**: 難しくても積極的に取り組む姿勢
    - **低い挑戦度**: 確実にできる範囲で取り組む姿勢
    """)
    
    # 挑戦度と問題の関係
    problem_challenge = df.groupby('problem_id').agg({
        'human_value': 'mean',
        'knowledge_component': 'first'
    }).reset_index()
    problem_challenge.columns = ['problem_id', '平均挑戦度', '知識要素']
    
    fig = px.bar(problem_challenge.sort_values('平均挑戦度', ascending=False),
                 x='知識要素', y='平均挑戦度',
                 title='問題別平均挑戦度（高い順）',
                 color='平均挑戦度',
                 color_continuous_scale='Viridis')
    fig.update_layout(height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig, key="problem_challenge_insight")


def main():
    st.title("🎯 挑戦度分析")
    st.markdown("AIが予測した挑戦度と人間の実際の挑戦度を比較分析します")
    
    # データ読み込み
    with st.spinner("データを読み込み中..."):
        sessions_df, responses_df, problems_df = load_raw_data()
    
    if responses_df is None or len(responses_df) == 0:
        st.error("データの読み込みに失敗しました")
        st.stop()
    
    # 挑戦度用のデータ準備
    df = prepare_analysis_data(responses_df, 'challenge', 'ai_predicted_challenge')
    
    st.info(f"📊 分析対象: {len(df)}件の回答データ（{df['session_id'].nunique()}ユーザー、{df['problem_id'].nunique()}問題）")
    
    # 各セクションを表示
    render_overview_section(df)
    st.divider()
    
    render_scatter_section(df)
    st.divider()
    
    render_baseline_section(df)
    st.divider()
    
    render_distribution_section(df)
    st.divider()
    
    render_problem_section(df, problems_df)
    st.divider()
    
    render_user_section(df)
    st.divider()
    
    render_difficulty_insight(df)


if __name__ == "__main__":
    main()
