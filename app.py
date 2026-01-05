import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import os
from pathlib import Path
from sqlalchemy import create_engine
import warnings

# 警告を抑制
warnings.filterwarnings('ignore', category=UserWarning)

# ページ設定
st.set_page_config(
    page_title="実験データ分析",
    page_icon="📊",
    layout="wide"
)

def load_env():
    """環境変数を.env.localから直接読み込む"""
    env_path = Path(__file__).parent.parent / '.env.local'
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    # クォートを除去
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value
    return env_vars

@st.cache_resource
def get_engine():
    """SQLAlchemyエンジンを取得"""
    env_vars = load_env()
    database_url = env_vars.get('DATABASE_URL') or os.getenv('DATABASE_URL')
    
    if not database_url:
        st.error("DATABASE_URLが設定されていません。.env.localファイルを確認してください。")
        return None
    
    try:
        engine = create_engine(database_url)
        return engine
    except Exception as e:
        st.error(f"データベース接続エラー: {e}")
        st.info("DATABASE_URLの形式を確認してください: postgresql://user:password@host:port/database")
        return None

@st.cache_data(ttl=60)
def load_data():
    """データベースからデータを読み込む"""
    engine = get_engine()
    if not engine:
        return None, None, None
    
    try:
        # 完了したセッションを取得
        sessions_query = """
            SELECT 
                s.id,
                s.user_id,
                s.started_at,
                s.ended_at,
                s.current_phase,
                u.grade,
                u.major,
                u.linear_algebra_status,
                u.confidence_rating
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.current_phase = 'completed'
            ORDER BY s.ended_at DESC
        """
        
        # final_checkフェーズの回答を取得
        responses_query = """
            SELECT 
                r.id,
                r.session_id,
                r.problem_id,
                r.phase,
                r.confidence,
                r.challenge,
                r.free_text,
                r.ai_predicted_confidence,
                r.ai_predicted_challenge,
                r.created_at,
                p.knowledge_component,
                p.description_main,
                p.description_sub,
                p.answer
            FROM responses r
            JOIN problems p ON r.problem_id = p.id
            JOIN sessions s ON r.session_id = s.id
            WHERE s.current_phase = 'completed'
            AND r.phase = 'final_check'
            AND r.ai_predicted_confidence IS NOT NULL
            AND r.ai_predicted_challenge IS NOT NULL
            ORDER BY r.created_at ASC
        """
        
        # 全問題を取得
        problems_query = """
            SELECT id, knowledge_component, description_main, description_sub, answer
            FROM problems
            ORDER BY id
        """
        
        sessions_df = pd.read_sql(sessions_query, engine)
        responses_df = pd.read_sql(responses_query, engine)
        problems_df = pd.read_sql(problems_query, engine)
        
        # UUIDを文字列に変換（JSON シリアライズ対応）
        for col in ['id', 'user_id', 'session_id', 'problem_id']:
            if col in sessions_df.columns:
                sessions_df[col] = sessions_df[col].astype(str)
            if col in responses_df.columns:
                responses_df[col] = responses_df[col].astype(str)
            if col in problems_df.columns:
                problems_df[col] = problems_df[col].astype(str)
        
        # GAPを計算
        responses_df['human_gap'] = responses_df['confidence'] - responses_df['challenge']
        responses_df['ai_gap'] = responses_df['ai_predicted_confidence'] - responses_df['ai_predicted_challenge']
        responses_df['gap_difference'] = (responses_df['human_gap'] - responses_df['ai_gap']).abs()
        
        return sessions_df, responses_df, problems_df
        
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None, None

def calculate_correlation(x, y):
    """ピアソン相関係数を計算"""
    if len(x) < 2:
        return 0, 1
    return stats.pearsonr(x, y)

def main():
    st.title("📊 実験データ分析ダッシュボード")
    st.markdown("AIの予測と人間の回答の比較分析")
    
    # データ読み込み
    with st.spinner("データを読み込み中..."):
        sessions_df, responses_df, problems_df = load_data()
    
    if sessions_df is None or responses_df is None:
        st.error("データの読み込みに失敗しました")
        st.stop()
    
    if len(responses_df) == 0:
        st.warning("分析対象のデータがありません")
        st.stop()
    
    # サイドバー - フィルター
    st.sidebar.header("🔧 フィルター")
    
    # 学年フィルター
    grades = ['全て'] + sorted(sessions_df['grade'].dropna().unique().tolist())
    selected_grade = st.sidebar.selectbox("学年", grades)
    
    # 知識要素フィルター
    kcs = ['全て'] + sorted(responses_df['knowledge_component'].unique().tolist())
    selected_kc = st.sidebar.selectbox("知識要素", kcs)
    
    # フィルタリング適用
    filtered_responses = responses_df.copy()
    if selected_grade != '全て':
        session_ids = sessions_df[sessions_df['grade'] == selected_grade]['id'].tolist()
        filtered_responses = filtered_responses[filtered_responses['session_id'].isin(session_ids)]
    if selected_kc != '全て':
        filtered_responses = filtered_responses[filtered_responses['knowledge_component'] == selected_kc]
    
    # リロードボタン
    if st.sidebar.button("🔄 データを再読み込み"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    
    # 統計情報を計算
    corr, p_value = calculate_correlation(
        filtered_responses['human_gap'].values,
        filtered_responses['ai_gap'].values
    )
    exact_match = (filtered_responses['human_gap'] == filtered_responses['ai_gap']).sum()
    exact_rate = exact_match / len(filtered_responses) * 100 if len(filtered_responses) > 0 else 0
    within1 = (filtered_responses['gap_difference'] <= 1).sum()
    within1_rate = within1 / len(filtered_responses) * 100 if len(filtered_responses) > 0 else 0
    within2 = (filtered_responses['gap_difference'] <= 2).sum()
    within2_rate = within2 / len(filtered_responses) * 100 if len(filtered_responses) > 0 else 0
    
    # ========== 概要統計 ==========
    st.header("📈 概要統計")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("完了セッション数", len(sessions_df))
    with col2:
        st.metric("総回答数 (final_check)", len(filtered_responses))
    with col3:
        st.metric("相関係数 (r)", f"{corr:.4f}", help=f"p値: {p_value:.6f}")
    with col4:
        st.metric("完全一致率", f"{exact_rate:.1f}%", f"{exact_match}件")
    
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        st.metric("±1以内", f"{within1_rate:.1f}%", f"{within1}件")
    with col6:
        st.metric("±2以内", f"{within2_rate:.1f}%", f"{within2}件")
    with col7:
        avg_human_gap = filtered_responses['human_gap'].mean()
        st.metric("平均人間GAP", f"{avg_human_gap:.2f}")
    with col8:
        avg_ai_gap = filtered_responses['ai_gap'].mean()
        st.metric("平均AI GAP", f"{avg_ai_gap:.2f}")
    
    # 相関の統計的有意性
    st.divider()
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        r_squared = corr ** 2
        st.metric("決定係数 (R²)", f"{r_squared:.4f}")
    with col_stat2:
        # p値を科学的表記法で表示
        if p_value < 1e-10:
            p_display = f"{p_value:.2e}"
        else:
            p_display = f"{p_value:.6f}"
        st.metric("p値", p_display, help="p値が非常に小さい場合は科学的表記法で表示")
    with col_stat3:
        if p_value < 0.001:
            st.success("✅ 統計的に非常に有意 (p < 0.001)")
        elif p_value < 0.01:
            st.success("✅ 統計的に有意 (p < 0.01)")
        elif p_value < 0.05:
            st.info("ℹ️ 統計的に有意 (p < 0.05)")
        else:
            st.warning("⚠️ 統計的に有意ではない (p ≥ 0.05)")
    
    st.divider()
    
    # ========== 可視化 ==========
    st.header("📊 可視化")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["散布図・ヒートマップ", "分布", "問題別分析", "知識要素別", "セッション別", "問題一覧"])
    
    with tab1:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # 人間GAP vs AI GAPの散布図（ジッター付き）
            plot_df = filtered_responses.copy()
            np.random.seed(42)
            plot_df['human_gap_jitter'] = plot_df['human_gap'] + np.random.uniform(-0.15, 0.15, len(plot_df))
            plot_df['ai_gap_jitter'] = plot_df['ai_gap'] + np.random.uniform(-0.15, 0.15, len(plot_df))

            fig = px.scatter(
                plot_df,
                x='human_gap_jitter',
                y='ai_gap_jitter',
                color='knowledge_component',
                hover_data=['problem_id', 'confidence', 'challenge', 'ai_predicted_confidence', 'ai_predicted_challenge', 'human_gap', 'ai_gap'],
                title='人間GAP vs AI GAP (ジッター付き散布図)',
                labels={'human_gap_jitter': '人間GAP', 'ai_gap_jitter': 'AI予測GAP'}
            )
            fig.add_trace(go.Scatter(
                x=[-4, 4], y=[-4, 4],
                mode='lines',
                name='完全一致線',
                line=dict(dash='dash', color='red')
            ))
            fig.update_layout(height=450)
            st.plotly_chart(fig, width="stretch")
        
        with col_chart2:
            # ヒートマップ（頻度分布）
            heatmap_data = pd.crosstab(filtered_responses['ai_gap'], filtered_responses['human_gap'])
            # 欠けている値を埋める
            all_gaps = range(-4, 5)
            for gap in all_gaps:
                if gap not in heatmap_data.index:
                    heatmap_data.loc[gap] = 0
                if gap not in heatmap_data.columns:
                    heatmap_data[gap] = 0
            heatmap_data = heatmap_data.sort_index().sort_index(axis=1)
            
            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="人間GAP", y="AI GAP", color="回答数"),
                x=heatmap_data.columns,
                y=heatmap_data.index,
                text_auto=True,
                color_continuous_scale='Blues',
                title="GAPの組み合わせ頻度ヒートマップ"
            )
            fig_heat.update_layout(height=450)
            st.plotly_chart(fig_heat, width="stretch")
        
        # 相関分析の詳細
        st.subheader("📉 相関分析の詳細")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.write(f"**ピアソン相関係数 (r)**: {corr:.4f}")
            st.write(f"**決定係数 (R²)**: {r_squared:.4f}")
        with col_b:
            st.write(f"**p値**: {p_value:.2e}")
            st.write(f"**サンプルサイズ (n)**: {len(filtered_responses)}")
        with col_c:
            mae = filtered_responses['gap_difference'].mean()
            rmse = np.sqrt((filtered_responses['gap_difference'] ** 2).mean())
            st.write(f"**平均絶対誤差 (MAE)**: {mae:.2f}")
            st.write(f"**RMSE**: {rmse:.2f}")
    
    with tab2:
        col_dist1, col_dist2 = st.columns(2)
        
        with col_dist1:
            fig_human = px.histogram(
                filtered_responses,
                x='human_gap',
                nbins=9,
                title='人間GAPの分布',
                labels={'human_gap': '人間GAP', 'count': '回答数'},
                color_discrete_sequence=['#636EFA']
            )
            fig_human.update_layout(bargap=0.1)
            st.plotly_chart(fig_human, width="stretch")
        
        with col_dist2:
            fig_ai = px.histogram(
                filtered_responses,
                x='ai_gap',
                nbins=9,
                title='AI予測GAPの分布',
                labels={'ai_gap': 'AI GAP', 'count': '回答数'},
                color_discrete_sequence=['#EF553B']
            )
            fig_ai.update_layout(bargap=0.1)
            st.plotly_chart(fig_ai, width="stretch")
        
        # GAP差の分布
        fig_diff = px.histogram(
            filtered_responses,
            x='gap_difference',
            nbins=9,
            title='GAP差（|人間GAP - AI GAP|）の分布',
            labels={'gap_difference': 'GAP差', 'count': '回答数'},
            color_discrete_sequence=['#00CC96']
        )
        st.plotly_chart(fig_diff, width="stretch")
        
        # 自信度・挑戦度の比較
        st.subheader("自信度・挑戦度の比較")
        col_conf, col_chal = st.columns(2)
        with col_conf:
            fig_conf = go.Figure()
            fig_conf.add_trace(go.Histogram(x=filtered_responses['confidence'], name='人間の自信度', opacity=0.7))
            fig_conf.add_trace(go.Histogram(x=filtered_responses['ai_predicted_confidence'], name='AI予測の自信度', opacity=0.7))
            fig_conf.update_layout(barmode='overlay', title='自信度の分布比較')
            st.plotly_chart(fig_conf, width="stretch")
        with col_chal:
            fig_chal = go.Figure()
            fig_chal.add_trace(go.Histogram(x=filtered_responses['challenge'], name='人間の挑戦度', opacity=0.7))
            fig_chal.add_trace(go.Histogram(x=filtered_responses['ai_predicted_challenge'], name='AI予測の挑戦度', opacity=0.7))
            fig_chal.update_layout(barmode='overlay', title='挑戦度の分布比較')
            st.plotly_chart(fig_chal, width="stretch")
    
    # ========== 新しいタブ: 問題別分析 ==========
    with tab3:
        st.subheader("📊 問題別GAP分析")
        
        # 問題ごとの統計を計算
        problem_stats = filtered_responses.groupby(['problem_id', 'knowledge_component', 'description_main']).agg({
            'human_gap': ['mean', 'std'],
            'ai_gap': ['mean', 'std'],
            'gap_difference': ['mean', 'std', 'max'],
            'id': 'count'
        }).reset_index()
        problem_stats.columns = ['問題ID', '知識要素', '問題文', '人間GAP平均', '人間GAP標準偏差', 
                                  'AI GAP平均', 'AI GAP標準偏差', 'GAP差平均', 'GAP差標準偏差', 'GAP差最大', '回答数']
        
        # GAP差平均でソート（降順）
        problem_stats_sorted = problem_stats.sort_values('GAP差平均', ascending=False)
        
        # GAP差が大きい問題 Top 10
        st.markdown("### 🔴 GAP差が大きい問題 Top 10（AIの予測がずれやすい問題）")
        top_gap_problems = problem_stats_sorted.head(10)
        
        fig_top = px.bar(
            top_gap_problems,
            x='問題ID',
            y='GAP差平均',
            color='知識要素',
            hover_data=['問題文', '回答数', '人間GAP平均', 'AI GAP平均'],
            title='GAP差が大きい問題 Top 10',
            error_y='GAP差標準偏差'
        )
        fig_top.update_layout(height=400)
        st.plotly_chart(fig_top, width="stretch")
        
        # GAP差が小さい問題 Top 10
        st.markdown("### 🟢 GAP差が小さい問題 Top 10（AIの予測が正確な問題）")
        bottom_gap_problems = problem_stats_sorted.tail(10).sort_values('GAP差平均')
        
        fig_bottom = px.bar(
            bottom_gap_problems,
            x='問題ID',
            y='GAP差平均',
            color='知識要素',
            hover_data=['問題文', '回答数', '人間GAP平均', 'AI GAP平均'],
            title='GAP差が小さい問題 Top 10',
            error_y='GAP差標準偏差'
        )
        fig_bottom.update_layout(height=400)
        st.plotly_chart(fig_bottom, width="stretch")
        
        # 全問題の散布図（人間GAP vs AI GAP、問題ごと）
        st.markdown("### 📈 問題ごとの平均GAP比較")
        fig_problem_scatter = px.scatter(
            problem_stats,
            x='人間GAP平均',
            y='AI GAP平均',
            size='回答数',
            color='知識要素',
            hover_data=['問題ID', '問題文', 'GAP差平均'],
            title='問題ごとの平均人間GAP vs 平均AI GAP',
        )
        fig_problem_scatter.add_trace(go.Scatter(
            x=[-3, 3], y=[-3, 3],
            mode='lines',
            name='完全一致線',
            line=dict(dash='dash', color='red')
        ))
        fig_problem_scatter.update_layout(height=500)
        st.plotly_chart(fig_problem_scatter, width="stretch")
        
        # 問題別統計テーブル（ソート可能）
        st.markdown("### 📋 問題別統計テーブル")
        sort_option = st.selectbox(
            "ソート基準",
            ["GAP差平均（降順）", "GAP差平均（昇順）", "回答数（降順）", "問題ID（昇順）"]
        )
        
        if sort_option == "GAP差平均（降順）":
            display_stats = problem_stats.sort_values('GAP差平均', ascending=False)
        elif sort_option == "GAP差平均（昇順）":
            display_stats = problem_stats.sort_values('GAP差平均', ascending=True)
        elif sort_option == "回答数（降順）":
            display_stats = problem_stats.sort_values('回答数', ascending=False)
        else:
            display_stats = problem_stats.sort_values('問題ID', ascending=True)
        
        # 表示用に列を選択
        display_cols = ['問題ID', '知識要素', '回答数', '人間GAP平均', 'AI GAP平均', 'GAP差平均', 'GAP差最大']
        st.dataframe(display_stats[display_cols].round(2), width="stretch", hide_index=True)
        
        # GAP差が大きい問題の詳細表示
        st.markdown("### 🔍 問題の詳細")
        selected_problem_id = st.selectbox(
            "問題を選択して詳細を表示",
            options=problem_stats_sorted['問題ID'].tolist(),
            format_func=lambda x: f"問題 {x} (GAP差: {problem_stats_sorted[problem_stats_sorted['問題ID'] == x]['GAP差平均'].values[0]:.2f})"
        )
        
        if selected_problem_id:
            problem_data = filtered_responses[filtered_responses['problem_id'] == selected_problem_id]
            problem_info = problem_stats[problem_stats['問題ID'] == selected_problem_id].iloc[0]
            
            st.markdown(f"**問題文:** {problem_info['問題文']}")
            st.markdown(f"**知識要素:** {problem_info['知識要素']}")
            
            col_detail1, col_detail2, col_detail3, col_detail4 = st.columns(4)
            with col_detail1:
                st.metric("回答数", int(problem_info['回答数']))
            with col_detail2:
                st.metric("平均人間GAP", f"{problem_info['人間GAP平均']:.2f}")
            with col_detail3:
                st.metric("平均AI GAP", f"{problem_info['AI GAP平均']:.2f}")
            with col_detail4:
                st.metric("平均GAP差", f"{problem_info['GAP差平均']:.2f}")
            
            # この問題の回答分布
            col_prob1, col_prob2 = st.columns(2)
            with col_prob1:
                fig_prob_human = px.histogram(
                    problem_data,
                    x='human_gap',
                    nbins=9,
                    title=f'問題{selected_problem_id}: 人間GAPの分布',
                    color_discrete_sequence=['#636EFA']
                )
                st.plotly_chart(fig_prob_human, width="stretch")
            with col_prob2:
                fig_prob_ai = px.histogram(
                    problem_data,
                    x='ai_gap',
                    nbins=9,
                    title=f'問題{selected_problem_id}: AI GAPの分布',
                    color_discrete_sequence=['#EF553B']
                )
                st.plotly_chart(fig_prob_ai, width="stretch")
    
    with tab4:
        # 知識要素別の分析
        kc_stats = filtered_responses.groupby('knowledge_component').agg({
            'human_gap': ['mean', 'std'],
            'ai_gap': ['mean', 'std'],
            'gap_difference': 'mean',
            'id': 'count'
        }).reset_index()
        kc_stats.columns = ['知識要素', '人間GAP平均', '人間GAP標準偏差', 'AI GAP平均', 'AI GAP標準偏差', 'GAP差平均', '回答数']
        kc_stats = kc_stats.sort_values('回答数', ascending=False)
        
        fig_kc = px.bar(
            kc_stats,
            x='知識要素',
            y=['人間GAP平均', 'AI GAP平均'],
            barmode='group',
            title='知識要素別 平均GAP比較',
            error_y=kc_stats['人間GAP標準偏差']
        )
        fig_kc.update_layout(height=400)
        st.plotly_chart(fig_kc, width="stretch")
        
        st.subheader("知識要素別統計テーブル")
        st.dataframe(kc_stats.round(2), width="stretch", hide_index=True)
        
        # 知識要素別相関係数
        st.subheader("知識要素別相関係数")
        kc_corr_list = []
        for kc in filtered_responses['knowledge_component'].unique():
            kc_data = filtered_responses[filtered_responses['knowledge_component'] == kc]
            if len(kc_data) >= 3:
                kc_corr, kc_p = calculate_correlation(kc_data['human_gap'].values, kc_data['ai_gap'].values)
                kc_corr_list.append({
                    '知識要素': kc,
                    '相関係数': kc_corr,
                    'p値': kc_p,
                    'サンプル数': len(kc_data)
                })
        if kc_corr_list:
            kc_corr_df = pd.DataFrame(kc_corr_list).sort_values('サンプル数', ascending=False)
            st.dataframe(kc_corr_df.round(4), width="stretch", hide_index=True)
    
    with tab5:
        # セッション別の分析
        session_stats = filtered_responses.groupby('session_id').agg({
            'human_gap': 'mean',
            'ai_gap': 'mean',
            'gap_difference': 'mean',
            'id': 'count'
        }).rename(columns={'id': 'count'}).reset_index()
        
        session_stats = session_stats.merge(
            sessions_df[['id', 'grade', 'major', 'confidence_rating']],
            left_on='session_id',
            right_on='id',
            how='left'
        )
        
        fig_session = px.scatter(
            session_stats,
            x='human_gap',
            y='ai_gap',
            size='count',
            color='grade',
            hover_data=['session_id', 'major', 'count', 'gap_difference'],
            title='セッション別 平均GAP (点のサイズ = 回答数)',
            labels={'human_gap': '平均人間GAP', 'ai_gap': '平均AI GAP'}
        )
        fig_session.add_trace(go.Scatter(
            x=[-2, 2], y=[-2, 2],
            mode='lines',
            name='完全一致線',
            line=dict(dash='dash', color='red')
        ))
        st.plotly_chart(fig_session, width="stretch")
        
        st.subheader("セッション別統計")
        session_display = session_stats[['session_id', 'grade', 'major', 'human_gap', 'ai_gap', 'gap_difference', 'count']].copy()
        session_display.columns = ['セッションID', '学年', '専攻', '平均人間GAP', '平均AI GAP', '平均GAP差', '回答数']
        st.dataframe(session_display.round(2), width="stretch", hide_index=True)
    
    with tab6:
        st.subheader("📋 問題一覧")
        
        if problems_df is not None and len(problems_df) > 0:
            # 問題ごとの回答統計を追加
            problem_stats_tab6 = filtered_responses.groupby('problem_id').agg({
                'human_gap': 'mean',
                'ai_gap': 'mean',
                'gap_difference': 'mean',
                'id': 'count'
            }).rename(columns={'id': 'response_count'}).reset_index()
            
            problems_with_stats = problems_df.merge(problem_stats_tab6, left_on='id', right_on='problem_id', how='left')
            problems_with_stats['response_count'] = problems_with_stats['response_count'].fillna(0).astype(int)
            
            # 表示用にフォーマット
            for idx, row in problems_with_stats.iterrows():
                with st.expander(f"問題 {row['id']}: {row['knowledge_component']} (回答数: {row['response_count']})"):
                    st.markdown(f"**問題文:** {row['description_main']}")
                    if pd.notna(row['description_sub']):
                        st.markdown(f"**補足:** {row['description_sub']}")
                    if pd.notna(row['answer']):
                        st.markdown(f"**答え:** {row['answer']}")
                    
                    if row['response_count'] > 0:
                        col_p1, col_p2, col_p3 = st.columns(3)
                        with col_p1:
                            st.metric("平均人間GAP", f"{row['human_gap']:.2f}" if pd.notna(row['human_gap']) else "N/A")
                        with col_p2:
                            st.metric("平均AI GAP", f"{row['ai_gap']:.2f}" if pd.notna(row['ai_gap']) else "N/A")
                        with col_p3:
                            st.metric("平均GAP差", f"{row['gap_difference']:.2f}" if pd.notna(row['gap_difference']) else "N/A")
        else:
            st.info("問題データがありません")
    
    st.divider()
    
    # ========== 詳細データ ==========
    st.header("📋 詳細データ")
    
    with st.expander("回答データ一覧", expanded=False):
        display_cols = [
            'session_id', 'problem_id', 'knowledge_component',
            'confidence', 'challenge', 'human_gap',
            'ai_predicted_confidence', 'ai_predicted_challenge', 'ai_gap',
            'gap_difference', 'free_text'
        ]
        st.dataframe(
            filtered_responses[display_cols].round(2),
            width="stretch",
            height=400
        )
    
    with st.expander("セッション一覧", expanded=False):
        st.dataframe(sessions_df, width="stretch")
    
    # CSVダウンロード
    st.subheader("📥 データエクスポート")
    col_dl1, col_dl2, col_dl3 = st.columns(3)
    
    with col_dl1:
        csv_responses = filtered_responses.to_csv(index=False)
        st.download_button(
            label="回答データをCSVでダウンロード",
            data=csv_responses,
            file_name="responses_data.csv",
            mime="text/csv"
        )
    
    with col_dl2:
        csv_sessions = sessions_df.to_csv(index=False)
        st.download_button(
            label="セッションデータをCSVでダウンロード",
            data=csv_sessions,
            file_name="sessions_data.csv",
            mime="text/csv"
        )
    
    with col_dl3:
        if problems_df is not None:
            csv_problems = problems_df.to_csv(index=False)
            st.download_button(
                label="問題データをCSVでダウンロード",
                data=csv_problems,
                file_name="problems_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
