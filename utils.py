"""
共通ユーティリティ関数
"""
import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import os
from pathlib import Path
from sqlalchemy import create_engine


def load_env():
    """環境変数を.envまたは.env.localから読み込む"""
    # まず同じディレクトリの.envを試す
    env_path = Path(__file__).parent / '.env'
    if not env_path.exists():
        # 親ディレクトリの.env.localを試す
        env_path = Path(__file__).parent.parent / '.env.local'
    
    env_vars = {}
    if env_path.exists():
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    env_vars[key.strip()] = value
    return env_vars


@st.cache_resource
def get_engine():
    """SQLAlchemyエンジンを取得"""
    env_vars = load_env()
    database_url = env_vars.get('DATABASE_URL') or os.getenv('DATABASE_URL')
    
    if not database_url:
        st.error("DATABASE_URLが設定されていません。")
        return None
    
    try:
        engine = create_engine(database_url)
        return engine
    except Exception as e:
        st.error(f"データベース接続エラー: {e}")
        return None


@st.cache_data(ttl=60)
def load_raw_data():
    """データベースから生データを読み込む"""
    engine = get_engine()
    if not engine:
        return None, None, None
    
    try:
        sessions_query = """
            SELECT 
                s.id, s.user_id, s.started_at, s.ended_at, s.current_phase,
                u.grade, u.major, u.linear_algebra_status, u.confidence_rating
            FROM sessions s
            JOIN users u ON s.user_id = u.id
            WHERE s.current_phase = 'completed'
            ORDER BY s.ended_at DESC
        """
        
        responses_query = """
            SELECT 
                r.id, r.session_id, r.problem_id, r.phase,
                r.confidence, r.challenge, r.free_text,
                r.ai_predicted_confidence, r.ai_predicted_challenge,
                r.created_at,
                p.knowledge_component, p.description_main, p.description_sub, p.answer
            FROM responses r
            JOIN problems p ON r.problem_id = p.id
            JOIN sessions s ON r.session_id = s.id
            WHERE s.current_phase = 'completed'
            AND r.phase = 'final_check'
            AND r.ai_predicted_confidence IS NOT NULL
            AND r.ai_predicted_challenge IS NOT NULL
            ORDER BY r.created_at ASC
        """
        
        problems_query = """
            SELECT id, knowledge_component, description_main, description_sub, answer
            FROM problems ORDER BY id
        """
        
        sessions_df = pd.read_sql(sessions_query, engine)
        responses_df = pd.read_sql(responses_query, engine)
        problems_df = pd.read_sql(problems_query, engine)
        
        # UUIDを文字列に変換
        for col in ['id', 'user_id', 'session_id', 'problem_id']:
            if col in sessions_df.columns:
                sessions_df[col] = sessions_df[col].astype(str)
            if col in responses_df.columns:
                responses_df[col] = responses_df[col].astype(str)
            if col in problems_df.columns:
                problems_df[col] = problems_df[col].astype(str)
        
        return sessions_df, responses_df, problems_df
        
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return None, None, None


def prepare_analysis_data(responses_df, human_col, ai_col, value_range=None):
    """
    分析用データを準備する汎用関数
    
    Args:
        responses_df: 生のレスポンスデータ
        human_col: 人間の値のカラム名
        ai_col: AIの予測値のカラム名
        value_range: 値の範囲（タプル）。Noneの場合は自動計算
    
    Returns:
        分析用に加工されたDataFrame
    """
    df = responses_df.copy()
    
    # 分析対象のカラムを標準名に変換
    df['human_value'] = df[human_col]
    df['ai_value'] = df[ai_col]
    df['value_difference'] = (df['human_value'] - df['ai_value']).abs()
    
    # 全体平均
    global_mean = df['human_value'].mean()
    df['global_mean'] = global_mean
    
    # 問題別平均（Leave-One-Out方式）
    def calc_problem_loo_mean(row, data):
        same_problem = data[(data['problem_id'] == row['problem_id']) & (data['session_id'] != row['session_id'])]
        if len(same_problem) > 0:
            return same_problem['human_value'].mean()
        else:
            return data[data['session_id'] != row['session_id']]['human_value'].mean()
    
    df['problem_mean'] = df.apply(lambda row: calc_problem_loo_mean(row, df), axis=1)
    
    # ユーザー別平均（Leave-One-Out方式）
    def calc_user_loo_mean(row, data):
        other_problems = data[(data['session_id'] == row['session_id']) & (data['problem_id'] != row['problem_id'])]
        if len(other_problems) > 0:
            return other_problems['human_value'].mean()
        else:
            return data[data['session_id'] != row['session_id']]['human_value'].mean()
    
    df['user_mean'] = df.apply(lambda row: calc_user_loo_mean(row, df), axis=1)
    
    # 各ベースラインとの誤差
    df['llm_error'] = df['value_difference']
    df['global_error'] = (df['human_value'] - df['global_mean']).abs()
    df['problem_error'] = (df['human_value'] - df['problem_mean']).abs()
    df['user_error'] = (df['human_value'] - df['user_mean']).abs()
    
    return df


def calculate_metrics(actual, predicted):
    """各種メトリクスを計算"""
    if len(actual) < 2:
        return {'r': 0, 'p_value': 1, 'r2': 0, 'mae': 0, 'rmse': 0}
    
    # 定数配列の場合は相関係数を計算できない
    if np.std(actual) == 0 or np.std(predicted) == 0:
        r, p_value = 0, 1
    else:
        r, p_value = stats.pearsonr(actual, predicted)
    
    mae = np.abs(actual - predicted).mean()
    rmse = np.sqrt(((actual - predicted) ** 2).mean())
    
    return {
        'r': r,
        'p_value': p_value,
        'r2': r ** 2,
        'mae': mae,
        'rmse': rmse
    }
