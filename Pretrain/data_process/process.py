import pandas as pd
from typing import List, Set, Tuple

def merge_datasets(dataframes: List[pd.DataFrame], 
                   prot_seq_col: str = 'prot_seq',
                   pep_seq_col: str = 'pep_seq') -> pd.DataFrame:
    """
    合并若干个数据集
    
    参数:
        dataframes: 要合并的DataFrame列表
        prot_seq_col: 蛋白质序列列名（默认'prot_seq'）
        pep_seq_col: 肽链序列列名（默认'pep_seq'）
    
    返回:
        合并后的DataFrame
    """
    if not dataframes:
        return pd.DataFrame()
    
    unified_dfs = []
    for df in dataframes:
        df_copy = df.copy()
        if 'Prot_seq' in df_copy.columns:
            df_copy = df_copy.rename(columns={'Prot_seq': prot_seq_col, 'Pep_seq': pep_seq_col})
        unified_dfs.append(df_copy)
    
    merged = pd.concat(unified_dfs, ignore_index=True)
    return merged

def deduplicate_dataset(df: pd.DataFrame,
                       prot_seq_col: str = 'prot_seq',
                       pep_seq_col: str = 'pep_seq',
                       keep: str = 'first') -> pd.DataFrame:
    """
    数据集内部去重（按蛋白质和肽链完全一致判断重复）
    
    参数:
        df: 要去重的DataFrame
        prot_seq_col: 蛋白质序列列名（默认'prot_seq'）
        pep_seq_col: 肽链序列列名（默认'pep_seq'）
        keep: 保留策略，'first'保留第一个，'last'保留最后一个（默认'first'）
    
    返回:
        去重后的DataFrame
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    if 'Prot_seq' in df_copy.columns:
        df_copy = df_copy.rename(columns={'Prot_seq': prot_seq_col, 'Pep_seq': pep_seq_col})
    
    deduplicated = df_copy.drop_duplicates(subset=[prot_seq_col, pep_seq_col], keep=keep)
    return deduplicated

def remove_overlap(df: pd.DataFrame,
                  exclude_datasets: List[pd.DataFrame],
                  prot_seq_col: str = 'prot_seq',
                  pep_seq_col: str = 'pep_seq') -> pd.DataFrame:
    """
    在给定的数据集中，去除另一个列表中指出的数据集中出现过的数据
    （按蛋白质和肽链完全一致判断重复）
    
    参数:
        df: 要处理的主数据集
        exclude_datasets: 要排除的数据集列表（这些数据集中的序列组合会被从主数据集中移除）
        prot_seq_col: 蛋白质序列列名（默认'prot_seq'）
        pep_seq_col: 肽链序列列名（默认'pep_seq'）
    
    返回:
        去除重复后的DataFrame
    """
    if df.empty:
        return df
    
    df_copy = df.copy()
    if 'Prot_seq' in df_copy.columns:
        df_copy = df_copy.rename(columns={'Prot_seq': prot_seq_col, 'Pep_seq': pep_seq_col})
    
    exclude_sequences: Set[Tuple[str, str]] = set()
    
    for exclude_df in exclude_datasets:
        if exclude_df.empty:
            continue
        
        exclude_df_copy = exclude_df.copy()
        if 'Prot_seq' in exclude_df_copy.columns:
            exclude_df_copy = exclude_df_copy.rename(columns={'Prot_seq': prot_seq_col, 'Pep_seq': pep_seq_col})
        
        sequences = set(zip(exclude_df_copy[prot_seq_col], exclude_df_copy[pep_seq_col]))
        exclude_sequences.update(sequences)
    
    df_copy['_seq_combo'] = list(zip(df_copy[prot_seq_col], df_copy[pep_seq_col]))
    filtered = df_copy[~df_copy['_seq_combo'].isin(exclude_sequences)].drop('_seq_combo', axis=1)
    
    return filtered

