#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
全方位並行分子特徵生成腳本 (v7)(chingyuwang)

此腳本處理化學化合物檔案（SDF, MOL）或 SMILES 字串，以生成多種類型的分子特徵。
它整合了傳統的指紋計算以及基於 RDKit 的進階特徵工程。

此版本的主要升級：
- **極致效能**: 所有進階特徵計算均為內部實現，避免外部呼叫，速度極快。
- **功能模組化**: 提供獨立的 '--ac', '--metal', '--mw', '--stereo' 旗標，允許使用者按需組合計算。
- **分子量自訂精度**: --mw 旗標允許使用者自訂分子量的小數位數 (預設2，最多4)。
- **立體化學分析**: --stereo 旗標可計算 7 項關鍵的立體化學描述符。
- **彈性 ID 處理**: 針對 SMILES 輸入，可直接使用 SMILES 字串作為識別碼。
- **合併輸出**: 新增 --merge 旗標，可將所有選定的特徵合併成一個檔案。
- **易用性**: 新增檔案內參數設定區，讓使用者能更方便地調整執行參數。
- **ECFP 更新**: 遵循 RDKit 建議，使用新的 MorganGenerator 來生成 ECFP，消除棄用警告。
- **Rule of Five**: 新增 Rule of Five (Ro5) 特徵。
"""

import os
import glob
import argparse
import subprocess
import tempfile
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import threading
import time
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import Crippen, Lipinski
from tqdm import tqdm

# ==============================================================================
# --- 參數設定 --- 
# ==============================================================================
USER_CONFIG = {
    # 主要路徑設定
    'input_path': './test_1014.txt',
    'output_path': './output_dir_1014',
    'input_format': 'smiles',  # 可選: 'sdf', 'mol', 'smiles'

    'cores': 16,     # 並行處理的核心數
    'base_generate_path': '/data/chingyuwang/Moieties_compound_feature/generate',

    # 功能開關 (True = 開啟, False = 關閉)
    'checkmol': True,
    'pubchem': True,
    'ring': True,
    'ecfp': True,
    'ac': True,
    'metal': False,
    'stereo': True,
    'ro5': False,
    'mw': None,          # 分子量，None 為關閉，可填 0, 1, 2, 3, 4 指定小數位數
    'merge': True        # 合併所有選定的特徵到一個檔案
}

# ==============================================================================
# --- 1. 路徑設定  ---
# ==============================================================================
CHECKMOL_EXEC = os.path.join(USER_CONFIG['base_generate_path'], 'checkmol')
MATCHMOL_EXEC = os.path.join(USER_CONFIG['base_generate_path'], 'matchmol')
PUBCHEM_MOIETIES_PATH = os.path.join(USER_CONFIG['base_generate_path'], 'PCFP_168/*.mol')
RING_MOIETIES_PATH = os.path.join(USER_CONFIG['base_generate_path'], 'RD_147/*.mol')

# ==============================================================================
# --- 2. 常數定義 ---
# ==============================================================================
FINGERPRINT_SIZES = {
    'checkmol': 204,
    'pubchem': 168,
    'ring': 147,
    'ecfp': 512
}
log_lock = threading.Lock()

# ==============================================================================
# --- 3. 命令列參數解析 ---
# ==============================================================================
def get_args():
    # 從 USER_CONFIG 載入預設值
    parser = argparse.ArgumentParser(
        description="高效能並行生成多維度分子特徵。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 從 USER_CONFIG 讀取預設值
    parser.add_argument('-p', '--input_path', type=str, default=USER_CONFIG['input_path'], help='輸入檔案的目錄路徑或 SMILES 檔案路徑。')
    parser.add_argument('-o', '--output_path', type=str, default=USER_CONFIG['output_path'], help='儲存輸出 TSV 檔案的目錄。')
    parser.add_argument('-f', '--input_format', type=str, default=USER_CONFIG['input_format'], choices=['sdf', 'mol', 'smiles'], help='輸入檔案的格式。')
    parser.add_argument('-c', '--cores', type=int, default=USER_CONFIG['cores'], help='並行處理的核心數。')

    # 為所有標準指紋添加參數旗標
    for fp_type in FINGERPRINT_SIZES.keys():
        default_val = USER_CONFIG.get(fp_type, False)
        if default_val:
            parser.add_argument(f'--{fp_type}', action='store_true', default=default_val, help=f'產生 {fp_type.upper()} 指紋。')
        else:
            parser.add_argument(f'--{fp_type}', action='store_true', help=f'產生 {fp_type.upper()} 指紋。')

    # --- 進階特徵的獨立旗標 ---
    parser.add_argument('--ac', action='store_true', default=USER_CONFIG.get('ac', False), help='產生 AC (Atomic Composition) 10 項特徵。')
    parser.add_argument('--metal', action='store_true', default=USER_CONFIG.get('metal', False), help='產生金屬特徵集 (分類, 離子性, 原子量)。')
    parser.add_argument('--stereo', action='store_true', default=USER_CONFIG.get('stereo', False), help='產生 7 項立體化學特徵。')
    parser.add_argument('--ro5', action='store_true', default=USER_CONFIG.get('ro5', False), help='產生 Rule of Five (Ro5) 5 項特徵。')

    # 分子量參數的特殊處理
    mw_default = USER_CONFIG.get('mw')
    if mw_default is not None and len(sys.argv) > 1:
        # 如果命令列中有指定 --mw，但沒有指定值，則用 const=2   
        # 如果命令列中沒有指定 --mw，則用 USER_CONFIG 的預設值
        parser.add_argument(
            '--mw', nargs='?', const=2, default=mw_default, type=int, choices=[0, 1, 2, 3, 4],
            help='計算總分子量。可選填小數位數 (0-4)，預設為 2。'
        )
    else:
        parser.add_argument(
            '--mw', nargs='?', const=2, default=mw_default, type=int, choices=[0, 1, 2, 3, 4],
            help='計算總分子量。可選填小數位數 (0-4)，預設為 2。'
        )

    parser.add_argument('--merge', action='store_true', default=USER_CONFIG.get('merge', False), help='產生一個合併所有指定特徵的 TSV 檔案。')

    args = parser.parse_args()
    for key in ['checkmol', 'pubchem', 'ring', 'ecfp', 'ac', 'metal', 'stereo', 'merge']:
        if key in USER_CONFIG and USER_CONFIG[key] and not getattr(args, key):
            # 如果 USER_CONFIG 中為 True 且命令列沒有明確覆蓋，則設定為 True
            setattr(args, key, True)

    return args

# ==============================================================================
# --- 4. 內建核心特徵生成函數 ---
# ==============================================================================

# --- RDKit 指紋 ---
def generate_ecfp(mol):
    if mol is None: return None, "RDKit molecule is None"
    try:
        generator = GetMorganGenerator(radius=2, fpSize=512)
        fp = generator.GetFingerprint(mol)

        ecfp_arr = np.zeros((FINGERPRINT_SIZES['ecfp'],), dtype=int)
        DataStructs.ConvertToNumpyArray(fp, ecfp_arr)   # Bit Array -> 位元字串 NumPy array
        return ecfp_arr, None 
    except Exception as e: 
        return None, str(e)

# --- AC 特徵 --- 
def calculate_ac_features(mol):
    if mol is None: return None, "RDKit molecule is None for AC"
    try:
        c_r, c_o, n_r, n_o, o_r, o_o, p, s, x = 0, 0, 0, 0, 0, 0, 0, 0, 0
        halogens = {'F', 'Cl', 'Br', 'I'}
        Chem.GetSymmSSSR(mol)
        for atom in mol.GetAtoms():
            sym, is_ring = atom.GetSymbol(), atom.IsInRing()
            if sym == 'C':
                if is_ring: c_r += 1
                else: c_o += 1
            elif sym == 'N':
                if is_ring: n_r += 1
                else: n_o += 1
            elif sym == 'O':
                if is_ring: o_r += 1
                else: o_o += 1
            elif sym == 'P': p += 1
            elif sym == 'S': s += 1
            elif sym in halogens: x += 1
        return np.array([c_r, c_o, n_r, n_o, o_r, o_o, p, s, x, mol.GetRingInfo().NumRings()], dtype=int), None
    except Exception as e: return None, f"AC calculation failed: {e}"

# --- 金屬特徵 ---
def get_metal_categories():
    return [
        ("Toxic/Non_Essential_Metal ", {'Pb', 'Cd', 'Hg', 'Al', 'Tl', 'Be', 'Sb', 'Fr', 'Th', 'Ga', 'Ge', 'Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Ru', 'Rh', 'Pd', 'Ag', 'Au', 'Hf', 'Ta', 'Re', 'Os', 'Ir', 'Pt', 'La', 'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Dy', 'Er', 'Yb', 'Lu', 'Li', 'Rb', 'Cs', 'Sr', 'Ba'}),
        # 非必需/有毒或替代元素
        ("Potentially_Essential_Metal", {'Cr', 'Ni', 'V', 'W', 'B'}),
        # 潛在必需元素
        ("Trace_Metal", {'Fe', 'Zn', 'Cu', 'Mn', 'Co', 'Mo'}),
        # 微量元素
        ("Bulk_Metal", {'Na', 'K', 'Mg', 'Ca'})
        # 大量元素
    ]

def calculate_metal_features(mol):
    if mol is None: return None, "RDKit molecule is None for Metal analysis"
    try:
        # 初始化 features 字典，用於存儲各類別的計數
        features = {'is_ionic': 0}
        metal_categories = get_metal_categories()

        # 針對每種類別初始化計數
        for cat_name, _ in metal_categories:
            features[cat_name] = 0

        # 檢查離子性
        fragments = Chem.GetMolFrags(mol, asMols=True)
        if len(fragments) > 1:
            if any(Chem.GetFormalCharge(f) > 0 for f in fragments) and any(Chem.GetFormalCharge(f) < 0 for f in fragments):
                features['is_ionic'] = 1
        else:
            all_metals = set().union(*[cat[1] for cat in metal_categories])
            if not {atom.GetSymbol() for atom in mol.GetAtoms()}.isdisjoint(all_metals) and any(atom.GetFormalCharge() != 0 for atom in mol.GetAtoms()):
                features['is_ionic'] = 1

        # 檢查每種金屬類別的數量
        for atom in mol.GetAtoms():
            symbol = atom.GetSymbol()
            for cat_name, symbols in metal_categories:
                if symbol in symbols:
                    features[cat_name] += 1
                    break

        # 將結果轉換為 NumPy 陣列
        # 確保順序與 write_feature_files 中的欄位名稱一致
        ordered_keys = ['is_ionic'] + [cat[0] for cat in metal_categories]
        result_array = np.array([features[key] for key in ordered_keys], dtype=int)

        return result_array, None
    except Exception as e:
        return None, f"Metal calculation failed: {e}"


# --- 立體化學 & 分子量特徵 ---
def calculate_stereo_features(smiles):
    """
    基於 SMILES 字串計算立體化學特徵
    """
    if smiles is None:
        return None, "SMILES string is None for Stereo"
    try:
        import re

        # 1) '@' 的總字元數
        count_at = len(re.findall(r'@', smiles))

        # 2) 取得手性 token 序列（只保留順序中的 '@@' 與 '@')
        tokens = [m.group(0) for m in re.finditer(r'@@|@', smiles)]

        # 3) 第一個手性 token 的型別（互斥）
        is_first_at     = 1 if tokens and tokens[0] == '@'  else 0
        is_first_at_at  = 1 if tokens and tokens[0] == '@@' else 0 

        # 4) 工具：計算某 target（'@' 或 '@@'）在 token 序列的最長連續段長；<2 → 0 
        def max_streak(toks, target):
            best = cur = 0
            for t in toks:
                if t == target:
                    cur += 1
                    if cur > best:
                        best = cur
                else:
                    cur = 0
            return best if best >= 2 else 0

        # 5) 依論文定義計算最大連續段（不會出現 1
        max_con_at    = max_streak(tokens, '@')    # Stereo_Max_Cont_@   
        max_con_at_at = max_streak(tokens, '@@')   # Stereo_Max_Cont_@@  

        # 6) 斜線次數（排除 l/ 或 /l 的規則其實可省略；為了相容性保留）  
        count_forward_slash  = len(re.findall(r'(?<!l)\/(?![l])', smiles))   
        count_backward_slash = len(re.findall(r'(?<!l)\\(?![l])', smiles))   
        features = [
            int(count_at),            # Stereo_Count_@                
            int(max_con_at),          # Stereo_Max_Cont_@             
            int(max_con_at_at),       # Stereo_Max_Cont_@@            
            int(is_first_at),         # Stereo_Is_First_@             
            int(is_first_at_at),      # Stereo_Is_First_@@            
            int(count_forward_slash), # Stereo_Count_/                
            int(count_backward_slash) # Stereo_Count_\                
        ]
        return np.array(features, dtype=int), None

    except Exception as e:
        return None, f"Stereo calculation failed: {e}"

# --- Rule of Five (Ro5) 特徵 --- 
def calculate_ro5_features(mol):
    """
    計算 分子量、LogP、氫鍵供體、氫鍵受體，以及是否滿足 Ro5 準則。
    """
    if mol is None: return None, "RDKit molecule is None for Ro5"
    try:
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)      # 氫鍵供體
        hba = Lipinski.NumHAcceptors(mol)   # 氫鍵受體
        
        # 判斷是否滿足 Ro5 準則
        ro5_pass = 1 if (mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10) else 0

        features = [mw, logp, hbd, hba, ro5_pass]
        
        return np.array(features), None
    except Exception as e:
        return None, f"Ro5 calculation failed: {e}"

# --- 總分子量特徵 --- 
def calculate_mw_feature(mol, precision):
    if mol is None: return None, "RDKit molecule is None for MW"
    try:
        mw = Descriptors.MolWt(mol)
        if precision is None or not isinstance(precision, int):
            # 如果沒有給定或型態錯誤，則使用預設值 2
            return f"{mw:.2f}", None
        return f"{mw:.{precision}f}", None
    except Exception as e: return None, f"MW calculation failed: {e}"

# ==============================================================================
# --- 5. 外部工具相關函式 ---
# ==============================================================================
def run_subprocess(command):
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, encoding='utf-8', errors='ignore')
        return result.stdout, None
    except FileNotFoundError: return None, f"執行檔找不到: {command[0]}"
    except subprocess.CalledProcessError as e: return None, f"執行失敗，返回碼 {e.returncode}: {e.stderr.strip()}"
    except Exception as e: return None, str(e)

def generate_checkmol(mol_path):
    fp_size = FINGERPRINT_SIZES['checkmol']
    fp = np.zeros(fp_size, dtype=int)
    stdout, error = run_subprocess([CHECKMOL_EXEC, '-p', mol_path])
    if error: return None, error
    for line in stdout.strip().split('\n'):
        if ':' in line and '#' in line:
            try:
                parts = line.split(':');
                if int(parts[1].strip()) >= 1:
                    fea_num = int(parts[0].split('#')[1])
                    if 1 <= fea_num <= fp_size: fp[fea_num - 1] = 1
            except (ValueError, IndexError): continue
    return fp, None

def generate_moiety_fingerprint(mol_path, moiety_files, log_file_path, identifier):
    fp_size = len(moiety_files);
    if fp_size == 0: return None, "沒有提供 moiety 檔案"
    fp = np.zeros(fp_size, dtype=int)
    for i, moiety_path in enumerate(moiety_files):
        stdout, error = run_subprocess([MATCHMOL_EXEC, moiety_path, mol_path])
        if error: log_error_to_file(identifier, f"moiety_{i}", f"matchmol 失敗: {error}", log_file_path)
        elif 'T' in stdout: fp[i] = 1
    return fp, None

# ==============================================================================
# --- 6. 核心處理與錯誤記錄 ---
# ==============================================================================
def log_error_to_file(identifier, fp_type, reason, log_file_path):
    with log_lock:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{identifier}\t{fp_type}\t{reason}\n")

def process_item(job, args, moiety_lists, log_file_path):
    job_type, data, identifier = job
    results = {'identifier': identifier}
    
    mol = None
    smiles_str = None
    RDLogger.DisableLog('rdApp.warning')
    try:
        if job_type == 'smiles': 
            smiles_str = data
            mol = Chem.MolFromSmiles(data, sanitize=True)
        elif job_type == 'sdf_file_molecule':
            mol = data
            if mol:
                smiles_str = Chem.MolToSmiles(mol) 
            else:
                log_error_to_file(identifier, "RDKit_Loading", "SDF entry is invalid", log_file_path)
                mol = None
        else: 
            mol = Chem.MolFromMolFile(data, sanitize=True, removeHs=False)
            if mol:
                smiles_str = Chem.MolToSmiles(mol) 
    except Exception as e: 
        log_error_to_file(identifier, "RDKit_Loading", str(e), log_file_path)
    RDLogger.EnableLog('rdApp.warning')
    
    # --- 內部 RDKit 特徵計算 ---
    if mol is None:
        log_error_to_file(identifier, "RDKit_MOL_Object", "無法創建 RDKit 物件，跳過所有內部計算", log_file_path)
    else:
        if args.ecfp:
            fp, err = generate_ecfp(mol); results['ecfp'] = fp if not err else None
            if err: log_error_to_file(identifier, 'ecfp', err, log_file_path)
        if args.ac:
            fp, err = calculate_ac_features(mol); results['ac'] = fp if not err else None
            if err: log_error_to_file(identifier, 'ac', err, log_file_path)
        if args.metal:
            feats, err = calculate_metal_features(mol); results['metal'] = feats if not err else None
            if err: log_error_to_file(identifier, 'metal', err, log_file_path)
        if args.ro5:
            ro5_feats, err = calculate_ro5_features(mol); results['ro5'] = ro5_feats if not err else None
            if err: log_error_to_file(identifier, 'ro5', err, log_file_path)
        if args.mw is not None:
            val, err = calculate_mw_feature(mol, args.mw); results['mw'] = val if not err else None
            if err: log_error_to_file(identifier, 'mw', err, log_file_path)
    if args.stereo:
        if job_type == 'smiles':
            smiles_str = data
            feats, err = calculate_stereo_features(smiles_str)
            results['stereo'] = feats if not err else None
            if err: log_error_to_file(identifier, 'stereo', err, log_file_path)
        else:
            if mol:
                smiles_str = Chem.MolToSmiles(mol)
                feats, err = calculate_stereo_features(smiles_str)
                results['stereo'] = feats if not err else None
                if err: log_error_to_file(identifier, 'stereo', err, log_file_path)
            else:
                log_error_to_file(identifier, 'stereo', '無法從 RDKit 物件獲取 SMILES 字串', log_file_path)

    # --- 外部工具指紋計算 ---
    path_for_externals, tmp_created = None, False
    needs_external = args.checkmol or args.pubchem or args.ring
    if needs_external and mol: # 如果需要外部工具，且 mol 物件存在
        if job_type in ['mol_file', 'sdf_file_molecule']: path_for_externals = data
        elif job_type == 'smiles' and mol:
            fd, path_for_externals = tempfile.mkstemp(suffix='.mol', text=True)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as tmp:
                    tmp.write(Chem.MolToMolBlock(mol))
                tmp_created = True
            except Exception as e:
                log_error_to_file(identifier, "External_Tools_TempFile", f"無法寫入臨時檔案: {e}", log_file_path)
                path_for_externals = None
        
        if path_for_externals:
            if args.checkmol:
                fp, err = generate_checkmol(path_for_externals)
                if err: log_error_to_file(identifier, 'checkmol', err, log_file_path)
                else: results['checkmol'] = fp

            if args.pubchem:
                pubchem_moieties = moiety_lists.get('pubchem', []) # 確保列表存在
                fp, err = generate_moiety_fingerprint(path_for_externals, pubchem_moieties, log_file_path, identifier)
                if err: log_error_to_file(identifier, 'pubchem', err, log_file_path)
                else: results['pubchem'] = fp
            
            if args.ring:
                ring_moieties = moiety_lists.get('ring', [])
                fp, err = generate_moiety_fingerprint(path_for_externals, ring_moieties, log_file_path, identifier)
                if err: log_error_to_file(identifier, 'ring', err, log_file_path)
                else: results['ring'] = fp
        else:
            log_error_to_file(identifier, "External_Tools", "無有效 .mol 檔案路徑", log_file_path)

        if tmp_created:
            os.remove(path_for_externals)
        
    return results

# ==============================================================================
# --- 7. 結果寫入函數 ---
# ==============================================================================
def write_feature_files(results, output_dir, args):
    if not results: return
    all_ids = [r['identifier'] for r in results]

    # 建立一個空的 DataFrame 用於合併，並加入 identifier
    if args.merge:
        merged_df = pd.DataFrame({'identifier': all_ids})

    # 寫入傳統指紋
    for fp_type in FINGERPRINT_SIZES.keys():
        if getattr(args, fp_type):
            data = [r.get(fp_type) if r.get(fp_type) is not None else np.zeros(FINGERPRINT_SIZES[fp_type], dtype=int) for r in results]
            df = pd.DataFrame(data, columns=[f"#{fp_type.upper()}_{i+1}" for i in range(FINGERPRINT_SIZES[fp_type])])
            df.insert(0, 'identifier', all_ids)
            df.to_csv(os.path.join(output_dir, f"{fp_type}.tsv"), sep='\t', index=False)
            print(f"成功寫入 {len(df)} 筆 {fp_type.upper()} 指紋記錄。")
            if args.merge:
                merged_df = pd.merge(merged_df, df, on='identifier', how='left')

    # 寫入 AC 特徵
    if args.ac:
        ac_data = [r.get('ac', [0]*10) for r in results]
        ac_cols = ['AC_C_on_ring', 'AC_C_on_other', 'AC_N_on_ring', 'AC_N_on_other', 'AC_O_on_ring', 'AC_O_on_other', 'AC_P_total', 'AC_S_total', 'AC_X_total', 'AC_Ring_count']
        df_ac = pd.DataFrame(ac_data, columns=ac_cols); df_ac.insert(0, 'identifier', all_ids)
        df_ac.to_csv(os.path.join(output_dir, "ac_features.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(df_ac)} 筆 AC 特徵記錄。")
        if args.merge:
            merged_df = pd.merge(merged_df, df_ac, on='identifier', how='left')

    # 寫入金屬特徵
    if args.metal:
        metal_categories = get_metal_categories()
        # 創建新的欄位名稱列表
        metal_cols = ['Metal_Ionic_Compound'] + [f"Metal_{cat[0].replace('/', '_')}_Count" for cat in metal_categories]
        # 處理缺失值
        metal_data = [r.get('metal') if r.get('metal') is not None else [0] * len(metal_cols) for r in results]
        
        df_metal = pd.DataFrame(metal_data, columns=metal_cols);
        df_metal.insert(0, 'identifier', all_ids)
        df_metal.to_csv(os.path.join(output_dir, "metal_features.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(df_metal)} 筆金屬特徵記錄。")
        if args.merge:
            merged_df = pd.merge(merged_df, df_metal, on='identifier', how='left')

    # 寫入立體化學特徵
    if args.stereo:
        stereo_data = [r.get('stereo', [0]*7) for r in results]
        stereo_cols = [
            'Stereo_Count_@',
            'Stereo_Max_Cont_@',
            'Stereo_Max_Cont_@@',
            'Stereo_Is_First_@',
            'Stereo_Is_First_@@',
            'Stereo_Count_/',
            'Stereo_Count_\\'
        ]
        df_stereo = pd.DataFrame(stereo_data, columns=stereo_cols)
        df_stereo.insert(0, 'identifier', all_ids)
        df_stereo.to_csv(os.path.join(output_dir, "stereo_features.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(df_stereo)} 筆立體化學特徵記錄。")
        if args.merge:
            merged_df = pd.merge(merged_df, df_stereo, on='identifier', how='left')

    # 寫入 Ro5 特徵
    if args.ro5:
        # 處理缺失值，對於 Ro5 特徵，使用 NaN 填充更合適
        ro5_data_raw = [r.get('ro5') if r.get('ro5') is not None else [np.nan] * 5 for r in results]
        # 格式化浮點數為小數點後2位
        ro5_data = []
        for row in ro5_data_raw:
            if not np.isnan(row[0]):    # 確保資料有效才進行格式化
                formatted_row = [f"{row[0]:.2f}", f"{row[1]:.2f}", int(row[2]), int(row[3]), int(row[4])]
            else:
                formatted_row = ['N/A'] * 5     # 數據缺失時，用 NaN 或 'N/A' 填充
            ro5_data.append(formatted_row)
        ro5_cols = ['Ro5_MolWt', 'Ro5_LogP', 'Ro5_HBD', 'Ro5_HBA', 'Ro5_Pass']
        df_ro5 = pd.DataFrame(ro5_data, columns=ro5_cols)
        df_ro5.insert(0, 'identifier', all_ids)
        df_ro5.to_csv(os.path.join(output_dir, "ro5_features.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(df_ro5)} 筆 Ro5 特徵記錄。")
        if args.merge:
            merged_df = pd.merge(merged_df, df_ro5, on='identifier', how='left')

    # 寫入分子量
    if args.mw is not None:
        mw_data = [r.get('mw', 'N/A') for r in results]
        df_mw = pd.DataFrame({'identifier': all_ids, 'molecular_weight': mw_data})
        df_mw.to_csv(os.path.join(output_dir, "mw_features.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(df_mw)} 筆分子量記錄。")
        if args.merge:
            merged_df = pd.merge(merged_df, df_mw, on='identifier', how='left')

    # 最終寫入合併檔案
    if args.merge and not merged_df.empty:
        merged_df.to_csv(os.path.join(output_dir, "merge_feature.tsv"), sep='\t', index=False)
        print(f"成功寫入 {len(merged_df)} 筆合併特徵記錄到 merge_feature.tsv。")


# ==============================================================================
# --- 8. 主執行區塊 ---
# ==============================================================================
def main():
    args = get_args()
    os.makedirs(args.output_path, exist_ok=True)
    log_file_path = os.path.join(args.output_path, 'error_log.tsv')
    with open(log_file_path, 'w', encoding='utf-8') as f: f.write("Timestamp\tIdentifier\tFeatureType\tReason\n")

    moiety_lists = {
        'pubchem': sorted(glob.glob(PUBCHEM_MOIETIES_PATH)),
        'ring': sorted(glob.glob(RING_MOIETIES_PATH))
    }

    jobs = []
    if args.input_format == 'mol':
        all_files = sorted(glob.glob(os.path.join(args.input_path, f'*.{args.input_format}')))
        if not all_files:
            raise ValueError("No .mol files found in the specified folder.")
        for f_path in all_files:
            jobs.append(('mol_file', f_path, os.path.basename(f_path).rsplit('.', 1)[0]))

    elif args.input_format == 'sdf':
        all_files = sorted(glob.glob(os.path.join(args.input_path, f'*.{args.input_format}')))
        for f_path in all_files:
            suppl = Chem.SDMolSupplier(f_path)
            file_name = os.path.basename(f_path).rsplit('.', 1)[0]
            for i, mol in enumerate(suppl):
                # identifier 格式為 檔名_第幾筆
                identifier = f"{file_name}_{i+1}"
                jobs.append(('sdf_file_molecule', mol, identifier))

    elif args.input_format == 'smiles':
        with open(args.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    smiles = line.split()[0]    # 只取第一個空白前的字串
                    if smiles:
                        jobs.append(('smiles', smiles, smiles)) # 使用 smiles 本身作為 identifier

    
    if not jobs: print("找不到任何項目進行處理。"); return
    
    all_results = []
    task_processor = partial(process_item, args=args, moiety_lists=moiety_lists, log_file_path=log_file_path)
    with ProcessPoolExecutor(max_workers=args.cores) as executor:
        future_to_job = {executor.submit(task_processor, job): job for job in jobs}
        pbar = tqdm(as_completed(future_to_job), total=len(jobs), desc="正在處理分子")
        for future in pbar:
            try:
                result = future.result(); all_results.append(result)
            except Exception as e:
                log_error_to_file(future_to_job[future][2], "Process_Crash", str(e), log_file_path)

    # 照順序輸出
    identifier_order = [job[2] for job in jobs]
    results_dict = {r['identifier']: r for r in all_results}
    ordered_results = [results_dict.get(identifier, {'identifier': identifier}) for identifier in identifier_order]

    write_feature_files(ordered_results, args.output_path, args)
        
    print(f"\n處理完成。詳細錯誤請參見日誌檔案: {log_file_path}")

if __name__ == '__main__':
    main()