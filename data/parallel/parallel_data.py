import os
import sys
import orjson
import argparse
from functools import partial
from pathlib import Path
import random
import yaml
import pandas as pd
import numpy as np
from tqdm.contrib.concurrent import process_map
from os.path import join as pj
import time

from mappings import ISO639_3_SCRIPT_TO_NAME as nllb_to_name
import gc
from datasets import disable_caching
disable_caching()

ISO_TO_NLLB = {
    "af": "afr_Latn", "afr": "afr_Latn",
    "am": "amh_Ethi", "amh": "amh_Ethi",
    "ar": "arb_Arab", "arb": "arb_Arab",
    "ary": "ary_Arab", "arz": "arz_Arab", "aeb": "aeb_Arab",
    "en": "eng_Latn", "eng": "eng_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn",
    "ha": "hau_Latn", "hau": "hau_Latn",
    "ig": "ibo_Latn", "ibo": "ibo_Latn",
    "rw": "kin_Latn", "kin": "kin_Latn",
    "mg": "plt_Latn", "plt": "plt_Latn",
    "ny": "nya_Latn", "nya": "nya_Latn",
    "om": "orm_Latn", "orm": "orm_Latn",
    "pt": "por_Latn", "por": "por_Latn",
    "sn": "sna_Latn", "sna": "sna_Latn",
    "so": "som_Latn", "som": "som_Latn",
    "st": "sot_Latn", "sot": "sot_Latn",
    "sw": "swa_Latn", "swa": "swa_Latn", "swh": "swa_Latn",
    "ti": "tir_Ethi", "tir": "tir_Ethi",
    "tn": "tsn_Latn", "tsn": "tsn_Latn",
    "xh": "xho_Latn", "xho": "xho_Latn",
    "yo": "yor_Latn", "yor": "yor_Latn",
    "zu": "zul_Latn", "zul": "zul_Latn",
}


def get_nllb_lang_code(lang):
    if "_" in lang:
        return lang
    return ISO_TO_NLLB.get(lang, lang)


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg

def apply_length_filter(df, src_col, tgt_col, min_char=0, max_char=10**9):
    mask = df[src_col].astype(str).str.len().between(min_char, max_char) & df[tgt_col].astype(str).str.len().between(min_char, max_char)
    return df[mask]

def apply_score_filter(df, score_col, strategy='percentile', threshold=0.7):
    if strategy == 'absolute':
        df = df[df[score_col] >= threshold]
    elif strategy == 'percentile':
        cutoff = np.percentile(df[score_col].values, threshold * 100)
        df = df[df[score_col] >= cutoff]
    elif strategy == 'peak_percentile':
        # histogram-based peak detection
        hist, bin_edges = np.histogram(df[score_col].values, bins=1000, range=(0.0, 1.0))
        peak_bin = np.argmax(hist)
        peak_value = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2
        # percentile cutoff around peak
        lower_bound = peak_value + (1.0 - threshold) * (1.0 - peak_value)
        df = df[df[score_col] >= lower_bound]
    else:
        # unknown strategy: do nothing
        raise ValueError(f"Unknown score filter strategy: {strategy}")
    return df

def apply_sampling(df, strategy='uniform', max_num_samples=100000, seed=42, min_num_samples=0):
    if len(df) > max_num_samples:
        if strategy == 'uniform':
            return df.sample(n=min(max_num_samples, len(df)), random_state=seed).reset_index(drop=True)
        else:
            raise ValueError(f"Unknown sampling strategy: {strategy}")

    # repeat until min_num_samples is reached
    if len(df) < min_num_samples // 2:
        repeat_times = min_num_samples // len(df)
        df = pd.concat([df] * repeat_times, ignore_index=True)
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        # df = df.sample(n=min_num_samples, random_state=seed).reset_index(drop=True)
    return df

def swap_bitext(df, src_col, tgt_col, prob=1.0, seed=42):
    rng = np.random.default_rng(seed)
    mask = rng.random(size=len(df)) < prob
    df.loc[mask, [src_col, tgt_col]] = df.loc[mask, [tgt_col, src_col]].values
    df["swapped"] = mask
    return df

def write_pt_json(df, src_col, tgt_col, source_code, source_lang, target_code, target_lang, out_path, templates):
    """
    Write pretraining format JSON with simple text field.
    Format: [{"text": "document"}, {"text": "document"}]
    """
    def build_text(content, target_content, source_code, source_lang, target_code, target_lang, templates):
        text = random.choice(templates[0]).format(
            source=source_lang,
            target=target_lang,
            source_code=source_code,
            target_code=target_code,
            source_content=content,
            target_content=target_content,
        )
        return text

    data = []
    swap_row = "swapped" in df.columns
    for _, row in df.iterrows():
        if pd.isna(row[src_col]) or pd.isna(row[tgt_col]) or str(row[src_col]).strip() == "" or str(row[tgt_col]).strip() == "":
            continue

        if swap_row and row["swapped"]:
            content = build_text(row[tgt_col], row[src_col], target_code, target_lang, source_code, source_lang, templates)
            # text_document = f"{text_prompt}\n{row[src_col]}"
        else:
            content = build_text(row[src_col], row[tgt_col], source_code, source_lang, target_code, target_lang, templates)
            # text_document = f"{text_prompt}\n{row[tgt_col]}"

        data.append({
            "text": content
        })

    # Handle large datasets by partitioning
    if len(data) > 2000000:
        for part_id, start in enumerate(range(0, len(data), 2000000)):
            part_data = data[start:start+2000000]
            part_out_path = out_path.replace('.json', f'-part-{part_id+1:03d}.json')
            with open(part_out_path, 'wb') as f:
                f.write(orjson.dumps(part_data))
            print(f"Wrote {len(part_data)} entries in {part_id+1} parts to {part_out_path}")
            gc.collect()
    else:
        with open(out_path, 'wb') as f:
            f.write(orjson.dumps(data))
        print(f"Wrote {len(data)} entries to {out_path}")
    gc.collect()

def write_sft_json(df, src_col, tgt_col, source_code, source_lang, target_code, target_lang, out_path, templates):
    # https://llamafactory.readthedocs.io/zh-cn/latest/getting_started/data_preparation.html#id4
    # into sharegpt format
    def build_input(content, source_code, source_lang, target_code, target_lang, templates):
        sys_tmp, inp_tmp = random.choice(templates[0]), random.choice(templates[1])
        # inst = " " + t.split("{{source}} sentence: {{content}}")[0].strip() # TODO
        sys = sys_tmp.format(
            source=source_lang,
            target=target_lang,
            source_content=content,
            source_code=source_code
        )
        inp = inp_tmp.format(
            source=source_lang,
            target=target_lang,
            source_content=content,
            source_code=source_code,
            target_code=target_code
        )
        return sys, inp

    data = []
    swap_row = "swapped" in df.columns
    for _, row in df.iterrows():
        if pd.isna(row[src_col]) or pd.isna(row[tgt_col]) or str(row[src_col]).strip() == "" or str(row[tgt_col]).strip() == "":
            continue

        if swap_row and row["swapped"]:
            (inst, i) = build_input(row[tgt_col], target_code, target_lang, source_code, source_lang, templates)
            i = row[tgt_col]
            o = row[src_col]
        else:
            (inst, i) = build_input(row[src_col], source_code, source_lang, target_code, target_lang, templates)
            i = row[src_col]
            o = row[tgt_col]

        data.append({
            # "instruction": inst,
            # "input": i,
            # "output": o
            # "source_language": source_code,
            # "target_language": target_code,
            # "source_language_name": source_lang,
            # "target_language_name": target_lang,
            "conversations": [
                {
                    "from": "human", "value": i
                },
                {
                    "from": "gpt", "value": o # add eos token
                }
            ],
            "system": inst,
        })

    # if data size exceeds 2M, sample, partiion into multiple files
    if len(data) > 2000000:
        for part_id, start in enumerate(range(0, len(data), 2000000)):
            part_data = data[start:start+2000000]
            part_out_path = out_path.replace('.json', f'-part-{part_id+1:03d}.json')
            with open(part_out_path, 'wb') as f:
                f.write(orjson.dumps(part_data))
            print(f"Wrote {len(part_data)} entries in {part_id+1} parts to {part_out_path}")
            gc.collect()
    else:
        with open(out_path, 'wb') as f:
            f.write(orjson.dumps(data))
        print(f"Wrote {len(data)} entries to {out_path}")
    gc.collect()

def validate_output_exists(out_path, file_base, out_folder, cfg):
    """
    Validate if output files exist and match expected sample count from stats.
    Returns True if valid output exists, False otherwise.
    """
    # Check if main file or first part exists
    main_exists = os.path.exists(out_path)
    part1_exists = os.path.exists(out_path.replace('.json', '-part-001.json'))
    
    if not main_exists and not part1_exists:
        return False
    
    # Read the latest stats for this file_base
    stats_path = pj(out_folder, f"../{cfg.get('version', 'test')}_stats.jsonl")
    if not os.path.exists(stats_path):
        # No stats file, can't validate
        return False
    
    # Find the latest entry for this file_base in stats
    latest_stat = None
    try:
        with open(stats_path, 'r', encoding='utf-8') as f:
            for line in f:
                stat = orjson.loads(line.strip())
                src_lang = stat.get('src_lang', '')
                tgt_lang = stat.get('tgt_lang', '')
                if f"{src_lang}-{tgt_lang}" == file_base:
                    latest_stat = stat
    except Exception as e:
        print(f"Error reading stats for {file_base}: {e}")
        return False
    
    if not latest_stat:
        # No stats found for this file
        return False
    
    expected_samples = latest_stat.get('final_samples', 0)
    if expected_samples == 0:
        return False
    
    # Count actual samples in output files
    actual_samples = 0
    
    try:
        if main_exists:
            # Count samples in main file
            with open(out_path, 'rb') as f:
                data = orjson.loads(f.read())
                actual_samples = len(data)
        else:
            # Count samples across all parts
            part_num = 1
            while True:
                part_path = out_path.replace('.json', f'-part-{part_num:03d}.json')
                if not os.path.exists(part_path):
                    break
                with open(part_path, 'rb') as f:
                    data = orjson.loads(f.read())
                    actual_samples += len(data)
                part_num += 1
    except Exception as e:
        print(f"Error counting samples in output files for {file_base}: {e}")
        return False
    
    # Validate: actual samples should match expected samples
    if actual_samples == expected_samples:
        return True

    print(f"Mismatch for {file_base}: expected {expected_samples} samples, found {actual_samples} samples")
    return False

def pipeline(file_base, cfg, orig_dir, score_dir):
    # read file
    orig_path = pj(orig_dir, file_base + '.parquet')
    score_path = pj(score_dir, file_base + '.parquet')
    if not os.path.exists(orig_path) or not os.path.exists(score_path):
        return f"skip:{file_base}"

    out_cfg = cfg["pipeline"]["output"]

    out_folder_templ = out_cfg.get('folder', './')
    out_folder = out_folder_templ.replace(
        "{{config_name}}", cfg.get('version', 'test'))
    os.makedirs(out_folder, exist_ok=True)
    
    src_short, tgt_short = file_base.split('-')
    nllb_src = get_nllb_lang_code(src_short)
    nllb_tgt = get_nllb_lang_code(tgt_short)
    nllb_src_name = nllb_to_name[nllb_src]
    nllb_tgt_name = nllb_to_name[nllb_tgt]

    prefix = out_cfg.get('prefix', '{{lang1}}-{{lang2}}.json').replace("{{lang1}}", nllb_src).replace("{{lang2}}", nllb_tgt)

    out_path = pj(out_folder, prefix)

    # Validate if output already exists and matches expected samples
    if validate_output_exists(out_path, file_base, out_folder, cfg):
        return f"exists:{file_base}"

    stats = {
        "nllb_src": nllb_src,
        "nllb_tgt": nllb_tgt,
        "nllb_src_name": nllb_src_name,
        "nllb_tgt_name": nllb_tgt_name,
    }

    # load original and score parquet (keep only required columns)
    df_orig = pd.read_parquet(orig_path, memory_map=True)
    df_score = pd.read_parquet(score_path, memory_map=True) # no header
    df_score.columns = ["comet-score"]
    assert len(df_orig) == len(df_score), f"Length mismatch: {len(df_orig)} vs {len(df_score)} in {file_base}"
    df = pd.concat([df_orig, df_score], axis=1)

    stats.update({
        "src_lang": file_base.split('-')[0],
        "tgt_lang": file_base.split('-')[1],
        "original_samples": len(df),
    })

    # df = invidual_process_pipeline(cfg, df).to_parquet()
    filters = cfg.get('pipeline', {}).get('single', [])
    for i, f in enumerate(filters, 1): # keep order
        if df.empty:
            break
        if 'FilterbyLength' == f['name']:
            df = apply_length_filter(df, 'sentence1', 'sentence2', **f['kargs'])
        elif 'FilterbyScore' == f['name']:
            # special: extra_file may point to folder; we already used score file
            df = apply_score_filter(df, 'comet-score', **f['kargs'])
        elif 'Sampling' == f['name']:
            # postpone sampling until we've filtered everything
            df = apply_sampling(df, **f['kargs'])
        elif 'SwapBitext' == f['name']:
            df = swap_bitext(df, 'sentence1', 'sentence2', **f['kargs'])
        else:
            raise ValueError(f"Unknown filter step: {f}")
        stats[f"step_{i}_{f['name']}"] = len(df)
    stats.update({
        "final_samples": len(df),
    })

    # metadata = {
    #     'source_language': nllb_src,
    #     'target_language': nllb_tgt,
    #     'tmp_files': file_base
    # }
    # return df

    # Read pipelines
        # Combine step
        # combine_steps = cfg.get('pipeline', {}).get('combine', [])
        # default behavior: write original direction to file
        # Determine languages: file_base is like src-tgt (using short/iso codes); map to nllb if possible
    print(f"Processing {file_base}: {len(df)} entries after filtering")

    sys_templates = cfg.get('task-templates', [])
    assert sys_templates, "No templates provided for output generation"
    inp_templates = cfg.get('content-templates', ["{{source_content}}\n"])
    assert inp_templates, "No content templates provided for output generation"
    
    # Determine task type (default to 'sft' for backward compatibility)
    task_type = cfg.get('task', 'sft')
    
    with open(pj(out_folder, f"../{cfg.get('version', 'test')}_stats.jsonl"), 'a', encoding='utf-8') as f:
        f.write(orjson.dumps(stats).decode('utf-8') + '\n')

    if df.empty:
        print(f"No data to write for {file_base}, skipping output.")
        return f"nodata:{file_base}"

    if task_type == 'pt':
        write_pt_json(df, 'sentence1', 'sentence2', nllb_src, nllb_src_name, nllb_tgt, nllb_tgt_name, out_path, (sys_templates, inp_templates))
    else:  # default to sft
        write_sft_json(df, 'sentence1', 'sentence2', nllb_src, nllb_src_name, nllb_tgt, nllb_tgt_name, out_path, (sys_templates, inp_templates))
    
    # write_stats_path = out_path.replace('.json', '.stats.json')
    time.sleep(random.random())

    gc.collect()
    return f"ok:{file_base}"
    # except Exception as e:
    #     return f"err:{file_base}:{e}"

def find_pairs(orig_dir, score_dir):
    orig_files = [Path(orig_dir).joinpath(p).stem for p in os.listdir(orig_dir) if p.endswith('.parquet')]
    score_files = {Path(score_dir).joinpath(p).stem for p in os.listdir(score_dir) if p.endswith('.parquet')}
    missing_scores = sorted(set(orig_files) - score_files)
    for file_name in missing_scores:
        print(f"Warning: {file_name} in orig_dir but not in score_dir")
    return orig_files

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="parallel/parallel_nllb.yaml", help="YAML pipeline config")
    parser.add_argument("--orig_dir", required=True, help="Directory with original bitext parquet files.")
    parser.add_argument("--score_dir", required=True, help="Directory with matching COMET score parquet files.")
    parser.add_argument("--jobs", type=int, default=max(1, 8), help="Parallel jobs")
    args = parser.parse_args()

    cfg = load_config(args.config)

    pairs = find_pairs(args.orig_dir, args.score_dir)
    if not pairs:
        print("No matching pairs found between orig and score folders.")
        return
    print(f"Found {len(pairs)} matching language pairs.")
    
    # filter pairs by languages specified in config.languages if present
    allowed = set(cfg.get('languages', []))
    if allowed:
        filtered = []
        excluded = []
        for p in pairs:
            try:
                s, t = p.split('-')
            except:
                excluded.append(p)
                continue
            ns = get_nllb_lang_code(s)
            nt = get_nllb_lang_code(t)
            if ns in allowed and nt in allowed:
                filtered.append(p)
            else:
                excluded.append(p)

        pairs = filtered
        
        for pa in sorted(filtered):
            print(f"{pa}")
            
        for ex in sorted(excluded):
        #     # os.remove(pj(args.orig_dir, f"{ex}.parquet"))
        #     # os.remove(pj(args.score_dir, f"{ex}.parquet"))
            print(f"{ex}")

        for pa in sorted(pairs):
            print(f"{pa}")
            
    print(f"Processing {len(pairs)} pairs with {args.jobs} jobs...")
    # exit(0)
    # pairs = pairs[:10] # DEBUG

    if args.jobs <= 1:
        for p in pairs:
            res = pipeline(p, cfg, args.orig_dir, args.score_dir)
            print(res)
    else:
        worker = partial(pipeline, cfg=cfg, orig_dir=args.orig_dir, score_dir=args.score_dir)
        for res in process_map(worker, pairs, max_workers=args.jobs, chunksize=8, desc="Processing pairs"):
            print(res)

    """
    {
        "mt-nllb-v1": {
            "file_name": "mt-nllb-v1/",
            "formatting": "sharegpt",
            "columns": {
            "messages": "conversations",
            "system": "system"
            }
        },
        "mt-nllb-v2": {
            "file_name": "mt-nllb-v2/",
            "columns": {
            "messages": "conversations",
            "system": "system"
            }
        }
    }
    """
    version = cfg["version"]
    
    ds_info_path = pj(cfg["pipeline"]["output"]["folder"].replace("/{{config_name}}", ""), "dataset_info.json")
    # Read or create dataset_info.json, update with current version info
    if os.path.exists(ds_info_path):
        with open(ds_info_path, 'rb') as f:
            dataset_info = orjson.loads(f.read())
    else:
        dataset_info = {}

    # Determine format based on task type
    task_type = cfg.get('task', 'sft')
    
    if task_type == 'pt':
        dataset_info[version] = {
            "file_name": version + "/",
            "columns": {
                "prompt": "text"
            }
        }
    else:  # sft format
        dataset_info[version] = {
            "file_name": version + "/",
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            }
        }

    with open(ds_info_path, 'wb') as f:
        f.write(orjson.dumps(dataset_info, option=orjson.OPT_INDENT_2))

if __name__ == "__main__":
    main()
