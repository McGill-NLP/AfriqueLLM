import os
import zipfile
import pandas as pd
import argparse

# Import dezip for faster zip processing
try:
    from dezip import _ZipDecrypter_C
    setattr(zipfile, '_ZipDecrypter', _ZipDecrypter_C)
    # print("Using optimized dezip for faster zip processing")
except ImportError:
    pass
    # print("Warning: dezip not available, using standard zipfile")

ISO_TO_NLLB = {
    "af": "afr_Latn", "afr": "afr_Latn",
    "am": "amh_Ethi", "amh": "amh_Ethi",
    "ar": "arb_Arab", "arb": "arb_Arab",
    "en": "eng_Latn", "eng": "eng_Latn",
    "fr": "fra_Latn", "fra": "fra_Latn",
    "ha": "hau_Latn", "hau": "hau_Latn",
    "ig": "ibo_Latn", "ibo": "ibo_Latn",
    "pt": "por_Latn", "por": "por_Latn",
    "sw": "swa_Latn", "swa": "swa_Latn", "swh": "swa_Latn",
    "xh": "xho_Latn", "xho": "xho_Latn",
    "yo": "yor_Latn", "yor": "yor_Latn",
    "zu": "zul_Latn", "zul": "zul_Latn",
}


def get_nllb_lang_code(lang):
    """Map language codes to NLLB format with special cases."""
    special = {
        "az_Arab": "azb_Arab", "ace": "ace_Arab", "kr_Arab": "knc_Arab",
        "kr_Latn": "knc_Latn", "ks_Arab": "kas_Arab", "ks_Deva": "kas_Deva",
        "ku_Arab": "ckb_Arab", "ku_Latn": "kmr_Latn", "zh_TW": "zh_Hant",
    }
    
    if lang in special:
        return special[lang]

    if "_" in lang:
        return lang
    return ISO_TO_NLLB.get(lang, lang)

def process_single_zip(zip_file_path, output_path):
    """Process a single NLLB zip file and save as parquet."""
    file_name = os.path.basename(zip_file_path)
    os.makedirs(output_path, exist_ok=True)

    try:
        # Extract language codes from filename
        src_lang, tgt_lang = file_name.split('_moses_')[1].replace(".txt.zip", "").split('-')
        subset_name = f"{src_lang}-{tgt_lang}"
        
        # Map to NLLB codes
        # nllb_src = get_nllb_lang_code(src_lang)
        # nllb_tgt = get_nllb_lang_code(tgt_lang)
        # Not Map
        nllb_src, nllb_tgt = src_lang, tgt_lang

        save_path = os.path.join(output_path, f"{nllb_src}-{nllb_tgt}.parquet")
        try:
            if os.path.exists(save_path) and pd.read_parquet(save_path, columns=['score']).shape[0] > 0:
                print(f"Skipping {file_name} - already exists")
                return
        except:
            pass

        # File names inside zip
        src_file = f"NLLB.{subset_name}.{src_lang}"
        tgt_file = f"NLLB.{subset_name}.{tgt_lang}"
        scores_file = f"NLLB.{subset_name}.scores"
        
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            # Read files
            with zf.open(src_file) as f:
                src_lines = [line.decode('utf-8').strip() for line in f]
            with zf.open(tgt_file) as f:
                tgt_lines = [line.decode('utf-8').strip() for line in f]
            with zf.open(scores_file) as f:
                scores = [float(line.decode('utf-8').strip()) for line in f]
        
        # Check lengths match
        if not (len(src_lines) == len(tgt_lines) == len(scores)):
            print(f"Error: Line count mismatch in {file_name}")
            return
        
        # Save as parquet
        # os.makedirs(output_path, exist_ok=True)
        df = pd.DataFrame({
            'sentence1': src_lines,
            'sentence2': tgt_lines,
            'score': scores
        })
        df.to_parquet(save_path, index=False)
        print(f"Processed {file_name} -> {os.path.basename(save_path)} ({len(df)} rows)")
        
    except Exception as e:
        print(f"Error processing {file_name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_file", help="Path to zip file to process")
    parser.add_argument("--output", default="data/parallel/parquet", help="Output directory")
    
    args = parser.parse_args()
    process_single_zip(args.zip_file, args.output)

if __name__ == "__main__":
    main()
