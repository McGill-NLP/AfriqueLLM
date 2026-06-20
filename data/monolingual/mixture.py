"""Build African monolingual pretraining mixtures for LlamaFactory."""

import argparse
import json
import logging
import os
import random
from multiprocessing import Pool
from functools import partial
from itertools import cycle

import numpy as np
import pandas as pd
from datasets import concatenate_datasets, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

DEFAULT_DATASET_ID = "<YourOrganization>/<YourRepo>"
DEFAULT_REVISION = "languageSubset_datasetSplit"

LANGUAGES = {
    'Tswana': 'tsn_Latn', 'West Central Oromo': 'gaz_Latn', 'Tunisian Arabic': 'aeb_Arab',
    'Tigrinya': 'tir_Ethi', 'Southern Sotho': 'sot_Latn', 'Nyanja': 'nya_Latn',
    'Yoruba': 'yor_Latn', 'Shona': 'sna_Latn', 'Xhosa': 'xho_Latn',
    'Plateau Malagasy': 'plt_Latn', 'Igbo': 'ibo_Latn', 'Zulu': 'zul_Latn',
    'Kinyarwanda': 'kin_Latn', 'Hausa': 'hau_Latn', 'Egyptian Arabic': 'arz_Arab',
    'Amharic': 'amh_Ethi', 'Arabic': 'arb_Arab', 'English': 'eng_Latn',
    'Portugese': 'por_Latn', 'French': 'fra_Latn', 'Somali': 'som_Latn',
    'Swahili': 'swh_Latn', 'Moroccan Arabic': 'ary_Arab', 'Afrikaans': 'afr_Latn'
}

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def update_dataset_info(output_path, dataset_name):
    info_path = os.path.join(output_path, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path, encoding="utf-8") as f:
            dataset_info = json.load(f)
    else:
        dataset_info = {}

    dataset_info[dataset_name] = {
        "file_name": f"{dataset_name}/",
        "columns": {"prompt": "text"},
    }
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(dataset_info, f, indent=2, ensure_ascii=False)


class DataMixture:
    def __init__(
        self,
        config,
        output_path,
        model_name='google/gemma-3-4b-pt',
        num_workers=4,
        format_type='llamafactory',
        token=None,
        fraction=1.0,
        subsets=None,
        seed=42,
        dataset_id=DEFAULT_DATASET_ID,
        revision=DEFAULT_REVISION,
        cache_dir=None,
    ):
        self.config = config
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.format_type = format_type
        self.model_name = model_name
        self.num_workers = int(num_workers)
        self.mixture_df = None
        self.token = token
        self.fraction = self._parse_fraction(fraction)
        self.fraction_text = fraction
        self.subsets = subsets or []
        self.seed = seed
        self.dataset_id = dataset_id
        self.revision = revision
        self.cache_dir = cache_dir
        set_seed(self.seed)
        
        assert os.path.exists(self.config), f"Input CSV not found: {self.config}"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    def _parse_fraction(self, fraction):
        """Parse fraction input - supports float, fraction string like '1/3', or int."""
        if isinstance(fraction, (int, float)):
            return float(fraction)
        elif isinstance(fraction, str) and '/' in fraction:
            parts = fraction.split('/')
            return float(parts[0]) / float(parts[1])
        else:
            return float(fraction)

    def _get_token_char_ratio(self, dataset, sample_size=1000000):
        """Calculates the ratio of tokens to characters from a sample of the dataset."""
        sample_text = ""
        for item in dataset:
            sample_text += item['text']
            if len(sample_text) >= sample_size:
                break
        
        sample_text = sample_text[:sample_size]
        if not sample_text:
            return 0

        num_chars = len(sample_text)
        num_tokens = len(self.tokenizer.encode(sample_text))
        ratio = num_tokens / num_chars if num_chars > 0 else 0
        print(f"Token/char ratio: {ratio:.4f}")
        return ratio

    def read_mixture_config(self):
        """Reads the data mixture CSV file."""
        self.mixture_df = pd.read_csv(self.config)
        self.mixture_df['token_count'] = self.mixture_df['token_count'].fillna(0)
        self.mixture_df['split'] = self.mixture_df['split'].fillna('')
        required_columns = ['subset', 'token_count', 'split']
        if not all(col in self.mixture_df.columns for col in required_columns):
            raise ValueError(f"Input CSV must contain the columns: {required_columns}")

    def _load_and_prepare_data(self, subset, splits):
        """Loads, shuffles, and prepares data from Hugging Face datasets."""
        logging.info("Loading %s from %s", subset, splits)

        all_datasets = []
        for split in splits:
            split = split.strip()
            if not split:
                continue
            dataset = load_dataset(
                self.dataset_id,
                name=subset,
                token=self.token,
                revision=self.revision,
                cache_dir=self.cache_dir,
                split=split,
                num_proc=1
            )
            all_datasets.append(dataset)
            logging.info("Loaded %s/%s", subset, split)

        if not all_datasets:
            return None
            
        if len(all_datasets) == 1:
            combined_dataset = all_datasets[0]
        else:
            combined_dataset = concatenate_datasets(all_datasets)
        
        return combined_dataset

    def _process_single_language(self, row_data, output_dir):
        """Process a single language with its data sources."""
        lang, target_token_count, sources = row_data
        
        if not sources:
            return 0

        full_dataset = self._load_and_prepare_data(lang, sources)
        if not full_dataset:
            return 0

        token_char_ratio = self._get_token_char_ratio(full_dataset)
        if token_char_ratio == 0:
            return 0

        accumulated_tokens = 0
        output_file = os.path.join(output_dir, f"{lang}.jsonl")

        with open(output_file, 'w', encoding="utf-8") as f, tqdm(total=target_token_count, desc=f"Processing {lang}") as pbar:
            for doc in cycle(full_dataset):
                if accumulated_tokens >= target_token_count:
                    break

                text = doc['text']
                estimated_tokens = doc.get('gemma_3_token_count') or int(len(text) * token_char_ratio)

                if self.format_type == 'llamafactory':
                    sample_data = {"text": text}
                    f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')

                tokens_added = estimated_tokens if estimated_tokens > 0 else len(self.tokenizer.encode(text))
                accumulated_tokens += tokens_added
                pbar.update(tokens_added)


        logging.info("Finished %s: %s tokens to %s", lang, f"{accumulated_tokens:,}", output_file)
        return accumulated_tokens

    def process_mixture(self):
        """Processes the data mixture and generates the output."""
        if self.mixture_df is None:
            self.read_mixture_config()
        assert self.mixture_df is not None, "Mixture DataFrame is not initialized."
        
        logging.info("Loaded mixture config with %d rows", len(self.mixture_df))
        self.mixture_df['token_count'] = self.mixture_df['token_count'].fillna(0)
        self.mixture_df['split'] = self.mixture_df['split'].fillna('')

        config_name = os.path.splitext(os.path.basename(self.config))[0]
        if self.fraction != 1.:
            config_name += "_f" + self.fraction_text.replace("/","-") # type: ignore
        output_dir = os.path.join(self.output_path, config_name)
        os.makedirs(output_dir, exist_ok=True)

        language_tasks = []
        for _, row in self.mixture_df.iterrows():
            subset = row['subset']
            original_token_count = int(row['token_count'])
            target_token_count = int(original_token_count * self.fraction)
            splits = row['split'].split(',') if isinstance(row['split'], str) else []

            if (self.subsets and subset in self.subsets) or (not self.subsets):
                language_tasks.append((subset, target_token_count, splits))
        random.shuffle(language_tasks)
        logging.info("Processing %d language subsets", len(language_tasks))
        
        total_tokens = 0
        if self.num_workers > 1:
            with Pool(processes=self.num_workers) as pool:
                process_func = partial(self._process_single_language, output_dir=output_dir)
                results = pool.map(process_func, language_tasks)
                total_tokens = sum(results)
        else:
            for task in language_tasks:
                total_tokens += self._process_single_language(task, output_dir)

        logging.info("Total tokens processed: %.2fM", total_tokens / 1_000_000)
        update_dataset_info(self.output_path, config_name)

def main():
    parser = argparse.ArgumentParser(description="Pretraining data mixture processor.")
    parser.add_argument("--config", type=str, help="Data mixture CSV.")
    parser.add_argument("--format", type=str, default="llamafactory", help="Output format.")
    parser.add_argument("--output", type=str, help="Output folder.")
    parser.add_argument("--model_name", type=str, default="google/gemma-3-4b-pt", help="Tokenizer model.")
    parser.add_argument("--num_workers", type=int, default=12, help="Worker count.")
    parser.add_argument("--token", type=str, default=None, help="Optional Hugging Face token.")
    parser.add_argument("--fraction", type=str, default="1.0", help="Subset ratio.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--dataset_id", type=str, default=DEFAULT_DATASET_ID, help="Source HF dataset id.")
    parser.add_argument("--revision", type=str, default=DEFAULT_REVISION, help="Source dataset revision.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Optional datasets cache directory.")
    parser.add_argument("--subsets", nargs='+', help="List of subsets to process. If not provided, all subsets in the CSV will be processed.")
    args = parser.parse_args()

    if not args.config or not args.output:
        parser.error("--config and --output are required.")

    data_mixture = DataMixture(
        args.config, 
        args.output,
        model_name=args.model_name,
        num_workers=args.num_workers,
        format_type=args.format,
        token=args.token,
        fraction=args.fraction,
        subsets=args.subsets,
        seed=args.seed,
        dataset_id=args.dataset_id,
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    data_mixture.process_mixture()

if __name__ == "__main__":
    main()
