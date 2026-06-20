#!/usr/bin/env python3
"""Expand benchmark aliases into comma-separated lm-eval task names."""

from __future__ import annotations

import argparse
import json


AFRO_LANGUAGES = ["eng", "swa", "kin", "hau", "amh", "xho", "sna", "zul", "ibo", "yor", "sot", "lin", "orm", "lug", "wol"]
AFRO_LANGUAGES_GAZ = ["eng", "swa", "kin", "hau", "amh", "xho", "sna", "zul", "ibo", "yor", "sot", "lin", "gaz", "lug", "wol"]

AFRIMGSM_FULL = ["amh", "ewe", "fra", "hau", "ibo", "kin", "lin", "lug", "orm", "sna", "sot", "swa", "twi", "vai", "wol", "xho", "yor", "zul"]
AFRIMMLU_XNLI_FULL = ["amh", "eng", "ewe", "fra", "hau", "ibo", "kin", "lin", "lug", "orm", "sna", "sot", "swa", "twi", "wol", "xho", "yor", "zul"]

AFRO_EXTRA_MGSM = ["ewe", "fra", "twi", "vai"]
AFRO_EXTRA_MMLU_XNLI = ["ewe", "fra", "twi"]
AFRO_EXTRA_INJONGO = ["ewe", "twi"]

AFRO_FLORES_EXTRA_LATN = [
    "afr", "aka", "bam", "bem", "cjk", "dik", "dyu", "ewe", "fon", "fuv",
    "kab", "kam", "kbp", "kea", "kik", "kmb", "knc", "kon", "lua", "luo",
    "mos", "nso", "nus", "nya", "run", "sag", "som", "ssw", "taq", "tsn",
    "tso", "tum", "twi", "umb",
]
AFRO_FLORES_EXTRA_SPECIAL = [
    ("aeb", "Arab"), ("ary", "Arab"), ("arz", "Arab"), ("knc", "Arab"),
    ("taq", "Tfng"), ("tir", "Ethi"), ("tzm", "Tfng"),
]
AFRO_FLORES_CUSTOM = ["afr", "ewe", "twi", "som", "nya", "tsn"]
AFRO_FLORES_CUSTOM_SPECIAL = [("arz", "Arab"), ("ary", "Arab"), ("tir", "Ethi"), ("aeb", "Arab")]

AFRO_EXTRA_BELEBELE = ["afr", "ary", "arz", "bam", "fra", "fuv", "kea", "luo", "nya", "plt", "por", "som", "ssw", "tir", "tsn", "tso"]
AFRO_BELEBELE_CUSTOM = ["afr", "fra", "por", "arz", "ary", "plt", "nya", "tsn", "som", "tir"]

AFRO_EXTRA_SIB = [
    "aeb", "afr", "aka", "ary", "arz", "bam", "bem", "cjk", "dik", "dyu",
    "ewe", "fon", "fra", "fuv", "kab", "kam", "kbp", "kea", "kik", "kmb",
    "knc", "kon", "lua", "luo", "mos", "nso", "nus", "nya", "plt", "por",
    "run", "sag", "som", "ssw", "taq", "tir", "tso", "tum", "twi", "tzm",
    "umb",
]
AFRO_SIB_CUSTOM = ["aeb", "afr", "ewe", "twi", "ary", "arz", "nya", "plt", "por", "som", "tir"]

MMMLU_LOCALES = ["ar_xy", "bn_bd", "de_de", "es_la", "fr_fr", "hi_in", "id_id", "it_it", "ja_jp", "ko_kr", "pt_br", "sw_ke", "yo_ng", "zh_cn"]
DAMT_LANGUAGES = ["am", "ha", "sw", "yo", "zu"]


def with_prompts(prefix: str, languages: list[str], count: int = 5) -> list[str]:
    return [f"{prefix}_{lang}_prompt_{prompt}" for lang in languages for prompt in range(1, count + 1)]


def flores_light_base() -> list[str]:
    latn = ["kin", "hau", "xho", "sna", "zul", "ibo", "yor", "sot", "lin", "lug", "wol"]
    return (
        [f"flores_{lang}_Latn-eng_Latn" for lang in latn]
        + ["flores_swh_Latn-eng_Latn", "flores_amh_Ethi-eng_Latn", "flores_gaz_Latn-eng_Latn"]
        + [f"flores_eng_Latn-{lang}_Latn" for lang in latn]
        + ["flores_eng_Latn-swh_Latn", "flores_eng_Latn-amh_Ethi", "flores_eng_Latn-gaz_Latn"]
    )


def flores_extra_base() -> list[str]:
    return (
        [f"flores_{lang}_Latn-eng_Latn" for lang in AFRO_FLORES_EXTRA_LATN]
        + [f"flores_{lang}_{script}-eng_Latn" for lang, script in AFRO_FLORES_EXTRA_SPECIAL]
        + [f"flores_eng_Latn-{lang}_Latn" for lang in AFRO_FLORES_EXTRA_LATN]
        + [f"flores_eng_Latn-{lang}_{script}" for lang, script in AFRO_FLORES_EXTRA_SPECIAL]
    )


def flores_custom_base() -> list[str]:
    return (
        [f"flores_{lang}_Latn-eng_Latn" for lang in AFRO_FLORES_CUSTOM]
        + [f"flores_eng_Latn-{lang}_Latn" for lang in AFRO_FLORES_CUSTOM]
        + [f"flores_{lang}_{script}-eng_Latn" for lang, script in AFRO_FLORES_CUSTOM_SPECIAL]
        + [f"flores_eng_Latn-{lang}_{script}" for lang, script in AFRO_FLORES_CUSTOM_SPECIAL]
    )


def flores_prompts(tasks: list[str]) -> list[str]:
    return [f"{task}_prompt_{prompt}" for task in tasks for prompt in range(1, 4)]


TASK_GROUPS: dict[str, list[str]] = {
    "afrimgsm_light_cot_tasks": with_prompts("afrimgsm_cot", AFRO_LANGUAGES),
    "afrimmlu_light_tasks": with_prompts("afrimmlu_direct", AFRO_LANGUAGES),
    "afrixnli_light_tasks": with_prompts("afrixnli", AFRO_LANGUAGES),
    "belebele_light_tasks": with_prompts("belebele", AFRO_LANGUAGES_GAZ),
    "sib_light_tasks": with_prompts("sib", AFRO_LANGUAGES_GAZ),
    "african_flores_light_tasks": flores_prompts(flores_light_base()),
    "injongointent_light_tasks": with_prompts("injongointent", AFRO_LANGUAGES),
    "injongointent_light_tasks_prompt_1": [f"injongointent_{lang}_prompt_1" for lang in AFRO_LANGUAGES],
    "injongointent_light_tasks_prompt_2": [f"injongointent_{lang}_prompt_2" for lang in AFRO_LANGUAGES],
    "injongointent_light_tasks_prompt_3": [f"injongointent_{lang}_prompt_3" for lang in AFRO_LANGUAGES],
    "injongointent_light_tasks_prompt_4": [f"injongointent_{lang}_prompt_4" for lang in AFRO_LANGUAGES],
    "injongointent_light_tasks_prompt_5": [f"injongointent_{lang}_prompt_5" for lang in AFRO_LANGUAGES],
    "afrimgsm_extra_cot_tasks": with_prompts("afrimgsm_cot", AFRO_EXTRA_MGSM),
    "afrimmlu_extra_tasks": with_prompts("afrimmlu_direct", AFRO_EXTRA_MMLU_XNLI),
    "afrixnli_extra_tasks": with_prompts("afrixnli", AFRO_EXTRA_MMLU_XNLI),
    "injongointent_extra_tasks": with_prompts("injongointent", AFRO_EXTRA_INJONGO),
    "african_flores_extra_tasks": flores_prompts(flores_extra_base()),
    "african_flores_custom_tasks": flores_prompts(flores_custom_base()),
    "belebele_extra_tasks": with_prompts("belebele", AFRO_EXTRA_BELEBELE),
    "belebele_custom_tasks": with_prompts("belebele", AFRO_BELEBELE_CUSTOM),
    "sib_extra_tasks": with_prompts("sib", AFRO_EXTRA_SIB),
    "sib_custom_tasks": with_prompts("sib", AFRO_SIB_CUSTOM),
    "afrimgsm_cot_tasks": with_prompts("afrimgsm_cot", AFRIMGSM_FULL),
    "afrimmlu_tasks": with_prompts("afrimmlu_direct", AFRIMMLU_XNLI_FULL),
    "afrixnli_tasks": with_prompts("afrixnli", AFRIMMLU_XNLI_FULL),
    "belebele_tasks": with_prompts("belebele", AFRO_LANGUAGES_GAZ + AFRO_EXTRA_BELEBELE),
    "sib_tasks": with_prompts("sib", AFRO_LANGUAGES_GAZ + AFRO_EXTRA_SIB),
    "african_flores_tasks": flores_prompts(flores_light_base() + flores_extra_base()),
    "injongointent_tasks": with_prompts("injongointent", AFRO_LANGUAGES + AFRO_EXTRA_INJONGO),
    "mmmlu_tasks": [f"mmmlu_{locale}" for locale in MMMLU_LOCALES],
    "damt_doc10_health_tasks": (
        [f"doc_damt_doc_health_10_en-{lang}_prompt{prompt}" for lang in DAMT_LANGUAGES for prompt in range(1, 5)]
        + [f"doc_damt_doc_health_10_{lang}-en_prompt{prompt}" for lang in DAMT_LANGUAGES for prompt in range(1, 5)]
    ),
    "damt_doc10_health_tasks_prompt_1": [f"doc_damt_doc_health_10_en-{lang}_prompt1" for lang in DAMT_LANGUAGES] + [f"doc_damt_doc_health_10_{lang}-en_prompt1" for lang in DAMT_LANGUAGES],
    "damt_doc10_health_tasks_prompt_2": [f"doc_damt_doc_health_10_en-{lang}_prompt2" for lang in DAMT_LANGUAGES] + [f"doc_damt_doc_health_10_{lang}-en_prompt2" for lang in DAMT_LANGUAGES],
    "damt_doc10_health_tasks_prompt_3": [f"doc_damt_doc_health_10_en-{lang}_prompt3" for lang in DAMT_LANGUAGES] + [f"doc_damt_doc_health_10_{lang}-en_prompt3" for lang in DAMT_LANGUAGES],
    "damt_doc10_health_tasks_prompt_4": [f"doc_damt_doc_health_10_en-{lang}_prompt4" for lang in DAMT_LANGUAGES] + [f"doc_damt_doc_health_10_{lang}-en_prompt4" for lang in DAMT_LANGUAGES],
}

TASK_GROUPS["afrobench_lite"] = [
    task
    for group in [
        "afrimgsm_light_cot_tasks",
        "afrimmlu_light_tasks",
        "afrixnli_light_tasks",
        "belebele_light_tasks",
        "sib_light_tasks",
        "african_flores_light_tasks",
        "injongointent_light_tasks",
    ]
    for task in TASK_GROUPS[group]
]
TASK_GROUPS["afrobench_extra"] = [
    task
    for group in [
        "afrimgsm_extra_cot_tasks",
        "afrimmlu_extra_tasks",
        "afrixnli_extra_tasks",
        "injongointent_extra_tasks",
        "african_flores_custom_tasks",
        "belebele_custom_tasks",
        "sib_custom_tasks",
    ]
    for task in TASK_GROUPS[group]
]


def resolve(groups: list[str]) -> list[str]:
    tasks: list[str] = []
    seen: set[str] = set()
    for group in groups:
        names = TASK_GROUPS.get(group, [group])
        for name in names:
            if name not in seen:
                tasks.append(name)
                seen.add(name)
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description="Expand lm-eval benchmark task aliases.")
    parser.add_argument("groups", nargs="*", help="Task group aliases or task names.")
    parser.add_argument("--list", action="store_true", help="List known aliases and task counts.")
    parser.add_argument("--json", action="store_true", help="Print tasks as JSON.")
    parser.add_argument("--newline", action="store_true", help="Print one task per line.")
    args = parser.parse_args()

    if args.list:
        for name in sorted(TASK_GROUPS):
            print(f"{name}\t{len(TASK_GROUPS[name])}")
        return
    if not args.groups:
        parser.error("provide at least one task group")

    tasks = resolve(args.groups)
    if args.json:
        print(json.dumps(tasks, indent=2))
    elif args.newline:
        print("\n".join(tasks))
    else:
        print(",".join(tasks))


if __name__ == "__main__":
    main()
