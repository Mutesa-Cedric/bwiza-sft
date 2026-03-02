#!/usr/bin/env python3
"""Build prompt seed JSONL for Gemini SFT distillation."""

from __future__ import annotations

import argparse
from hashlib import sha1
import json
from pathlib import Path

RW_TOPICS = [
    "uburezi mu Rwanda",
    "ubuhinzi n'ubworozi",
    "ubuzima rusange",
    "ikoranabuhanga",
    "ubukungu bw'umuryango",
    "amakuru yo ku mbuga nkoranyambaga",
    "uburere bw'abana",
    "kwiga no gutegura ibizamini",
    "akazi n'imyuga",
    "iterambere ry'icyaro",
    "ubwikorezi bwo mu mujyi",
    "imihindagurikire y'ibihe",
    "ubucuruzi buto",
    "imiyoborere myiza",
    "ubukerarugendo",
    "kubungabunga umuco",
]

RW_TASK_TEMPLATES = [
    "Sobanura {topic} mu buryo bworoshye ku munyeshuri wa S3.",
    "Andika incamake ngufi ya {topic} mu nteruro 5.",
    "Tanga intambwe 7 zifatika ku muntu ushaka kunoza {topic}.",
    "Mpa ibibazo 5 n'ibisubizo byabyo kuri {topic}.",
    "Vuga ibyiza n'imbogamizi za {topic} mu Rwanda uyu munsi.",
    "Andika inama zifatika 10 ku bijyanye na {topic}.",
    "Sobanura amakosa akunze gukorwa muri {topic} n'uko wayakosora.",
    "Andika ubutumwa bugufi bwo kumenyesha abaturage ibyerekeye {topic}.",
]

CS_TASK_TEMPLATES = [
    "Mpa practical guide kuri {topic}, use Kinyarwanda but keep technical English terms.",
    "Sobanura {topic} in Kinyarwanda, then give a short English recap.",
    "Nkeneye step-by-step plan ya {topic}, but make it easy for beginners.",
    "Rewrite this policy message about {topic} in RW+EN mixed style for youth.",
]

EN_TASK_TEMPLATES = [
    "Explain {topic} for a Rwandan beginner in clear, practical language.",
    "Give a concise action plan for {topic} with 7 steps.",
]

FR_TASK_TEMPLATES = [
    "Explique {topic} de facon simple pour un eleve debutant.",
]

SW_TASK_TEMPLATES = [
    "Eleza {topic} kwa lugha rahisi kwa mwanafunzi wa sekondari.",
]

LANG_CONTROL = [
    "Sobanura {topic} mu Kinyarwanda gusa.",
    "Explain {topic} in English only.",
    "Sobanura {topic} mu Gifaransa gusa.",
    "Eleza {topic} kwa Kiswahili tu.",
    "User asks in English about {topic}; answer in Kinyarwanda.",
]


def _dedup(records: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for r in records:
        p = " ".join(str(r["prompt"]).strip().split()).lower()
        if p in seen:
            continue
        seen.add(p)
        out.append(r)
    return out


def _id(prompt: str) -> str:
    return sha1(prompt.encode("utf-8")).hexdigest()[:16]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build prompt seed JSONL")
    p.add_argument("--output_jsonl", default="outputs/sft/prompts.seed.jsonl")
    p.add_argument("--target", type=int, default=200)
    p.add_argument("--overwrite", action="store_true", default=False)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path} (pass --overwrite)")

    records: list[dict] = []

    for topic in RW_TOPICS:
        for t in RW_TASK_TEMPLATES:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"rw_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "rw_instruction",
                    "lang_mode": "rw",
                    "source": "template_seed_v1",
                }
            )

        for t in CS_TASK_TEMPLATES:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"cs_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "code_switch_instruction",
                    "lang_mode": "rw_mixed",
                    "source": "template_seed_v1",
                }
            )

        for t in EN_TASK_TEMPLATES:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"en_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "multilingual_retention",
                    "lang_mode": "en",
                    "source": "template_seed_v1",
                }
            )

        for t in FR_TASK_TEMPLATES:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"fr_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "multilingual_retention",
                    "lang_mode": "fr",
                    "source": "template_seed_v1",
                }
            )

        for t in SW_TASK_TEMPLATES:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"sw_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "multilingual_retention",
                    "lang_mode": "sw",
                    "source": "template_seed_v1",
                }
            )

        for t in LANG_CONTROL:
            prompt = t.format(topic=topic)
            records.append(
                {
                    "id": f"lc_{_id(prompt)}",
                    "prompt": prompt,
                    "task_type": "language_control",
                    "lang_mode": "control",
                    "source": "template_seed_v1",
                }
            )

    records = _dedup(records)

    if args.target > 0:
        records = records[: args.target]

    with out_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=True) + "\n")

    print(json.dumps({"output": str(out_path), "count": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
