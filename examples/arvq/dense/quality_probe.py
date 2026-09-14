# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small fixed teacher-forced diagnostic, not a model accuracy benchmark."""

import argparse
import json
import math
import os
import pathlib
import time
import urllib.request

# ruff: noqa: E501
# Preserve the exact fixed diagnostic passages as single string literals.

TEXTS = [
    "The rain stopped just before breakfast. Maya opened the kitchen window and noticed that the street was quiet. She put two slices of bread in the toaster, filled the kettle, and checked the timetable. Her train would leave in forty minutes, so there was enough time to eat before walking to the station.",
    "A carpenter has a board that is 120 centimeters long. She cuts off a piece measuring 35 centimeters and another measuring 28 centimeters. The total length removed is 35 + 28 = 63 centimeters. The remaining piece measures 120 - 63 = 57 centimeters. These calculations ignore the small amount of material lost to the saw.",
    "In Python, a dictionary maps keys to values. To count words, start with an empty dictionary and update the count for each word. For example:\ncounts = {}\nfor word in text.split():\n    counts[word] = counts.get(word, 0) + 1\nThe get method supplies zero when a word has not been encountered before.",
    "To reverse a list without changing the original, create a new list from a reverse slice. For example, if values = [2, 4, 6, 8], then values[::-1] produces [8, 6, 4, 2]. The original list remains [2, 4, 6, 8]. Slicing is convenient, although the new list requires additional memory.",
    "There are three boxes labeled red, green, and blue. The red box contains four pencils, the green box contains six pencils, and the blue box contains two pencils. Moving one pencil from the green box to the blue box leaves four, five, and three pencils respectively. The total remains twelve because no pencils were added or removed.",
    "An experiment should compare a proposed change with a clear baseline. Keep the inputs and measurement procedure consistent, repeat the measurements, and record unexpected results. A faster operation does not necessarily make the whole program faster: its contribution depends on how often it runs and how much time other operations require.",
    "Elena escribió una lista antes de ir al mercado. Necesitaba pan, tomates y una botella de aceite. Al llegar, encontró todos los productos excepto el pan, que estaba agotado. Decidió comprarlo en la pequeña tienda de la esquina cuando regresara a casa.",
    "Luc a préparé son sac la veille du départ. Il y a mis une veste, un carnet et quelques vêtements. Le lendemain matin, il a fermé les fenêtres, vérifié son billet et quitté la maison. Il voulait arriver à la gare assez tôt pour trouver tranquillement son quai.",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--base-url", default="http://localhost:8001")
    ap.add_argument("--image-id", default="unspecified")
    ap.add_argument("--dense-flag", choices=("0", "1"), required=True)
    ap.add_argument("--output", type=pathlib.Path, required=True)
    a = ap.parse_args()
    result = {
        "label": a.label,
        "diagnostic": "Fixed authored passages; teacher-forced token NLL, not task accuracy.",
        "image_id": a.image_id,
        "dense_flag": a.dense_flag,
        "cases": [],
    }
    for index, prompt in enumerate(TEXTS):
        body = {
            "model": "jarrelscy/GLM-5.3-Vision-NVFP4-ARVQ-hybrid",
            "prompt": prompt,
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "temperature": 0,
        }
        req = urllib.request.Request(
            a.base_url.rstrip("/") + "/v1/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer " + os.environ.get("OPENAI_API_KEY", ""),
            },
        )
        start = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as response:
            x = json.load(response)
        vals = [
            v for v in x["choices"][0]["logprobs"]["token_logprobs"] if v is not None
        ]
        assert vals and all(math.isfinite(v) for v in vals)
        result["cases"].append(
            {
                "index": index,
                "prompt": prompt,
                "token_count": len(vals),
                "nll": -sum(vals) / len(vals),
                "token_logprobs": vals,
                "usage": x["usage"],
                "elapsed_s": time.perf_counter() - start,
            }
        )
    result["tokens"] = sum(c["token_count"] for c in result["cases"])
    result["nll"] = (
        sum(c["nll"] * c["token_count"] for c in result["cases"]) / result["tokens"]
    )
    result["perplexity"] = math.exp(result["nll"])
    p = a.output
    p.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("label", "tokens", "nll", "perplexity")}))


if __name__ == "__main__":
    main()
