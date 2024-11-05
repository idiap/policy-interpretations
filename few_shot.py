# SPDX-FileCopyrightText: 2024 Idiap Research Institute
#
# SPDX-License-Identifier: MIT

""" Script to run few-shot inference with LLMs. """

import json
import os
from collections import defaultdict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


def construct_messages(
    icl_examples, cur_example, is_instruct=False, use_system_msg=False
):
    messages = []
    if is_instruct:
        if use_system_msg:
            messages.append(
                {
                    "role": "system",
                    "content": "You are a chatbot that analyzes political announcements and replies with a coded interpretation of its main points.",
                }
            )
        for icl_ex in icl_examples:
            messages.append(
                {"role": "user", "content": f"Announcement: {icl_ex['src']}"}
            )
            messages.append(
                {"role": "assistant", "content": f"Interpretation: {icl_ex['tgt']}"}
            )
        messages.append(
            {"role": "user", "content": f"Announcement: {cur_example['src']}"}
        )
    else:
        for icl_ex in icl_examples:
            messages.append(f"Announcement: {icl_ex['src']}")
            messages.append(f"Interpretation: {icl_ex['tgt']}")
        messages.append(f"Announcement: {cur_example['src']}")
    return messages


def run(tokenizer, model, messages, is_instruct, args):
    if is_instruct:
        model_inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    else:
        model_inputs = tokenizer(["\n".join(messages)], return_tensors="pt")
        model_inputs = model_inputs.input_ids
    assert model_inputs.shape[0] == 1, "Supports only batch size of 1"
    input_length = model_inputs.shape[1]
    if torch.cuda.is_available():
        model_inputs = model_inputs.to("cuda")
    generated_ids = model.generate(
        input_ids=model_inputs,
        do_sample=True,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    assistant_response = tokenizer.batch_decode(
        generated_ids[:, input_length:], skip_special_tokens=True
    )[0].strip()
    if assistant_response.startswith("Interpretation:"):
        assistant_response = assistant_response[len("Interpretation:") :].strip()
    assistant_response = assistant_response.replace("\n", " ")
    return assistant_response


def main(args):
    set_seed(args.seed)

    # get Hugging Face token
    hf_token = None
    if args.hf_token:
        hf_token = open(args.hf_token).read().strip()

    # read in-context learning (ICL) examples
    with open(args.icl_path) as f:
        icl_examples = json.load(f)
    icl_examples = icl_examples[: args.num_shots]
    assert (
        len(icl_examples) == args.num_shots
    ), f"Wrong number of ICL examples: {len(icl_examples)}"

    # read inputs
    with open(args.eval_path) as f:
        data = json.load(f)
    inputs = [
        {
            "src": " ".join(example["src_sents"]),
            "tgt": example["tgt"],
            "name": example["name"],
        }
        for example in data
    ]

    # init tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, token=hf_token, padding_side="left"
    )
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.bf16:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            token=hf_token,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, token=hf_token, device_map="auto"
        )
    model.eval()
    is_instruct = "-it" in args.model_name or "-Instruct" in args.model_name

    # inference
    refdocs_processed = set()
    outputs = defaultdict(list)
    for cur_example in inputs:
        # check if refdoc has already been processed
        target = cur_example["tgt"]
        refdoc = cur_example["name"]
        if refdoc in refdocs_processed:
            index = outputs["refdocs"].index(refdoc)
            outputs["references"][index].append(target)
            continue

        # run model
        messages = construct_messages(
            icl_examples, cur_example, is_instruct, args.use_system_msg
        )
        candidate = run(tokenizer, model, messages, is_instruct, args)

        # store inputs and output
        outputs["sources"].append(cur_example["src"])
        outputs["references"].append([target])
        outputs["candidates"].append(candidate)
        outputs["refdocs"].append(refdoc)
        refdocs_processed.add(refdoc)

    # write results to files
    for name in ("sources", "candidates", "refdocs"):
        with open(os.path.join(args.output_dir, f"{name}.txt"), "w") as f:
            for output in outputs[name]:
                f.write(output + "\n")
    with open(os.path.join(args.output_dir, "references.txt"), "w") as f:
        for refdoc_references in outputs["references"]:
            f.write(args.reference_join.join(refdoc_references) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot inference with LLMs.")
    parser.add_argument("--output_dir", required=True, help="Path to output folder")
    parser.add_argument(
        "--reference_join", default="<ref>", help="Marker to join multiple references."
    )
    parser.add_argument(
        "--hf_token", default="hf_token.txt", help="Path to Hugging Face token file"
    )
    parser.add_argument(
        "--model_name", default="google/gemma-2-9b-it", help="HF model name"
    )
    parser.add_argument(
        "--icl_path", required=True, help="Path to json file with few-shot examples"
    )
    parser.add_argument(
        "--eval_path", required=True, help="Path to json file with eval data"
    )
    parser.add_argument(
        "--num_shots", type=int, default=5, help="Number of few-shot examples"
    )
    parser.add_argument(
        "--use_system_msg", action="store_true", help="Use a system message"
    )
    parser.add_argument("--bf16", action="store_true", help="Load model in bfloat16.")

    # generation args
    parser.add_argument("--max_new_tokens", type=int, default=250)
    parser.add_argument("--min_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    main(parser.parse_args())
