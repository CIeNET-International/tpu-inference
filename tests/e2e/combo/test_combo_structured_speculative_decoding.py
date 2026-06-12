# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""End-to-end coverage for structured output with speculative decoding.


These tests load real model weights and run the vLLM TPU engine across a full
v6e-8 slice. They are intentionally not suitable for host-only test suites.
"""

from __future__ import annotations

import gc
import json
import os
import time

import pytest
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

CHOICES = ["Positive", "Negative"]
SINGLE_TOKEN_CHOICES = ["0", "1"]
MULTI_TOKEN_CHOICES = ["Strongly Positive", "Strongly Negative"]
SINGLE_TOKEN_PROMPTS = [
    "Choose exactly one digit, 0 or 1. Return only the digit.",
    "Choose exactly one digit, 1 or 0. Return only the digit.",
]
PROMPTS = [
    "Choose exactly one label, Positive or Negative. The TPU inference result was fast and correct.",
    "Choose exactly one label, Positive or Negative. The request failed and returned an invalid result.",
]
MULTI_TOKEN_PROMPTS = [
    "Classify the sentiment as exactly Strongly Positive or Strongly Negative: TPU inference was excellent.",
    "Classify the sentiment as exactly Strongly Positive or Strongly Negative: The result was completely wrong.",
]
JSON_PROMPTS = [
    "Review the statement and return JSON containing sentiment (Positive or Negative) and score (1 to 5): TPU inference was excellent.",
    "Review the statement and return JSON containing sentiment (Positive or Negative) and score (1 to 5): The result was completely wrong.",
]
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": CHOICES,
        },
        "score": {
            "type": "integer",
            "minimum": 1,
            "maximum": 5,
        },
    },
    "required": ["sentiment", "score"],
    "additionalProperties": False,
}


def _get_tensor_parallel_size() -> int:
    return int(os.environ.get("TPU_TP_SIZE", "8"))


def _ngram_config(num_speculative_tokens: int = 3) -> tuple[str, dict]:
    return (
        os.environ.get("NGRAM_MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct"),
        {
            "method": "ngram",
            "prompt_lookup_max": 5,
            "prompt_lookup_min": 2,
            "num_speculative_tokens": num_speculative_tokens,
        },
    )


def _eagle3_config(num_speculative_tokens: int = 3) -> tuple[str, dict]:
    return (
        os.environ.get("EAGLE3_MODEL_NAME",
                       "meta-llama/Meta-Llama-3.1-8B-Instruct"),
        {
            "method":
            "eagle3",
            "model":
            os.environ.get("EAGLE3_DRAFT_MODEL_NAME",
                           "unkmaster/EAGLE3-LLaMA3.1-Instruct-8B"),
            "num_speculative_tokens":
            num_speculative_tokens,
            "draft_tensor_parallel_size":
            1,
        },
    )


def _create_llm(model_name: str, speculative_config: dict | None) -> LLM:
    return LLM(
        model=model_name,
        speculative_config=speculative_config,
        max_model_len=256,
        max_num_batched_tokens=256,
        max_num_seqs=len(PROMPTS),
        tensor_parallel_size=_get_tensor_parallel_size(),
        enable_prefix_caching=False,
        model_loader_extra_config={"enable_weights_track": False},
    )


def _shutdown_llm(llm: LLM) -> None:
    if hasattr(llm.llm_engine, "shutdown"):
        llm.llm_engine.shutdown()


@pytest.fixture(params=[
    pytest.param(_ngram_config(), id="ngram"),
    pytest.param(_eagle3_config(), id="eagle3"),
], )
def speculative_llm(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    model_name, speculative_config = request.param
    if speculative_config["method"] == "eagle3":
        model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
        monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    llm = _create_llm(model_name, speculative_config)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)
        del llm
        gc.collect()
        time.sleep(10)


@pytest.fixture(params=[
    pytest.param(_ngram_config(1), id="ngram-one-spec-token"),
    pytest.param(_eagle3_config(1),
                 id="eagle3-one-spec-token",
                 marks=pytest.mark.xfail(
                     reason="Eagle3 with one speculative token failure")),
], )
def single_spec_token_llm(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
):
    model_name, speculative_config = request.param
    if speculative_config["method"] == "eagle3":
        model_impl = os.environ.get("MODEL_IMPL_TYPE", "auto")
        monkeypatch.setenv("DRAFT_MODEL_IMPL_TYPE", model_impl)

    llm = _create_llm(model_name, speculative_config)
    try:
        yield llm
    finally:
        _shutdown_llm(llm)
        del llm
        gc.collect()
        time.sleep(10)


def test_single_token_structured_decoding_with_speculative_decoding(
    speculative_llm: LLM, ):
    """Exercise structured masking and speculative decoding in one step."""
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1,
        structured_outputs=StructuredOutputsParams(
            choice=SINGLE_TOKEN_CHOICES),
    )

    assert speculative_llm.llm_engine.vllm_config.speculative_config is not None

    outputs = speculative_llm.generate(SINGLE_TOKEN_PROMPTS, sampling_params)
    results = [output.outputs[0].text for output in outputs]
    assert all(result in SINGLE_TOKEN_CHOICES for result in results), results


def test_multistep_structured_decoding_with_one_speculative_token(
    single_spec_token_llm: LLM, ):
    """Isolate rollback behavior with one speculative token per step."""
    speculative_config = (
        single_spec_token_llm.llm_engine.vllm_config.speculative_config)
    assert speculative_config is not None
    assert speculative_config.num_speculative_tokens == 1

    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(choice=CHOICES),
    )

    outputs = single_spec_token_llm.generate(PROMPTS, sampling_params)
    results = [output.outputs[0].text for output in outputs]
    assert all(result in CHOICES for result in results), results


@pytest.mark.xfail(
    reason=("Multi-step structured output with speculative decoding currently "
            "crashes during speculative-token rollback "
            "(AttributeError: __delitem__)"))
def test_multistep_structured_decoding_with_speculative_decoding(
    speculative_llm: LLM, ):
    """Run structured speculative generation through a real TPU engine."""
    configured_tp_size = (speculative_llm.llm_engine.vllm_config.
                          parallel_config.tensor_parallel_size)
    assert configured_tp_size == _get_tensor_parallel_size()

    choice_params = SamplingParams(
        temperature=0,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(choice=CHOICES),
    )
    choice_outputs = speculative_llm.generate(PROMPTS, choice_params)
    choice_results = [output.outputs[0].text for output in choice_outputs]
    assert all(result in CHOICES for result in choice_results), choice_results

    multi_token_params = SamplingParams(
        temperature=0,
        max_tokens=8,
        structured_outputs=StructuredOutputsParams(choice=MULTI_TOKEN_CHOICES),
    )
    multi_token_outputs = speculative_llm.generate(MULTI_TOKEN_PROMPTS,
                                                   multi_token_params)
    multi_token_results = [
        output.outputs[0].text for output in multi_token_outputs
    ]
    assert all(result in MULTI_TOKEN_CHOICES
               for result in multi_token_results), multi_token_results

    json_params = SamplingParams(
        temperature=0,
        max_tokens=32,
        structured_outputs=StructuredOutputsParams(json=REVIEW_SCHEMA),
    )
    json_outputs = speculative_llm.generate(JSON_PROMPTS, json_params)
    for output in json_outputs:
        result = json.loads(output.outputs[0].text)
        assert set(result) == {"sentiment", "score"}
        assert result["sentiment"] in CHOICES
        assert isinstance(result["score"], int)
        assert not isinstance(result["score"], bool)
        assert 1 <= result["score"] <= 5
