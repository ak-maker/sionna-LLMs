#!/bin/bash
#
#  2024   jerrypeng1937@gmail.com
#  demo on the usage of the code snippets to perform retrieval-augmented LLMs on sionna
#  https://nvlabs.github.io/sionna/
#  Some steps are unncessary and can be skipped.
#  They are developped just for educational purpose.


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

set -o allexport
source .env set
set +o allexport

umask 000


stage=0.0
. utils/parse_options.sh

llm=gpt4-1106-preview
exp=./exp/large/gpt4-1106-preview/
data=./data
mkdir -p $exp


if (( $(echo "$stage <= 1.0" | bc -l) )); then
    for k in $(seq 0 4); do
        python RAG/code/answer.py batch \
            --top-k $k \
            --llm $llm \
            $data/question.jsonl \
            $exp/prediction_k${k}.jsonl \
            || { log "failed on stage: 1.0, k $k"; exit 1; }
    done
fi

if (( $(echo "$stage <= 1.1" | bc -l) )); then
    for k in $(seq 3 3); do
        python RAG/code/answer.py batch \
            --rerank \
            --top-k $k \
            --llm $llm \
            $data/question.jsonl \
            $exp/prediction_rerank_k${k}.jsonl \
            || { log "failed on stage: 1.1, k $k"; exit 1; }
    done
fi

# if (( $(echo "$stage <= 1.2" | bc -l) )); then
#     for k in $(seq 0 4); do
#         python code/eval/score.py \
#             $data/question.jsonl \
#             $data/answer.jsonl \
#             $exp/prediction_k${k}.jsonl \
#             $exp/evaluation_k${k}.jsonl \
#             || { log "failed on stage: 1.2, k $k"; exit 1; }
#     done
# fi

if (( $(echo "$stage <= 1.3" | bc -l) )); then
    for k in $(seq 3 3); do
        python RAG/code/eval/score.py \
            $data/question.jsonl \
            $data/answer.jsonl \
            $exp/prediction_rerank_k${k}.jsonl \
            $exp/evaluation_rerank_k${k}.jsonl \
            || { log "failed on stage: 1.3, k $k"; exit 1; }
    done
fi