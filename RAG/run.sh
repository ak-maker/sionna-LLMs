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
is_demo=false
. utils/parse_options.sh


if (( $(echo "$stage <= 0.0" | bc -l) )); then
    python RAG/code/preprocess/clean.py \
        RAG/data/demo \
        data/clean || { log "failed to clean"; exit 1; }
fi

if (( $(echo "$stage <= 0.1" | bc -l) )); then
    python RAG/code/preprocess/chunk.py run \
        RAG/data/clean \
        RAG/data/chunk.jsonl || { log "failed to chunk"; exit 1; }
fi

if (( $(echo "$stage <= 0.2" | bc -l) )); then
    python RAG/code/parallel_request.py \
        --model text-embedding-3-small \
        --base-url https://drchat.xyz \
        --api-key <YOUR_API_KEY> \
        RAG/data/chunk.jsonl \
        RAG/data/embed.jsonl || { log "failed to do parallel_request"; exit 1; }
fi

if (( $(echo "$stage <= 0.3" | bc -l) )); then
    python RAG/code/vectordb.py \
        RAG/data/chunk.jsonl \
        RAG/data/embed.jsonl || { log "failed to create vectordb"; exit 1; }
fi

if (( $(echo "$stage <= 1.0" | bc -l) )); then
    if [ "$is_demo" = true ]; then
        python code/answer.py demo \
            --rerank \
            RAG/data/chunk.jsonl \
            RAG/data/embed.jsonl
    else
        for k in $(seq 1 4); do
            python RAG/code/answer.py batch \
                --top-k $k \
                RAG/data/question.jsonl \
                RAG/data/prediction_k${k}.jsonl
        done
    fi
fi