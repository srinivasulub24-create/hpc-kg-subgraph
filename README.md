# HPC A1: GPU Knowledge-Graph Subgraph Extraction for Agentic RAG
Sequential baseline (C++), profiling, and parallelization plan.

## Reproduce
    python prep_data.py cooccur --dir data/raw/kevin --out data/g_kevin2.txt --names data/names_kevin2.txt --window 50 --max-per-window 8 --min-count 3
    python make_question_queries.py --csv "Kevin Scott Questions.csv" --graph data/g_kevin2.txt --names data/names_kevin2.txt --out data/q_real.txt
    ./run_a1.sh data/g_kevin2.txt data/q_real.txt

Dataset: https://github.com/microsoft/graphrag-benchmarking-datasets (unzip the input text into data/raw/).
Results: results_kevin_questions/, results_msft/, results_kevin2/, results/seq_1m_hops2.json
