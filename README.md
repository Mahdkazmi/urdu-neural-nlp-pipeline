# urdu-neural-nlp-pipeline

End-to-end neural NLP pipeline for BBC Urdu news — includes TF-IDF/PPMI word embeddings, Skip-gram Word2Vec, BiLSTM-CRF for POS tagging and NER, and a Transformer encoder for topic classification. Built from scratch in PyTorch.

## Overview

This project implements a complete NLP pipeline across three main components:

### Part 1: Word Embeddings & Similarity
- **TF-IDF embeddings** with category-level analysis and t-SNE visualization
- **PPMI (Positive Pointwise Mutual Information)** co-occurrence matrix
- **Skip-gram Word2Vec** trained on cleaned and raw Urdu corpus
- **4-condition comparison** (C1: PPMI baseline, C2: Skip-gram on raw, C3: Skip-gram on cleaned d=100, C4: Skip-gram on cleaned d=200)
- **Nearest-neighbor queries** and **analogy tests** for semantic evaluation

### Part 2: Sequence Tagging (POS & NER)
- **500 topic-stratified sentences** sampled from 229 BBC Urdu articles
- **Automatic annotation** using POS lexicon and NER gazetteers
- **BiLSTM encoder + Linear Chain CRF** for both POS tagging and NER
- **Frozen vs fine-tuned models** comparison
- **Ablation studies** (A1-A4): unidirectional LSTM, no dropout, random embeddings, softmax NER
- **Detailed evaluation**: confusion matrices, macro-F1, span-level NER metrics, error analysis

### Part 3: Topic Classification
- **5-way topic classification** (Politics, Sports, Economy, International, Health & Society)
- **Transformer encoder** with multi-head self-attention, positional encoding, and pre-LayerNorm blocks
- **BiLSTM baseline** for comparison
- **Attention heatmap visualization** on correctly classified examples
- **Training curves, confusion matrices, and macro-F1 evaluation**

## Requirements

- Python 3.11+
- PyTorch 2.11.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- NLTK

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/Mahdkazmi/urdu-neural-nlp-pipeline.git
cd urdu-neural-nlp-pipeline

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install torch numpy pandas scikit-learn nltk matplotlib seaborn
```

## Reproduction Instructions

### Running the Full Pipeline

The entire assignment is implemented in a single Jupyter notebook: `i23-2587_Assignment2_DS-C.ipynb`

```bash
# Start Jupyter
jupyter notebook

# Open i23-2587_Assignment2_DS-C.ipynb and run all cells sequentially (Kernel → Restart & Run All)
```

**Execution time:** ~10–15 minutes on CPU (skip-gram and Part 3 transformer training are the bottlenecks)

### Running Individual Parts

The notebook is structured as follows:

- **Cells 1–52**: Part 1 (Setup, TF-IDF, PPMI, Word2Vec, 4-condition comparison)
- **Cells 53–61**: Part 2 (Sampling, annotation, BiLSTM-CRF training, ablations, detailed evaluation)
- **Cells 62–70**: Part 3 (Part 3 data prep, Transformer & BiLSTM training, attention heatmaps, comparison)

You can run individual cells or sections independently if dependencies are already loaded.

### Input Files

- `cleaned.txt` — preprocessed BBC Urdu articles
- `raw.txt` — raw BBC Urdu articles
- `Metadata.json` — article metadata (date, title, URL, category)
- `assignment_text.txt` — assignment rubric specification

### Output Artifacts

All outputs are automatically saved:

**Part 1:**
- `embeddings/tfidf_matrix.npy` — TF-IDF embedding matrix (10,000 × 229)
- `embeddings/ppmi_matrix.npy` — PPMI co-occurrence matrix (10,000 × 10,000)
- `embeddings/embeddings_w2v.npy` — Word2Vec embeddings (10,000 × 100)
- `embeddings/word2idx.json` — word-to-index mapping
- `ppmi_tsne_top200.png` — t-SNE visualization
- `training_loss_c3.png` — Skip-gram training loss
- `training_loss_comparison.png` — 4-condition comparison plot

**Part 2:**
- `part2_data/` — annotated sentences, train/val/test splits (JSON & CoNLL format)
- `part2_training_curves.png` — POS/NER training/validation curves

**Part 3:**
- `part3_data/` — topic classification dataset splits (JSON)
- `models/transformer_cls.pt` — saved Transformer classifier checkpoint
- `models/bilstm_topic_cls.pt` — saved BiLSTM baseline checkpoint
- `part3_training_curves.png` — Transformer training curves
- `part3_bilstm_training_curves.png` — BiLSTM training curves
- `part3_attention_heatmaps.png` — attention visualization on correct examples

## Key Results

- **Part 1 MRR (best):** C1 PPMI baseline achieved 0.1000 MRR on nearest-neighbor queries
- **Part 2 POS Tagging:** BiLSTM fine-tuned achieved 0.671 test accuracy and 0.073 macro-F1
- **Part 2 NER:** CRF-based NER achieved 0.000 F1 (class imbalance; softmax alternative F1 = 0.031)
- **Part 3 Topic Classification:**
  - Transformer: 0.561 test accuracy, 0.144 macro-F1
  - BiLSTM baseline: 0.585 test accuracy, 0.189 macro-F1

## Project Structure

```
.
├── i23-2587_Assignment2_DS-C.ipynb    # Main notebook (all 3 parts)
├── cleaned.txt                         # Cleaned Urdu corpus
├── raw.txt                             # Raw Urdu corpus
├── Metadata.json                       # Article metadata
├── assignment_text.txt                 # Assignment rubric
├── embeddings/                         # Part 1 outputs
│   ├── tfidf_matrix.npy
│   ├── ppmi_matrix.npy
│   ├── embeddings_w2v.npy
│   └── word2idx.json
├── part2_data/                         # Part 2 outputs
│   ├── sentences_selected_500.json
│   ├── train/val/test_sentences.json
│   ├── annotated_sentences.json
│   ├── pos_lexicon.json
│   ├── ner_gazetteer.json
│   └── *.conll (CoNLL format files)
├── part3_data/                         # Part 3 outputs
│   ├── train_articles.json
│   ├── val_articles.json
│   └── test_articles.json
├── models/                             # Saved model checkpoints
│   ├── transformer_cls.pt
│   └── bilstm_topic_cls.pt
├── *.png                               # Visualizations and training curves
└── README.md                           # This file
```

## Technologies & Design Choices

- **PyTorch from scratch:** All neural modules (embeddings, encoders, CRF, Transformers) implemented manually without high-level frameworks
- **Urdu-specific preprocessing:** Custom tokenization, diacritic normalization, and Arabic script handling
- **BiLSTM + CRF:** Sequence labeling with structured prediction for sequence tagging tasks
- **Transformer encoder:** Multi-head self-attention with sinusoidal positional encoding and pre-LayerNorm architecture
- **Balanced sampling:** Topic-stratified 500-sentence dataset to minimize class imbalance in Parts 2 & 3

## Notes on Performance

- **Part 1:** Skip-gram analogies scored 0/10 due to the small corpus size (229 articles, ~369K tokens); PPMI baseline outperformed learnable embeddings on this dataset
- **Part 2:** NER severely hampered by entity scarcity in the small annotated set; POS tagging benefited from larger lexicon coverage
- **Part 3:** Transformer overfits to the dominant "Health & Society" class; both models predict this class for most test examples

## License

This is an academic assignment submission for a Neural NLP course.

## Author

**Student ID:** i23-2587

For questions or issues, please refer to the assignment rubric in `assignment_text.txt`.
