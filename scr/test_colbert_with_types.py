#!/usr/bin/env python3
"""
Test script for ColBERT model with type embeddings on test dataset.

This script:
1. Loads the trained ColBERT model with type embeddings
2. Encodes queries and layouts from test dataset
3. Computes retrieval scores using the modified ColBERT scoring
4. Evaluates performance using existing evaluation functions
"""

import pickle
import json
import torch
import pandas as pd
import numpy as np
from metric_eval import evaluate_layout, colbert_score, pad_tok_len
from tqdm import tqdm
import argparse
import os
import sys

# Ensure we can import training-time architectures
ROOT_DIR = os.path.dirname(__file__)
TRAIN_DIR = os.path.join(ROOT_DIR, "dataset", "train")
sys.path.insert(0, TRAIN_DIR)

from colbert.modeling.colbert import ColBERT
from colbert.infra import ColBERTConfig
from transformers import AutoTokenizer

from model_architectures import ARCHITECTURES, create_model, COLBERT_HIDDEN_DIM


def create_passthrough_model(base_model_path, type_embed_dim=32, device='cuda'):
    """
    Create a pass-through model with the same structure as ColBERTWithTypes,
    but skips type embedding computation entirely to save time.
    
    This is useful for testing/validation to ensure the model structure is correct
    and to compare against baseline ColBERT performance.
    """
    print(f"Creating pass-through model (skipping type embedding computation)...")
    print(f"Loading base ColBERT from {base_model_path}...")
    
    # Load base ColBERT
    config = ColBERTConfig(bsize=256, root='./', query_token_id='[Q]', doc_token_id='[D]')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = ColBERT(name=base_model_path, colbert_config=config)
    base_model.eval()
    
    # Create modified model with passthrough=True to skip type embedding computation
    model = ColBERTWithTypes(base_model, type_embed_dim=type_embed_dim, passthrough=True)
    
    model.eval()
    model = model.to(device)
    
    query_token_id = tokenizer.convert_tokens_to_ids(config.query_token_id)
    doc_token_id = tokenizer.convert_tokens_to_ids(config.doc_token_id)
    
    print("  ✓ Pass-through model created (skips type embedding, should match baseline ColBERT)")
    
    return model, tokenizer, query_token_id, doc_token_id


def load_trained_model(model_path, base_model_path, architecture='doc_type_only', type_embed_dim=32, dropout=0.0, device='cuda'):
    """Load trained ColBERT model + adapter (same architectures as train_with_wandb.py)."""
    print(f"Loading base ColBERT from {base_model_path}...")
    
    # Load base ColBERT (same as training)
    config = ColBERTConfig(bsize=256, root='./', query_token_id='[Q]', doc_token_id='[D]')
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    base_model = ColBERT(name=base_model_path, colbert_config=config)
    base_model.eval()
    
    # Create modified model using selected architecture (adapter only)
    if architecture not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. Available: {list(ARCHITECTURES.keys())}")
    model = create_model(architecture, base_model, type_embed_dim=type_embed_dim, dropout=dropout)
    
    # Load trained adapter weights (support both full-model and adapter-only checkpoints)
    print(f"Loading trained weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = None
    if isinstance(checkpoint, dict) and 'adapter_state_dict' in checkpoint:
        # Adapter-only checkpoint (recommended from train_with_wandb)
        state_dict = checkpoint['adapter_state_dict']
        print("  Detected adapter-only checkpoint (adapter_state_dict).")
    elif isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full model checkpoint (backward compatibility)
        state_dict = checkpoint['model_state_dict']
        print("  Detected full model checkpoint (model_state_dict).")
    else:
        # Raw state_dict
        state_dict = checkpoint
        print("  Detected raw state_dict checkpoint.")
    
    # Load into model; allow missing base_model.* keys if adapter-only
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Warning: missing keys when loading state_dict (likely base_model.*): {len(missing)} keys")
    if unexpected:
        print(f"  Warning: unexpected keys when loading state_dict: {unexpected}")
    
    model.eval()
    model = model.to(device)
    
    query_token_id = tokenizer.convert_tokens_to_ids(config.query_token_id)
    doc_token_id = tokenizer.convert_tokens_to_ids(config.doc_token_id)
    
    return model, tokenizer, query_token_id, doc_token_id


def apply_doc_adapter_on_precomputed(model, base_layout_embs, layout_types, device='cuda'):
    """
    Apply the trained type adapter on top of precomputed ColBERT document embeddings.
    
    Args:
        model: ColBERTWithTypes (trained adapter; base_model is frozen)
        base_layout_embs: list of np.ndarray, each [L, H] from encoded_layout_ColBERT.pkl
        layout_types: list of type strings aligned with base_layout_embs
    Returns:
        adapted_layout_embs: list of np.ndarray, each [L, H] after applying adapter
    """
    adapted = []
    arch = getattr(model, 'architecture', 'doc_type_only')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for emb_np, t in tqdm(zip(base_layout_embs, layout_types), desc="Applying doc adapter", total=len(base_layout_embs)):
            if emb_np is None or emb_np.size == 0:
                adapted.append(emb_np)
                continue
            base_embs = torch.from_numpy(emb_np).to(device)  # [L, H]
            base_embs = base_embs.unsqueeze(0)               # [1, L, H]
            type_id = torch.tensor([model.get_type_id(t)], device=device)
            
            batch_size, seq_len, hidden_dim = base_embs.shape
            
            if arch in ['doc_type_only', 'doc_type_query_proj']:
                # H + type via concat(H, type_emb) -> Linear(H+type_dim -> H)
                type_embs = model.type_embeddings(type_id)  # [1, type_dim]
                type_embs_expanded = type_embs.unsqueeze(1).expand(batch_size, seq_len, model.type_embed_dim)
                concat_embs = torch.cat([base_embs, type_embs_expanded], dim=-1)
                projected_embs = model.type_projection(concat_embs)
                projected_embs = model.dropout(projected_embs)
                projected_embs = projected_embs / (projected_embs.norm(dim=-1, keepdim=True) + 1e-8)
                adapted.append(projected_embs.squeeze(0).cpu().numpy())
            
            elif arch in ['residual_type', 'residual_type_query_proj']:
                # Residual: H + Project(type)
                type_embs = model.type_embeddings(type_id)          # [1, type_dim]
                type_projected = model.type_projection(type_embs)   # [1, H]
                type_projected_expanded = type_projected.unsqueeze(1).expand(batch_size, seq_len, hidden_dim)
                type_projected_expanded = model.dropout(type_projected_expanded)
                output_embs = base_embs + type_projected_expanded
                output_embs = output_embs / (output_embs.norm(dim=-1, keepdim=True) + 1e-8)
                adapted.append(output_embs.squeeze(0).cpu().numpy())
            
            else:
                # Fallback: no doc adapter
                adapted.append(emb_np)
    
    return adapted


def apply_query_adapter_on_precomputed(model, base_query_embs, device='cuda'):
    """
    Apply the trained query adapter (if any) on top of precomputed ColBERT query embeddings.
    
    For architectures without a query adapter, returns the base embeddings unchanged.
    """
    adapted = []
    arch = getattr(model, 'architecture', 'doc_type_only')
    model = model.to(device)
    model.eval()
    
    with torch.no_grad():
        for emb_np in tqdm(base_query_embs, desc="Applying query adapter", total=len(base_query_embs)):
            if emb_np is None or emb_np.size == 0:
                adapted.append(emb_np)
                continue
            base_embs = torch.from_numpy(emb_np).to(device)  # [L, H]
            
            if arch in ['doc_type_query_proj', 'residual_type_query_proj']:
                projected_embs = model.query_projection(base_embs)
                projected_embs = model.dropout(projected_embs)
                projected_embs = projected_embs / (projected_embs.norm(dim=-1, keepdim=True) + 1e-8)
                adapted.append(projected_embs.cpu().numpy())
            else:
                # No query adapter in this architecture; keep base embeddings
                adapted.append(emb_np)
    
    return adapted


def load_test_layouts(layouts_parquet_path):
    """
    Load and prepare test layouts with type information.
    
    Test dataset structure (MMDocIR_layouts.parquet):
    - Has 'type' column directly
    - Has 'text' column for text/equation types
    - Has 'ocr_text' or 'vlm_text' for table/image types
    """
    print(f"Loading test layouts from {layouts_parquet_path}...")
    df = pd.read_parquet(layouts_parquet_path)
    
    # Ensure we have a 'text' column for encoding
    # Use 'text' if available, otherwise use 'ocr_text' or 'vlm_text'
    if 'text' not in df.columns or df['text'].isna().all():
        if 'vlm_text' in df.columns:
            df['text'] = df['vlm_text'].fillna('')
        elif 'ocr_text' in df.columns:
            df['text'] = df['ocr_text'].fillna('')
        else:
            df['text'] = ''
    
    # Fill NaN values
    df['text'] = df['text'].fillna('')
    df['type'] = df['type'].fillna('text')
    
    print(f"  Loaded {len(df)} layouts")
    print(f"  Type distribution: {df['type'].value_counts().to_dict()}")
    
    return df



def main():
    parser = argparse.ArgumentParser(description="Test ColBERT model with type embeddings")
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (.pt file). Required unless --use_passthrough is set.')
    parser.add_argument('--use_passthrough', action='store_true',
                        help='Use pass-through model (same structure as trained model but passes through ColBERT unchanged). Useful for testing/validation.')
    parser.add_argument('--base_model_path', type=str, default='colbert-ir/colbertv2.0',
                        help='Path to base ColBERT model')
    parser.add_argument('--architecture', type=str, default='doc_type_only',
                        choices=list(ARCHITECTURES.keys()),
                        help='Adapter architecture (must match training)')
    parser.add_argument('--type_embed_dim', type=int, default=32,
                        help='Dimension of type embeddings (must match training)')
    parser.add_argument('--layouts_parquet', type=str, default='dataset/MMDocIR_layouts.parquet',
                        help='Path to test layouts parquet file')
    parser.add_argument('--annotations_jsonl', type=str, default='dataset/MMDocIR_annotations.jsonl',
                        help='Path to test annotations JSONL file')
    parser.add_argument('--encode_path', type=str, default='encode',
                        help='Path to directory with pre-encoded ColBERT files (for indices)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for encoding')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cuda, mps, cpu)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.use_passthrough and args.model_path is None:
        parser.error("Either --model_path must be provided or --use_passthrough must be set")
    
    # Determine device - prefer MPS on Mac
    if args.device == 'auto':
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # =========================================================================
    # Step 1: Load pre-encoded ColBERT embeddings + indices (for query-layout mapping)
    # =========================================================================
    print("\n" + "="*60)
    print("Step 1: Loading pre-encoded ColBERT embeddings and indices")
    print("="*60)
    
    query_pkl_path = f"{args.encode_path}/encoded_query_ColBERT.pkl"
    layout_pkl_path = f"{args.encode_path}/encoded_layout_ColBERT.pkl"
    
    print(f"Loading queries from {query_pkl_path}...")
    with open(query_pkl_path, "rb") as f:
        encoded_query, query_indices = pickle.load(f)
    print(f"  Loaded {len(query_indices)} query indices")
    print(f"  Query indices format: {query_indices[0] if query_indices else 'empty'}")
    
    print(f"Loading layouts from {layout_pkl_path}...")
    with open(layout_pkl_path, "rb") as f:
        encoded_layout, layout_indices = pickle.load(f)
    print(f"  Loaded {len(layout_indices)} layout indices")
    print(f"  Layout indices format: {layout_indices[0] if layout_indices else 'empty'}")
    
    # =========================================================================
    # Step 2: Load ground truth from annotations
    # =========================================================================
    print("\n" + "="*60)
    print("Step 2: Loading ground truth from annotations")
    print("="*60)
    
    gt_list = []
    queries = []
    for line in open(args.annotations_jsonl, 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        for qa in item["questions"]:
            qa["domain"] = item["domain"]
            queries.append(qa["Q"])
            gt_list.append(qa)
    
    print(f"  Loaded {len(queries)} test queries")
    print(f"  Loaded {len(gt_list)} ground truth items")
    
    use_gpu = device.type in ['cuda', 'mps']
    
    # =========================================================================
    # Step 3: Choose embeddings source
    # =========================================================================
    print("\n" + "="*60)
    if args.use_passthrough:
        # Baseline: use the SAME pre-encoded ColBERT embeddings as in encode.py/search.py
        print("Step 3: Using pre-encoded ColBERT embeddings (baseline, no re-encoding)")
        print("="*60)
        encoded_queries = encoded_query
        encoded_layouts = encoded_layout
        model = None
    else:
        # Train-time model: apply adapter on top of precomputed ColBERT embeddings
        print("Step 3: Loading trained ColBERT with type embeddings (adapter on top of precomputed embeddings)")
        print("="*60)
        model, tokenizer, query_token_id, doc_token_id = load_trained_model(
            args.model_path, args.base_model_path, architecture=args.architecture,
            type_embed_dim=args.type_embed_dim, dropout=0.0, device=device
        )
        
        # Apply query adapter on precomputed query embeddings (if architecture has one)
        encoded_queries = apply_query_adapter_on_precomputed(
            model, encoded_query, device
        )
        
        # Load test layouts for type information (order matches encoded_layout_ColBERT)
        layouts_df = load_test_layouts(args.layouts_parquet)
        layout_types = layouts_df['type'].fillna('text').tolist()
        
        # Apply doc adapter on top of precomputed document embeddings
        print("\n" + "="*60)
        print("Step 4: Applying adapter on precomputed ColBERT document embeddings")
        print("="*60)
        encoded_layouts = apply_doc_adapter_on_precomputed(
            model, encoded_layout, layout_types, device
        )
    
    # =========================================================================
    # Step 5: Evaluate ColBERT model
    # =========================================================================
    print("\n" + "="*60)
    if args.use_passthrough:
        print("Step 5: Evaluating pass-through ColBERT model (baseline)")
    else:
        print("Step 5: Evaluating trained ColBERT with type embeddings")
    print("="*60)
    
    # Build gt_list with new scores
    gt_list_with_types = []
    for line in open(args.annotations_jsonl, 'r', encoding="utf-8"):
        item = json.loads(line.strip())
        for qa in item["questions"]:
            qa["domain"] = item["domain"]
            gt_list_with_types.append(qa.copy())
    
    for idx, qi in enumerate(tqdm(query_indices, desc="Scoring with types")):
        # query_indices format: (query_id, start_pid, end_pid, start_lid, end_lid)
        if len(qi) == 5:
            qid, start_pid, end_pid, start_lid, end_lid = qi
        else:
            qid = idx
            _, _, start_lid, end_lid = qi[:4] if len(qi) >= 4 else (0, 0, 0, len(encoded_layouts)-1)
        
        query_vec = encoded_queries[qid] if qid < len(encoded_queries) else encoded_queries[idx]
        layout_vecs = encoded_layouts[start_lid:end_lid + 1]
        
        if len(layout_vecs) > 0:
            layout_vecs_pad, masks_layout = pad_tok_len(layout_vecs)
            scores_layout = colbert_score(query_vec, layout_vecs_pad, masks_layout, use_gpu=use_gpu)
            gt_list_with_types[idx]["scores_layout"] = scores_layout.tolist()
            gt_list_with_types[idx]["layout_indices"] = layout_indices[start_lid:end_lid + 1]
        else:
            gt_list_with_types[idx]["scores_layout"] = []
            gt_list_with_types[idx]["layout_indices"] = []
    
    if args.use_passthrough:
        model_name = "ColBERT_passthrough"
        print("\nPass-through ColBERT Results (baseline - should match original ColBERT):")
    else:
        model_name = "ColBERT_with_types"
        print("\nColBERT with Type Embeddings Results:")
    
    evaluate_layout(gt_list_with_types, model_name=model_name, topk=1, metric="recall")
    evaluate_layout(gt_list_with_types, model_name=model_name, topk=5, metric="recall")
    evaluate_layout(gt_list_with_types, model_name=model_name, topk=10, metric="recall")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)
    print("Testing Complete!")
    print("="*60)
    if args.use_passthrough:
        print("Pass-through ColBERT evaluation finished (baseline test).")
        print("This model should match original ColBERT performance.")
    else:
        print("ColBERT with type embeddings evaluation finished.")
    print("See results above for detailed metrics.")


if __name__ == "__main__":
    main()

