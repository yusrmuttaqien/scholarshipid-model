"""
Evaluator untuk Two-Tower retrieval: Recall@K, NDCG@K, MRR.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict


class Evaluator:
    """
    Hitung retrieval metrics dengan brute-force dot product.

    Usage:
        evaluator = Evaluator(k=5)
        metrics = evaluator.compute_metrics(
            df=val_df,
            student_tower=student_tower,
            scholarship_tower=scholarship_tower,
            stu_struct=stu_struct,
            sch_struct=sch_struct,
            stu_text_emb=stu_text_emb,
            sch_text_emb=sch_text_emb,
            stu_id_to_idx=stu_id_to_idx,
        )
    """

    def __init__(self, k: int = 5):
        self.k = k

    def compute_metrics(
        self,
        df: pd.DataFrame,
        student_tower: tf.keras.Model,
        scholarship_tower: tf.keras.Model,
        stu_struct: np.ndarray,
        sch_struct: np.ndarray,
        stu_text_emb: np.ndarray,
        sch_text_emb: np.ndarray,
        stu_id_to_idx: dict,
        sch_ids: list,
    ) -> Dict[str, float]:
        """
        Returns:
            {f"recall@{k}": float, f"ndcg@{k}": float, "mrr": float}
        """
        k = self.k

        # Pre-compute semua scholarship embeddings
        sch_feat_all = np.concatenate([sch_struct, sch_text_emb], axis=1)
        sch_emb_all  = scholarship_tower(sch_feat_all, training=False).numpy()  # (N_sch, 128)

        # Mapping scholarship_id → row index
        sch_id_to_idx_local = {sid: i for i, sid in enumerate(sch_ids)}

        # Group positives per student
        student_groups = df.groupby("student_id")["scholarship_id"].apply(set).to_dict()

        recall_vals, ndcg_vals, mrr_vals = [], [], []

        for stu_id, pos_sch_ids in student_groups.items():
            if stu_id not in stu_id_to_idx:
                continue
            idx      = stu_id_to_idx[stu_id]
            stu_feat = np.concatenate([stu_struct[[idx]], stu_text_emb[[idx]]], axis=1)
            stu_emb  = student_tower(stu_feat, training=False).numpy()[0]  # (128,)

            # Brute-force dot product vs semua 43 scholarships
            scores    = sch_emb_all @ stu_emb               # (N_sch,)
            top_k_idx = np.argsort(-scores)[:k]
            retrieved = [sch_ids[i] for i in top_k_idx]

            hits = set(retrieved) & pos_sch_ids
            recall_vals.append(min(len(hits) / max(len(pos_sch_ids), 1), 1.0))

            # NDCG@K
            dcg, idcg = 0.0, 0.0
            for rank, sid in enumerate(retrieved, start=1):
                if sid in pos_sch_ids:
                    dcg += 1.0 / np.log2(rank + 1)
            for rank in range(1, min(len(pos_sch_ids), k) + 1):
                idcg += 1.0 / np.log2(rank + 1)
            ndcg_vals.append(dcg / idcg if idcg > 0 else 0.0)

            # MRR
            mrr = 0.0
            for rank, sid in enumerate(retrieved, start=1):
                if sid in pos_sch_ids:
                    mrr = 1.0 / rank
                    break
            mrr_vals.append(mrr)

        return {
            f"recall@{k}": float(np.mean(recall_vals)),
            f"ndcg@{k}":   float(np.mean(ndcg_vals)),
            "mrr":         float(np.mean(mrr_vals)),
        }
