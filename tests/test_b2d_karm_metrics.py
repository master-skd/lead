import numpy as np

from scripts.p4.eval_b2d_karm_dynamic_safety import summarize_karm


def test_karm_summary_counts_safe_rescue_without_reranking_safe_winner():
    result = summarize_karm(
        confidence_logits=np.asarray([[3.0, 1.0, -1.0], [3.0, 1.0, -1.0]]),
        valid=np.asarray([[1, 1, 1], [1, 1, 0]], dtype=bool),
        collision=np.asarray([[0, 0, 1], [1, 0, 0]], dtype=bool),
        ttc_s=np.asarray([[np.inf, np.inf, 1.0], [0.5, np.inf, np.inf]]),
        ade=np.asarray([[0.1, 1.0, 2.0], [0.1, 1.0, 9.0]]),
    )
    assert result["confidence_winner_collision_rate"] == 0.5
    assert result["rescue_rate_given_unsafe_winner"] == 1.0
    assert result["oracle_safety_switch_rate"] == 0.5
    assert result["confidence_winner_is_expert_wta_rate"] == 1.0
    assert result["ade_confidence_winner"] == 0.1
    assert result["ade_oracle_safe_confidence"] == 0.55
