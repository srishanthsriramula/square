"""Register verified Mondrian crop improvements into the official registry."""

import math
import numpy as np
from src.scoring.registry import PackingRegistry
from src.geometry.sat import validate_packing_sat, compute_half_extents, compute_enclosing_side
from src.validation.validator import validate_submission

def main():
    reg = PackingRegistry()
    initial_score = reg.get_score()
    print(f"Initial score: {initial_score:.14f}")
    
    # 1. Update N=340 from N=417
    c417 = reg.get_packing(417)
    s417 = reg.get_side(417)
    h417 = compute_half_extents(c417[:, 2])
    target_340 = s417 - 2.0
    xmax = np.max(c417[:, 0] + h417)
    ymin = np.min(c417[:, 1] - h417)
    in_340 = np.where((c417[:, 0] - h417 >= xmax - target_340 - 1e-9) & (c417[:, 1] + h417 <= ymin + target_340 + 1e-9))[0]
    c340 = c417[in_340].copy()
    h340 = h417[in_340]
    c340[:, 0] -= np.min(c340[:, 0] - h340)
    c340[:, 1] -= np.min(c340[:, 1] - h340)
    val_340 = validate_packing_sat(c340, eps=1e-10)
    assert val_340["valid"], "N=340 failed SAT"
    reg.register_candidate(340, c340, source="mondrian_crop_from_n417")
    print(f"Registered N=340: side={val_340['side']:.10f}")

    # 2. Update N=304 from newly updated N=340
    target_304 = val_340["side"] - 1.0
    xmin340 = np.min(c340[:, 0] - h340)
    ymin340 = np.min(c340[:, 1] - h340)
    in_304 = np.where((c340[:, 0] + h340 <= xmin340 + target_304 + 1e-9) & (c340[:, 1] + h340 <= ymin340 + target_304 + 1e-9))[0]
    c304 = c340[in_304].copy()
    h304 = h340[in_304]
    c304[:, 0] -= np.min(c304[:, 0] - h304)
    c304[:, 1] -= np.min(c304[:, 1] - h304)
    val_304 = validate_packing_sat(c304, eps=1e-10)
    assert val_304["valid"], "N=304 failed SAT"
    reg.register_candidate(304, c304, source="mondrian_cascade_crop_from_n340")
    print(f"Registered N=304: side={val_304['side']:.10f}")

    # 3. Update N=463 from N=506
    c506 = reg.get_packing(506)
    s506 = reg.get_side(506)
    h506 = compute_half_extents(c506[:, 2])
    target_463 = s506 - 1.0
    xmin = np.min(c506[:, 0] - h506)
    ymin = np.min(c506[:, 1] - h506)
    in_463 = np.where((c506[:, 0] + h506 <= xmin + target_463 + 1e-9) & (c506[:, 1] + h506 <= ymin + target_463 + 1e-9))[0]
    c463 = c506[in_463].copy()
    h463 = h506[in_463]
    c463[:, 0] -= np.min(c463[:, 0] - h463)
    c463[:, 1] -= np.min(c463[:, 1] - h463)
    val_463 = validate_packing_sat(c463, eps=1e-10)
    assert val_463["valid"], "N=463 failed SAT"
    reg.register_candidate(463, c463, source="mondrian_crop_from_n506")
    print(f"Registered N=463: side={val_463['side']:.10f}")

    # 4. Update N=554 from N=602
    c602 = reg.get_packing(602)
    s602 = reg.get_side(602)
    h602 = compute_half_extents(c602[:, 2])
    target_554 = s602 - 1.0
    xmin = np.min(c602[:, 0] - h602)
    ymin = np.min(c602[:, 1] - h602)
    in_554 = np.where((c602[:, 0] + h602 <= xmin + target_554 + 1e-9) & (c602[:, 1] + h602 <= ymin + target_554 + 1e-9))[0]
    c554 = c602[in_554].copy()
    h554 = h602[in_554]
    c554[:, 0] -= np.min(c554[:, 0] - h554)
    c554[:, 1] -= np.min(c554[:, 1] - h554)
    val_554 = validate_packing_sat(c554, eps=1e-10)
    assert val_554["valid"], "N=554 failed SAT"
    reg.register_candidate(554, c554, source="mondrian_crop_from_n602")
    print(f"Registered N=554: side={val_554['side']:.10f}")

    # 5. Update N=592 from N=641 (k=1, LL)
    c641 = reg.get_packing(641)
    s641 = reg.get_side(641)
    h641 = compute_half_extents(c641[:, 2])
    target_592 = s641 - 1.0
    xmin = np.min(c641[:, 0] - h641)
    ymin = np.min(c641[:, 1] - h641)
    in_592 = np.where((c641[:, 0] + h641 <= xmin + target_592 + 1e-9) & (c641[:, 1] + h641 <= ymin + target_592 + 1e-9))[0]
    c592 = c641[in_592].copy()
    h592 = h641[in_592]
    c592[:, 0] -= np.min(c592[:, 0] - h592)
    c592[:, 1] -= np.min(c592[:, 1] - h592)
    val_592 = validate_packing_sat(c592, eps=1e-10)
    assert val_592["valid"], "N=592 failed SAT"
    reg.register_candidate(592, c592, source="mondrian_crop_from_n641")
    print(f"Registered N=592: side={val_592['side']:.10f}")

    # 6. Update N=591 from N=641 (k=1, LR)
    target_591 = s641 - 1.0
    xmax = np.max(c641[:, 0] + h641)
    ymin = np.min(c641[:, 1] - h641)
    in_591 = np.where((c641[:, 0] - h641 >= xmax - target_591 - 1e-9) & (c641[:, 1] + h641 <= ymin + target_591 + 1e-9))[0]
    c591 = c641[in_591].copy()
    h591 = h641[in_591]
    c591[:, 0] -= np.min(c591[:, 0] - h591)
    c591[:, 1] -= np.min(c591[:, 1] - h591)
    val_591 = validate_packing_sat(c591, eps=1e-10)
    assert val_591["valid"], "N=591 failed SAT"
    reg.register_candidate(591, c591, source="mondrian_crop_from_n641_lr")
    print(f"Registered N=591: side={val_591['side']:.10f}")

    # 7. Update N=498 from N=641 (k=3, LL)
    target_498 = s641 - 3.0
    xmin = np.min(c641[:, 0] - h641)
    ymin = np.min(c641[:, 1] - h641)
    in_498 = np.where((c641[:, 0] + h641 <= xmin + target_498 + 1e-9) & (c641[:, 1] + h641 <= ymin + target_498 + 1e-9))[0]
    c498 = c641[in_498].copy()
    h498 = h641[in_498]
    c498[:, 0] -= np.min(c498[:, 0] - h498)
    c498[:, 1] -= np.min(c498[:, 1] - h498)
    val_498 = validate_packing_sat(c498, eps=1e-10)
    assert val_498["valid"], "N=498 failed SAT"
    reg.register_candidate(498, c498, source="mondrian_crop_from_n641_k3")
    print(f"Registered N=498: side={val_498['side']:.10f}")

    # 8. Update N=647 from N=699 (k=1, LL)
    c699 = reg.get_packing(699)
    s699 = reg.get_side(699)
    h699 = compute_half_extents(c699[:, 2])
    target_647 = s699 - 1.0
    xmin = np.min(c699[:, 0] - h699)
    ymin = np.min(c699[:, 1] - h699)
    in_647 = np.where((c699[:, 0] + h699 <= xmin + target_647 + 1e-9) & (c699[:, 1] + h699 <= ymin + target_647 + 1e-9))[0]
    c647 = c699[in_647].copy()
    h647 = h699[in_647]
    c647[:, 0] -= np.min(c647[:, 0] - h647)
    c647[:, 1] -= np.min(c647[:, 1] - h647)
    val_647 = validate_packing_sat(c647, eps=1e-10)
    assert val_647["valid"], "N=647 failed SAT"
    reg.register_candidate(647, c647, source="mondrian_crop_from_n699")
    print(f"Registered N=647: side={val_647['side']:.10f}")

    new_score = reg.get_score()
    total_gain = initial_score - new_score
    print(f"\nAll 8 candidates successfully registered!")
    print(f"New total score: {new_score:.14f}")
    print(f"Verified score reduction: {total_gain:.14f}")

    # Save to submissions/best_submission.csv
    reg.save_submission()
    print("Saved submission to submissions/best_submission.csv")

    # Run full independent SAT validation on the entire submission
    print("\nRunning full independent SAT validation on all 1,000 instances...")
    res = validate_submission("submissions/best_submission.csv", eps=1e-10)
    print(f"Full Submission Validation: valid={res['valid']}, total_score={res['total_score']:.14f}")
    assert res["valid"], "Submission validation failed!"
    print("SUCCESS: 1,000 / 1,000 instances verified with ZERO overlaps!")

if __name__ == "__main__":
    main()
