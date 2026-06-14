import torch

from veomni.data.data_collator import MakeMicroBatchCollator


def _identity_collator(features):
    return features


def test_make_microbatch_collator_keeps_all_coverage_variants():
    collator = MakeMicroBatchCollator(num_micro_batch=2, internal_data_collator=_identity_collator)

    features = [
        [
            {"coverage_variant_idx": torch.tensor(0), "value": torch.tensor(10)},
            {"coverage_variant_idx": torch.tensor(1), "value": torch.tensor(11)},
        ],
        [
            {"coverage_variant_idx": torch.tensor(0), "value": torch.tensor(20)},
            {"coverage_variant_idx": torch.tensor(1), "value": torch.tensor(21)},
        ],
    ]

    micro_batches = collator(features)
    assert len(micro_batches) == 2

    flattened = [sample for micro_batch in micro_batches for sample in micro_batch]
    assert len(flattened) == 4

    values = sorted(int(sample["value"]) for sample in flattened)
    assert values == [10, 11, 20, 21]


def test_make_microbatch_collator_supports_plain_features():
    collator = MakeMicroBatchCollator(num_micro_batch=2, internal_data_collator=_identity_collator)

    features = [
        {"value": torch.tensor(1)},
        {"value": torch.tensor(2)},
        {"value": torch.tensor(3)},
        {"value": torch.tensor(4)},
    ]

    micro_batches = collator(features)
    assert len(micro_batches) == 2
    assert [len(mb) for mb in micro_batches] == [2, 2]

