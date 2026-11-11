# TinyMind v3 - the 500M upgrade, learned from the train-50m CI logs

| 50m failure (run #15) | v3 fix |
|---|---|
| greedy loops: rep 0.515, looping 62.5% (sampled diagnostic healthy) | >=20 tok/param natural text; greedy gated + sampled diagnostic |
| EOS rate 0.000 | EOS gated from stage4; termination needs tokens to learn |
| 32-example source at 3,156 passes (warning only) | --max-source-passes hard-fails at start-up; stage1 99.99% natural |
| val 0.94 on 859 memorized tokens vs 5.41 nats real text | val_records 20000; bpb primary |
| 11.1M tokens = 0.22 tok/param @ 341.7 tok/s | 11.2B = 22.3 tok/param; GPU substrate |

Model: 501,090,560 exact (d1280/L26/20h/4kv/SwiGLU3584/32k BPE/ctx2048/tied/dropout 0).
S0 (tinymind_gpu): torch mirror + parity (fwd <1e-4, grad cos >0.999; conventions
verified vs oracle at 2.2e-6), trainer with reference LRSchedule/AdamW/normalizer
semantics, exact resume, .tm export. gpu-parity.yml runs the contract on CI.
S1 next: reference TokenizedDataset wiring, DDP/FSDP, long-ctx variant, DPO, INT4+distill.
Substrate: NumPy 500M ~34 tok/s (10B tokens = 9.2 years); 8xA100 bf16 ~10-14 h.
