# src/contact_prediction_rl/pretraining/__init__.py

"""Fine-tuning and benchmarking of pretrained policies (VLAs) on demonstration data.

The heavy dependencies (lerobot for SmolVLA and pi0, peft for LoRA, openpi) are optional and imported only when a
loader needs them, so importing this package never affects the PPO code.
"""
