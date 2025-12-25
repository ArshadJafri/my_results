# Training Analysis: train_v4_7471.out

**Run ID:** train_v4_7471
**Date:** December 6, 2025 (14:56:52 - 02:37:37, ~12 hours)
**Corresponding Loss Plot:** loss_plot_0.0002_8870_2_2_10_0.01_100.jpg

---

## Training Configuration

### Command Line Parameters
```bash
Finetuning_V5.py 0.0002 8870 2 2 10 0.01 100
```

**Parsed as:**
- Learning Rate: 0.0002
- Max Sequence Length: 8870
- Train Batch Size: 2
- Eval Batch Size: 2
- Gradient Accumulation Steps: 10
- Weight Decay: 0.01
- Number of Epochs: 100

### Hardware Setup
- **GPUs:** 8 (CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7)
- **Framework:** DeepSpeed distributed training
- **Effective Batch Size per GPU:** 20 (2 × 10 gradient accumulation)
- **Total Effective Batch Size:** 160 (20 × 8 GPUs)

### Model Configuration
- **Base Model:** marcelbinz/Llama-3.1-Centaur-70B-adapter
- **LoRA r:** 8
- **LoRA alpha:** 8
- **LoRA dropout:** 0 (no dropout!)
- **Quantization:** 4-bit loading
- **Precision:** BFloat16

### Training Strategy
- **Optimizer:** adamw_8bit
- **LR Scheduler:** constant (config) - but shows decay in practice
- **Warmup Steps:** 0
- **Evaluation Strategy:** Every 10 steps
- **Save Strategy:** Every 100 steps
- **Gradient Checkpointing:** Enabled

---

## Complete Training Timeline

### Epoch-by-Epoch Progression

| Epoch | Train Loss | Eval Loss | Learning Rate | Grad Norm | Observation |
|-------|------------|-----------|---------------|-----------|-------------|
| 0.83  | 0.1660    | -         | 0.00000      | 0.5529    | Initial training |
| 1.0   | 0.1671    | -         | 0.00006021   | 0.4522    | Warmup phase |
| 5.0   | 0.1695    | **0.1551** | 0.00020     | 0.0772    | First eval |
| 10.0  | 0.1130    | **0.1447** | 0.00019053  | 0.0870    | Improving |
| 15.0  | 0.1282    | **0.1376** | 0.00018     | 0.0854    | **BEST EVAL LOSS** |
| 20.0  | 0.1224    | **0.1387** | 0.00016947  | 0.1120    | Eval starts rising |
| 25.0  | 0.1202    | **0.1429** | 0.00015895  | 0.2237    | Clear divergence |
| 30.0  | 0.0781    | **0.1544** | 0.00014842  | 0.2991    | Train drops, eval rises |
| 35.0  | 0.0288    | **0.2098** | 0.00013789  | 0.2316    | Severe overfitting |
| 40.0  | 0.0081    | **0.2641** | 0.00012737  | 0.1516    | Eval exploding |
| 45.0  | 0.0044    | **0.3354** | 0.00011684  | 0.1861    | Catastrophic gap |
| 50.0  | 0.0035    | **0.3899** | 0.00010632  | 0.1043    | Train ~0, eval ruined |
| 55.0  | 0.0007    | **0.4417** | 0.00009579  | 0.0281    | Near-zero training loss |
| 60.0  | 0.0001    | **0.4581** | 0.00008526  | 0.0273    | Memorization complete |
| 65.0  | 0.0001    | **0.4376** | 0.00007474  | 0.0466    | Eval plateaus |
| 70.0  | 0.0001    | **0.4563** | 0.00006421  | 0.0204    | Minimal gradients |
| 75.0  | 0.0000    | **0.4700** | 0.00005368  | 0.0009    | Training loss = 0 |
| 80.0  | 0.0000    | **0.4825** | 0.00004316  | 0.0006    | Perfect memorization |
| 85.0  | 0.0000    | **0.4875** | 0.00003263  | 0.0012    | Vanishing gradients |
| 90.0  | 0.0000    | **0.4911** | 0.00002211  | 0.0003    | No learning occurring |
| 95.0  | 0.0000    | **0.4902** | 0.00001158  | 0.0002    | Stagnant |
| 100.0 | 0.0000    | **0.4918** | 0.00000105  | 0.0003    | Final: complete failure |

---

## What's Happening: Detailed Phase Analysis

### Phase 1: Initial Learning (Epochs 1-15)
**Duration:** ~1.5 hours
**Train Loss:** 0.167 → 0.128
**Eval Loss:** 0.155 → 0.138

**What's occurring:**
- Model is genuinely learning patterns from the PTSD cognition tasks
- Both training and evaluation losses decrease together
- Gradient norms are healthy (0.05-0.20), indicating active learning
- Learning rate is at target value (0.0002) with minimal decay
- The model is finding generalizable representations

**Evidence of healthy training:**
- Eval loss tracks training loss closely
- No divergence between train/eval curves
- Gradient norms stable and reasonable
- Loss decreasing at steady pace

**Epoch 15 is THE optimal checkpoint** - eval_loss = 0.1376 is the best the model ever achieves.

---

### Phase 2: Divergence Begins (Epochs 15-25)
**Duration:** ~1 hour
**Train Loss:** 0.128 → 0.120
**Eval Loss:** 0.138 → 0.143

**What's occurring:**
- Training loss continues decreasing (model fitting training data better)
- Evaluation loss STOPS decreasing and begins increasing
- This is the critical warning signal that overfitting has begun
- The model starts learning training-specific patterns that don't generalize
- Gradient norms increase (0.11 → 0.22) showing aggressive fitting

**Why this happens:**
- Model has learned the main patterns (done by epoch 15)
- Now it's starting to memorize training-specific noise
- Without dropout, LoRA adapters have no regularization
- No early stopping means training blindly continues

**This is where training should have stopped** - any continuation past epoch 20 is detrimental.

---

### Phase 3: Severe Overfitting (Epochs 25-45)
**Duration:** ~2 hours
**Train Loss:** 0.120 → 0.004
**Eval Loss:** 0.143 → 0.335

**What's occurring:**
- Training loss plummets from 0.12 to nearly zero
- Evaluation loss EXPLODES from 0.14 to 0.34 (2.4x worse!)
- Model is rapidly memorizing individual training examples
- The gap between train/eval is enormous and growing
- Gradient norms fluctuate wildly (0.11-0.37) as model overfits

**Mechanism of overfitting:**
- LoRA adapters are learning to perfectly reproduce training outputs
- With dropout=0, there's no stochastic regularization
- The model finds "shortcuts" - memorizing exact input→output mappings
- These shortcuts are specific to training data and fail on new examples
- Weight decay (0.01) is insufficient to prevent this memorization

**Evidence of memorization:**
- By epoch 40, train_loss = 0.0081 (almost perfect)
- But eval_loss = 0.2641 (nearly 2x the optimal 0.138)
- The model "knows" training data but can't generalize
- It's like a student who memorized answers but didn't learn concepts

---

### Phase 4: Complete Memorization (Epochs 45-70)
**Duration:** ~2.5 hours
**Train Loss:** 0.004 → 0.0001
**Eval Loss:** 0.335 → 0.456

**What's occurring:**
- Training loss reaches effectively zero
- Model outputs PERFECT predictions on training set
- Evaluation loss continues worsening, approaching 0.5
- Gradient norms start collapsing (0.186 → 0.020)
- Learning rate decaying (0.00012 → 0.00006)

**What "memorization" means:**
- For every training example, model predicts exact expected tokens
- Loss = 0.0044 at epoch 45 means near-perfect token prediction
- This is NOT learning - it's storing a lookup table
- Like memorizing a dictionary without understanding language
- Model has lost all ability to handle novel inputs

**Why gradients are collapsing:**
- With training loss near zero, gradients become tiny
- Model is at a "perfect" fit (for training data), so no gradient signal
- Vanishing gradients mean learning has effectively stopped
- The model can't improve further even if it wanted to

**The evaluation loss explosion:**
- Eval loss = 0.456 (3.3x worse than optimal!)
- Model has become WORSE than it was at epoch 15
- This is the cost of memorization - destroyed generalization
- Every pattern learned after epoch 15 hurts real-world performance

---

### Phase 5: Dead Training (Epochs 70-100)
**Duration:** ~6 hours of wasted computation
**Train Loss:** 0.0001 → 0.0000 (literally zero)
**Eval Loss:** 0.456 → 0.492 (plateaus)

**What's occurring:**
- Training loss is exactly 0.0000 (machine precision zero)
- Evaluation loss plateaus around 0.49 and stops changing
- Gradient norms collapse to 0.0002-0.0009 (vanishing)
- Learning rate decays to ~0.00001 (nearly zero)
- The training process is completely stagnant

**Why training is "dead":**
- With loss=0, there's no error signal to learn from
- Gradients are 1000x smaller than early training
- Learning rate is 200x smaller than initial value
- Model parameters barely changing (0.0003 gradient norms)
- Evaluation metrics show no improvement for 30 epochs

**What's happening to the model:**
- It has perfectly overfit to training data
- Cannot be improved further on training set (already perfect)
- Cannot improve on eval set (learning capacity exhausted)
- The model is "stuck" in a memorization state
- No amount of additional training will help

**The computational waste:**
- 30 epochs (70-100) accomplish literally nothing
- 6+ hours of 8-GPU training with zero benefit
- Electricity, compute time, researcher attention - all wasted
- This is why early stopping is critical

**Final state (epoch 100):**
- Train loss: 0.0000 (perfect memorization)
- Eval loss: 0.4918 (3.6x worse than optimal)
- The model is completely unusable for real-world inference
- Would need to discard and start over

---

## The Learning Rate Mystery

### Configuration vs Reality

**Configuration says:**
```python
lr_scheduler_type="constant"
```

**Logs show:**
| Epoch | Learning Rate | Expected (constant) | Actual Behavior |
|-------|---------------|---------------------|-----------------|
| 5     | 0.00020      | 0.0002             | Matches ✓ |
| 10    | 0.00019053   | 0.0002             | Decaying! |
| 25    | 0.00015895   | 0.0002             | Linear decay |
| 50    | 0.00010632   | 0.0002             | Half original |
| 75    | 0.00005368   | 0.0002             | 1/4 original |
| 100   | 0.00000105   | 0.0002             | ~Zero! |

**Analysis:**
The learning rate is following a **linear decay schedule** from 0.0002 → 0, despite configuration specifying "constant". This is a clear discrepancy.

**Why this matters:**
- If LR stayed at 0.0002, overfitting would be EVEN WORSE
- The unintended decay actually limited the damage
- With constant high LR, model would memorize faster and more completely
- This "bug" accidentally provided some regularization

**Possible causes:**

1. **DeepSpeed Override**
   - The `ds_config.json` file may specify its own scheduler
   - DeepSpeed config takes precedence over TrainingArguments
   - Common issue when using DeepSpeed with HuggingFace Trainer

2. **Transformers Default Behavior**
   - Even "constant" scheduler may have subtle decay
   - Some schedulers implement minimal decay for stability
   - Could be interaction with warmup (even though warmup_steps=0)

3. **Optimizer Settings**
   - adamw_8bit optimizer might have built-in LR adjustment
   - Though this would be unusual behavior

**Recommendation:** Check the `ds_config.json` file contents to identify the actual scheduler being used.

---

## Why Overfitting Happened: Root Causes

### 1. No Dropout (Primary Cause)
```python
lora_dropout=0
```

**Impact:**
- LoRA adapters have zero stochastic regularization
- Every training example contributes deterministically
- Adapters can freely memorize without constraint
- Standard practice is lora_dropout=0.05 to 0.1

**How dropout would help:**
- Randomly disables adapter neurons during training
- Forces model to learn redundant representations
- Prevents reliance on specific neuron combinations
- Makes memorization much harder

### 2. Too Many Epochs (Primary Cause)
```python
num_train_epochs=100
```

**Impact:**
- Model sees each example 100+ times
- Ample opportunity to memorize training set
- Far beyond what's needed for convergence (optimal at epoch 15)
- 85 epochs of harmful training after optimal point

**How this causes overfitting:**
- First 15 epochs: learning patterns
- Epochs 15-30: starting to memorize
- Epochs 30-70: memorizing aggressively
- Epochs 70-100: perfect memorization achieved

### 3. No Early Stopping (Primary Cause)
**Status:** Not implemented

**Impact:**
- Training continues blindly regardless of eval performance
- No automatic detection of overfitting
- System follows epoch count even when harmful
- Wastes compute on training that hurts the model

**What early stopping would do:**
- Monitor eval_loss at each evaluation
- Stop when eval_loss increases for N consecutive checks
- Would have stopped around epoch 20-25
- Automatically saves best checkpoint

### 4. High Learning Rate (Contributing Factor)
```python
learning_rate=0.0002
```

**Impact:**
- 2x higher than successful runs (0.0001)
- Allows rapid fitting to training data
- Combined with many epochs, enables fast memorization
- Higher LR = faster overfitting

**Comparison:**
- Successful runs: lr=0.0001, converge smoothly
- This run: lr=0.0002, overfits aggressively

### 5. Modest Weight Decay (Contributing Factor)
```python
weight_decay=0.01
```

**Impact:**
- Provides some L2 regularization but not enough
- Against 100 epochs of training, 0.01 is weak
- Would need 0.05-0.1 to counteract this training duration

**How weight decay helps:**
- Penalizes large parameter values
- Encourages simpler, more generalizable models
- But strength must match training intensity

### 6. Small Per-GPU Batch Size (Minor Factor)
```python
per_device_train_batch_size=2
```

**Impact:**
- Creates noisier gradient estimates
- Noisy gradients can lead to overfitting on specific examples
- Though gradient accumulation (×10) helps smooth this

---

## Key Metrics and Their Meaning

### Training Loss = 0.0000
**First occurrence:** Epoch 72
**Meaning:**
- Model outputs exact correct tokens for all training examples
- Prediction probability for correct tokens ≈ 100%
- This is complete memorization
- Model has become a lookup table for training data

**Why this is bad:**
- Real-world data won't be in the training set
- Model has no understanding, just memorized mappings
- Generalization capability is zero

### Eval Loss = 0.4918 (Final)
**Comparison to optimal:** 3.6x worse (optimal was 0.138)
**Meaning:**
- Model performs terribly on unseen data
- Predictions are highly uncertain (high loss = low confidence)
- The model is confused by anything not in training set

**What this means for inference:**
- Poor quality responses to new PTSD task prompts
- High perplexity (uncertainty in predictions)
- Would generate inconsistent or incorrect outputs

### Gradient Norm Collapse
**Early training:** 0.10-0.60
**Late training:** 0.0002-0.0009

**Meaning:**
- Gradient norms measure "how much to update parameters"
- Large norms = active learning, small norms = stagnant
- 1000x decrease = learning effectively stopped
- Model can't improve even if we wanted it to

### Learning Rate Decay to ~0
**Start:** 0.0002
**End:** 0.00000105

**Meaning:**
- Even if gradients existed, updates would be tiny
- LR of 1e-06 means parameters change negligibly
- Combined with vanishing gradients = zero learning
- The training is computationally running but mathematically dead

---

## Evaluation Points and Decisions

Training evaluated at these steps:
- Step 10 (Epoch 5): eval_loss = 0.1551 ✓ Improving
- Step 20 (Epoch 10): eval_loss = 0.1447 ✓ Improving
- Step 30 (Epoch 15): eval_loss = 0.1376 ✓ **BEST - SHOULD STOP HERE**
- Step 40 (Epoch 20): eval_loss = 0.1387 ⚠️ Starting to increase - STOP NOW
- Step 50 (Epoch 25): eval_loss = 0.1429 ⚠️ Clear overfitting - MUST STOP
- Step 60 (Epoch 30): eval_loss = 0.1544 ❌ Getting worse - too late
- Step 70 (Epoch 35): eval_loss = 0.2098 ❌ Severely damaged
- Step 80+ (Epoch 40-100): eval_loss 0.26-0.49 ☠️ Catastrophic

**Optimal stopping point:** Step 30-40 (Epoch 15-20)
**Actual stopping point:** Step 200 (Epoch 100)
**Wasted training:** 160-170 steps (80-85 epochs)

---

## Checkpoints Saved

### checkpoint-100 (Step 100, ~Epoch 50)
**Metrics at this point:**
- Train loss: 0.0035
- Eval loss: ~0.39

**Assessment:**
- Already severely overfit
- Eval loss 3x worse than optimal
- Not recommended for use
- Would produce poor quality outputs

### checkpoint-200 (Step 200, Epoch 100)
**Metrics at this point:**
- Train loss: 0.0000
- Eval loss: 0.4918

**Assessment:**
- Catastrophically overfit
- Completely memorized training data
- Unusable for any real-world task
- Should be discarded

### Missing: Optimal Checkpoint
**Should have saved:** Step 30-40 (Epoch 15-20)
**Metrics:** eval_loss ~0.138
**Status:** Not saved because training didn't stop

This is the critical missing piece - the model's best state was never preserved.

---

## Computational Cost Analysis

### Total Training Time
**Duration:** 14:56:52 → 02:37:37 = ~11 hours 40 minutes
**Hardware:** 8× GPUs running continuously

### Useful Training Time
**Duration:** First ~2 hours (epochs 1-20)
**Percent useful:** ~17% of total time

### Wasted Training Time
**Duration:** ~9 hours 40 minutes (epochs 20-100)
**Percent wasted:** ~83% of total time
**Cost:** 8 GPUs × 9.67 hours = 77 GPU-hours wasted

**What was accomplished in wasted time:**
- Made the model worse (eval loss 0.14 → 0.49)
- Used electricity with no benefit
- Occupied GPUs that could run other experiments
- Delayed research progress

**With early stopping:**
- Would stop at epoch 20 (~2.5 hours)
- Save 75% of compute time
- Achieve better final model
- Run 4× more experiments in same time

---

## What Should Have Happened

### Ideal Training Run

**With proper configuration:**
```python
num_train_epochs=20  # Not 100
lora_dropout=0.1     # Not 0
learning_rate=0.0001 # Not 0.0002
early_stopping_patience=3
```

**Expected timeline:**
- Epochs 1-15: Training and eval loss decrease together
- Epoch 15: Reach optimal eval_loss ~0.13-0.14
- Epochs 16-18: Eval loss plateaus or increases slightly
- Epoch 18-20: Early stopping triggers after 3 checks
- **Training stops automatically**

**Outcomes:**
- Best model saved at epoch 15-18
- Total training time: ~2.5 hours (vs 11.7 hours actual)
- Final eval loss: ~0.14 (vs 0.49 actual)
- Model is usable and generalizes well
- 78% reduction in compute time
- Better final result

---

## Comparison to Successful Runs

### This Run (Overfit)
- LR: 0.0002
- Epochs: 100
- Dropout: 0
- Early stopping: No
- Final eval loss: 0.492
- Status: Unusable

### Successful Run (loss_plot_0.0001_2092_1_1_4_0.01_10.jpg)
- LR: 0.0001 (2× lower)
- Epochs: 10 (10× fewer)
- Dropout: 0 (same issue)
- Early stopping: No (but stopped early anyway)
- Final eval loss: ~0.005
- Status: Excellent

**Key differences:**
- 10× fewer epochs prevented memorization
- 2× lower LR provided gentler updates
- Stopped before overfitting could occur
- Even without early stopping, short training helped

**Lesson:** The combination of high LR + many epochs is toxic for generalization.

---

## Signs Visible During Training

If monitoring this training live, these would be red flags:

### Epoch 15-20: Yellow Flags ⚠️
- "Eval loss stopped decreasing"
- "Train loss still going down"
- **Action:** Prepare to stop soon

### Epoch 20-30: Orange Flags 🟧
- "Eval loss is increasing!"
- "Gap between train/eval growing"
- **Action:** Stop training now

### Epoch 30-50: Red Flags 🚨
- "Eval loss doubled from optimal"
- "Training loss near zero"
- **Action:** Stop immediately, damage occurring

### Epoch 50-100: Black Flags ☠️
- "Training loss is literally zero"
- "Gradients vanishing"
- "Eval loss 3-4x worse than optimal"
- **Action:** Kill the job, it's wasting resources

---

## Recommendations for Future Runs

### Critical Changes (Must Implement)

1. **Add Early Stopping**
   ```python
   from transformers import EarlyStoppingCallback

   trainer = Trainer(
       callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
   )
   ```

2. **Add LoRA Dropout**
   ```python
   lora_dropout=0.1
   ```

3. **Reduce Epochs**
   ```python
   num_train_epochs=20  # Max
   ```

4. **Lower Learning Rate**
   ```python
   learning_rate=0.0001
   ```

### Helpful Changes (Strongly Recommended)

5. **Increase Weight Decay**
   ```python
   weight_decay=0.05
   ```

6. **Add LR Warmup + Cosine Schedule**
   ```python
   warmup_steps=10
   lr_scheduler_type="cosine"
   ```

7. **Save Best Model**
   ```python
   load_best_model_at_end=True
   metric_for_best_model="eval_loss"
   ```

---

## Conclusion

This training run demonstrates severe overfitting caused by:
- Training for 100 epochs (85 epochs past optimal)
- No dropout regularization
- No early stopping mechanism
- Learning rate 2× higher than successful runs

The model reached its optimal performance at epoch 15 (eval_loss=0.138) but training continued for 85 more epochs, degrading evaluation performance by 3.6× to 0.492. The final model achieved perfect performance on training data (loss=0.0000) but became unusable for real-world inference.

**The solution:** Implement early stopping, add dropout, reduce epochs to 20, and lower learning rate to 0.0001. These changes would stop training automatically around epoch 18-20, saving 83% of compute time while producing a 3.6× better model.

**Key insight:** More training is not always better. The optimal model existed briefly and was destroyed by excessive training. Early stopping would have captured it automatically.
