# DPO: Direct Preference Optimization

## The short answer

DPO teaches a model to prefer good responses over bad ones. Instead of
training a separate reward model like RLHF does DPO works directly
with pairs of responses. Each training example shows the model a chosen
response and a rejected response for the same prompt. The model learns
to increase the probability of the chosen one and decrease the
probability of the rejected one. No reward model needed. No
reinforcement learning needed. Just a dataset of preferences and a small
modification to the training loss.

## Where it sits

DPO is a fine-tuning method. It takes a base chat model that already
knows how to follow instructions and improves the quality of its
responses. RLHF came first. DPO came later and is simpler. Both achieve
the same goal: making the model's outputs more aligned with what humans
want.

```
Base model (knows language)
  → Instruction tuning (knows how to follow instructions)
    → DPO or RLHF (knows which responses are better)
      → Aligned model (helpful, harmless, honest)
```

## Why DPO replaced RLHF for most teams

RLHF has three steps. Train a reward model on human preference data.
Use the reward model to score the language model's outputs. Update the
language model using reinforcement learning to maximize the reward. Each
step is complex and error prone. The reward model can be fooled. The RL
training can be unstable. The whole process requires constant
monitoring.

DPO has one step. Train the language model directly on the preference
data using a modified loss function. The loss encourages the model to
increase the probability of chosen responses relative to rejected
responses. No separate reward model. No RL algorithm. No instability.
The training loop is the same as standard fine-tuning. Only the loss
function changes.

## How DPO works

DPO compares two responses to the same prompt. One was preferred by a
human. One was rejected. The model sees both and adjusts its weights to
make the chosen response more likely and the rejected response less
likely.

The DPO loss is a modified cross entropy loss.

```
Given:
  prompt: "Explain gravity to a child"
  chosen: "Gravity is the force that pulls things toward each other."
  rejected: "Gravity is a fundamental interaction."

The DPO loss:
  log_ratio = log(P(chosen | prompt)) - log(P(rejected | prompt))
  loss = -log(sigmoid(beta * log_ratio))
```

The beta parameter controls how strongly the model is pushed toward the
chosen responses. Higher beta means stronger preference signal. Lower
beta is more conservative. Typical beta values are 0.1 to 0.5.

The reference model is the model before DPO training. DPO compares the
current model's probabilities to the reference model's probabilities.
This prevents the model from drifting too far from its original
behavior. Without the reference model the model could learn to repeat
the chosen responses verbatim. With the reference model it learns the
general pattern of what makes a good response.

```
Full DPO loss with reference model:

log_ratio_current = log(P_current(chosen | prompt)) - log(P_current(rejected | prompt))
log_ratio_ref = log(P_ref(chosen | prompt)) - log(P_ref(rejected | prompt))
loss = -log(sigmoid(beta * (log_ratio_current - log_ratio_ref)))
```

The subtraction of the reference model's log ratio is what makes DPO
work. It asks: does the current model prefer the chosen response MORE
than the reference model did. If yes the loss is small. If no the loss
is large and the model is updated.

## The data format

DPO training data is simple. Each example has a prompt, a chosen
response and a rejected response.

```json
{
  "prompt": "Explain gravity to a child.",
  "chosen": "Gravity is what keeps your feet on the ground. It pulls everything toward the Earth.",
  "rejected": "Gravity is a fundamental interaction. Newton described it mathematically."
}
```

The chosen response is better. Maybe it uses simpler language. Maybe it
is more complete. Maybe it avoids factual errors. The rejected response
is worse. Maybe it uses jargon. Maybe it is too brief. Maybe it contains
a mistake. The model learns from the difference.

The quality of DPO training depends entirely on the quality of
preference pairs. If the rejected response is actually better than the
chosen one the model learns the wrong lesson. Every preference pair must
be carefully curated. High quality data is expensive to produce but
essential for good results.

## A tiny code example

```python
import torch
import torch.nn.functional as F

def dpo_loss(policy_model, reference_model, prompt_ids,
             chosen_ids, rejected_ids, beta=0.1):
    """
    Compute DPO loss for a single preference pair.
    policy_model is the model being trained.
    reference_model is frozen and represents pre-DPO behavior.
    """
    with torch.no_grad():
        ref_chosen_logp = reference_model(chosen_ids)[0].log_softmax(-1).sum()
        ref_rejected_logp = reference_model(rejected_ids)[0].log_softmax(-1).sum()
    ref_log_ratio = ref_chosen_logp - ref_rejected_logp

    policy_chosen_logp = policy_model(chosen_ids)[0].log_softmax(-1).sum()
    policy_rejected_logp = policy_model(rejected_ids)[0].log_softmax(-1).sum()
    policy_log_ratio = policy_chosen_logp - policy_rejected_logp

    logits = policy_log_ratio - ref_log_ratio
    loss = -F.logsigmoid(beta * logits)

    return loss


# Example usage (conceptual)
preference_pairs = [
    {
        "prompt": "Explain gravity to a child.",
        "chosen": "Gravity is what keeps your feet on the ground.",
        "rejected": "Gravity is a fundamental physical interaction.",
    },
    {
        "prompt": "What is the capital of France?",
        "chosen": "The capital of France is Paris. It is known for the Eiffel Tower.",
        "rejected": "Paris.",
    },
]

policy_model = ...  # Your chat model
reference_model = ...  # Frozen copy of the same model

for pair in preference_pairs:
    prompt_ids = tokenize(pair["prompt"])
    chosen_ids = tokenize(pair["chosen"])
    rejected_ids = tokenize(pair["rejected"])

    loss = dpo_loss(policy_model, reference_model, prompt_ids,
                    chosen_ids, rejected_ids, beta=0.1)
    loss.backward()
    optimizer.step()
```

## DPO versus RLHF

| Aspect | RLHF | DPO |
|---|---|---|
| Steps | 3 (reward model + PPO + KL penalty) | 1 (direct loss) |
| Stability | Can be unstable. Needs careful tuning | Stable. Standard supervised learning |
| Compute | Higher. Must run reward model | Lower. Only the policy model |
| Code complexity | High. PPO implementation is tricky | Low. Modified loss function |
| Data | Same preference pairs | Same preference pairs |
| Quality | Established. Used by ChatGPT | Comparable in many benchmarks |

DPO has largely replaced RLHF in the open source community. It is
simpler to implement and achieves comparable results on most benchmarks.
The main advantage of RLHF is that it can learn from online feedback
where the model generates responses and humans rate them in real time.
DPO requires pre collected preference pairs. For offline datasets the
two methods perform similarly.

## When to use DPO

Use DPO when you have access to human preference data and want to
improve the quality of a model's responses beyond what instruction
tuning alone can achieve. The model must already follow instructions.
DPO cannot teach a base model to chat. It can only improve the quality
of a model that already chats.

Use DPO when you want to reduce harmful outputs or improve factual
accuracy or make responses more concise or align with specific style
guidelines. Any quality dimension that can be expressed as a preference
between two responses can be optimized with DPO.

## What you need to remember

DPO trains a model directly on preference pairs without a separate
reward model. The loss function encourages the model to prefer chosen
responses over rejected ones. A reference model prevents the model from
drifting too far from its original behavior. DPO is simpler and more
stable than RLHF and achieves comparable results. It is the standard
preference optimization method in the open source community.
