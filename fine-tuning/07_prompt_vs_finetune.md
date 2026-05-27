# Prompt Engineering versus Fine-Tuning

## The short answer

Both prompt engineering and fine-tuning change how a model behaves. The
difference is where the change lives. Prompt engineering changes the
input. Fine-tuning changes the model itself. Choose prompt engineering
when you need quick results on a few use cases. Choose fine-tuning when
you need consistent behavior at scale.

## Where they sit

```
Raw user input
  → Prompt engineering adds context and instructions
    → Base or fine-tuned model
      → Response
```

Prompt engineering wraps the user's request in additional text that
guides the model. The model weights never change. Every query gets the
same prompt template. The model interprets the template and responds
accordingly.

Fine-tuning bakes the guidance into the model's weights. The prompt can
be simple because the model already knows how to behave. The behavior
is consistent because it is encoded in the weights not in the prompt
text.

## When to prompt engineer

Prompt engineering is the first thing to try. It is free. It takes
minutes. It works with any model through any API.

Use prompt engineering when:
- You have a few well defined use cases
- The behavior you want can be described in a few sentences
- You do not need perfect consistency across hundreds of variations
- You are prototyping and iterating quickly
- You are using a hosted API and cannot modify the model

Prompt engineering can achieve remarkable results. A well written system
prompt can make a base chat model behave like a domain expert for
medical questions or a creative writing coach or a code reviewer. The
technique improves as base models get better. Each generation of models
requires less prompting to achieve the same results.

```python
# Prompt engineering example
system_prompt = """
You are a helpful medical assistant. Always:
- Use simple language a patient can understand
- Never diagnose. Suggest seeing a doctor instead
- Cite reliable sources when possible
- Ask clarifying questions if symptoms are vague
"""

user_question = "My head hurts and I feel dizzy."

full_prompt = f"{system_prompt}\n\nPatient: {user_question}\nAssistant:"
response = model.generate(full_prompt)
```

The prompt does all the work. The model is unchanged. Different prompts
can be used for different use cases. A medical prompt for health
questions. A legal prompt for contract review. A creative prompt for
story generation. All running on the same base model.

## When to fine-tune

Prompt engineering has limits. Long prompts consume context window
space. Complex behaviors are hard to describe in words. Some patterns
are easier to demonstrate than to explain. Prompt injection attacks can
override system prompts. Consistency across thousands of query
variations is hard to guarantee.

Fine-tuning addresses these limits. The model learns the desired
behavior from examples. The context window is free for the actual
user input. The behavior is encoded in the weights and cannot be
overridden by a cleverly crafted user message. The consistency comes
from thousands of training examples covering every edge case.

Use fine-tuning when:
- You have hundreds or thousands of examples of the desired behavior
- The behavior is too complex to describe in a prompt
- You need the context window for user input not instructions
- You need consistent behavior that cannot be prompt injected
- You are serving the model in production at scale

```python
# Fine-tuning example with LoRA
training_data = [
    {"symptom": "headache and dizziness", "response": "These symptoms can have many causes..."},
    {"symptom": "chest pain", "response": "Chest pain should be evaluated by a doctor immediately..."},
    {"symptom": "sore throat", "response": "A sore throat is often caused by a viral infection..."},
    # ... hundreds more examples
]

model = load_base_model()
model = add_lora(model, rank=16)

for example in training_data:
    prompt = f"Patient: {example['symptom']}\nAssistant:"
    response = example['response']
    loss = compute_loss(model, prompt, response)
    loss.backward()

# Now the model responds medically without any prompt instructions
response = model.generate("Patient: My head hurts and I feel dizzy.\nAssistant:")
```

After fine-tuning the model produces medical responses without needing
a system prompt. The behavior is in the weights. The context window is
free for the patient's detailed description of symptoms.

## The cost comparison

| Aspect | Prompt Engineering | Fine-Tuning |
|---|---|---|
| Time to implement | Minutes | Hours to days |
| Cost | Free (API costs per query) | GPU time + data collection |
| Expertise needed | Writing skills | ML engineering skills |
| Consistency | Varies with prompt phrasing | Consistent across variations |
| Context window used | 200 to 2000 tokens for prompt | 0 tokens for instructions |
| Vulnerability to injection | High | Low |
| Iteration speed | Instant | Hours per experiment |
| Model dependency | Must redo for each model | Adapter transfers between models |

## The hybrid approach

Most production systems use both. Fine-tuning teaches the model the
core behavior. Prompt engineering handles the situation specific
details that change with each query.

```
Fine-tuning: teaches the model to be a medical assistant
Prompt: adds patient specific context, recent lab results, medication list

Fine-tuning: teaches the model to review Python code
Prompt: adds the specific code to review, coding standards, context

Fine-tuning: teaches the model to write in a brand voice
Prompt: adds the specific topic, target audience, desired length
```

The fine-tuned model provides the foundation. The prompt provides the
specifics. Together they are more effective than either alone.

## The decision flowchart

```
Do you have a few use cases and need results today?
  → Prompt engineer

Do you need consistent behavior across thousands of user inputs?
  → Fine-tune

Can you describe the desired behavior clearly in words?
  → Prompt engineer

Is the behavior hard to describe but easy to demonstrate with examples?
  → Fine-tune

Are you using a hosted API and cannot modify the model?
  → Prompt engineer

Do you have a dataset of good responses for your use case?
  → Fine-tune

Is each query unique with specific context the model needs?
  → Prompt engineer

Do you need to prevent users from overriding instructions?
  → Fine-tune
```

## What you need to remember

Prompt engineering and fine-tuning are complementary. Prompt
engineering is fast and flexible and changes with every query.
Fine-tuning is slow and consistent and lives in the model's weights.
Start with prompt engineering. Move to fine-tuning when you need
consistency or scale or resistance to injection attacks. Most
production systems combine both: fine-tuned models with task specific
prompts.
