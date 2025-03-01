import torch
import torch.nn as nn
import torch.optim as optim
import random
from transformer_lens import HookedTransformer, HookedTransformerConfig
from model_registry import ModelRegistry, ModelConfig
from dataclasses import asdict

# ----- Device Configuration -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- Tokenization -----
# Vocabulary: digits 0-9, period, plus, equals.
chars = "0123456789.+=-"
token_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_token = {i: ch for ch, i in token_to_id.items()}

def tokenize(text):
    return [token_to_id[ch] for ch in text]

def detokenize(tokens):
    return "".join(id_to_token[t] for t in tokens)

# ----- Data Generation -----
def generate_addition_problem(min_digits=1, max_digits=6):
    """
    Generates a single addition problem string in the form:
    .<a>+<b>.=<a+b>.
    """
    digit_length = random.randint(min_digits, max_digits)
    a = random.randint(10**(digit_length - 1), 10**digit_length - 1)
    b = random.randint(10**(digit_length - 1), 10**digit_length - 1)
    problem = f".{a}.+.{b}.=.{a+b}."
    return problem

def generate_addition_dataset(dataset_size=1000, min_digits=1, max_digits=3):
    return [generate_addition_problem(min_digits, max_digits) for _ in range(dataset_size)]

def create_tensor_dataset(dataset, pad_token='-', max_length=None):
    tokenized = [tokenize(problem) for problem in dataset]
    if max_length is None:
        max_length = max(len(seq) for seq in tokenized)
    padded = [seq + [token_to_id[pad_token]] * (max_length - len(seq)) for seq in tokenized]
    return torch.tensor(padded, dtype=torch.long).to(device)


def get_batch(dataset_tensor, batch_size):
    indices = torch.randint(0, dataset_tensor.size(0), (batch_size,))
    batch = dataset_tensor[indices]
    # Input tokens: all tokens except the last one.
    # Targets: all tokens except the first one.
    return batch, batch[:, 1:]

def generate_completion(model, prompt_tokens, max_gen=10):
    model.eval()
    generated = prompt_tokens.tolist()[0]
    for _ in range(max_gen):
        inp = torch.tensor([generated], dtype=torch.long).to(device)
        logits = model(inp)
        next_token = torch.argmax(logits[0, -1]).item()
        generated.append(next_token)
        if next_token == token_to_id['.'] and len(generated) > len(prompt_tokens[0]) + 1:
            break
    return generated

def generate_completion_with_cache(model, prompt_tokens, max_gen=10):
    """
    Similar to generate_completion but also returns the cache and logits from the final forward pass.
    Args:
        model: The transformer model
        prompt_tokens: Input tokens tensor
        max_gen: Maximum number of tokens to generate
    Returns:
        tuple: (generated_tokens, cache, logits)
    """
    model.eval()
    generated = prompt_tokens.tolist()[0]
    for _ in range(max_gen):
        inp = torch.tensor([generated], dtype=torch.long).to(device)
        logits, cache = model.run_with_cache(inp)
        next_token = torch.argmax(logits[0, -1]).item()
        generated.append(next_token)
        if next_token == token_to_id['.'] and len(generated) > len(prompt_tokens[0]) + 1:
            break
    return generated, cache, logits

def extract_numbers_from_problem(problem_str):
    # Split by periods to get the meaningful parts
    parts = problem_str.split('.')
    
    # The format is ".{a}.+.{b}.=.{a+b}." so parts would be ['', '{a}', '+', '{b}', '=', '{a+b}', '']
    # Extract the operands
    num1 = int(parts[1])
    num2 = int(parts[3])
    
    # Get the answer if it exists
    answer = int(parts[5]) if len(parts) > 5 and parts[5] else None
    
    return num1, num2, answer

def evaluate_model(model, test_dataset):
    total, correct = 0, 0
    for problem in test_dataset:
        # Find the equals sign followed by period
        eq_index = problem.find("=.")
        if eq_index == -1:
            continue
            
        # Create prompt up to and including the equals sign and period
        prompt = problem[:eq_index+2]
        prompt_tokens = torch.tensor([tokenize(prompt)], dtype=torch.long).to(device)
        
        # Generate completion
        generated_tokens = generate_completion(model, prompt_tokens, max_gen=10)
        generated_str = detokenize(generated_tokens)
        
        # Extract predicted and true answers
        try:
            # Extract numbers from both the original problem and the generated completion
            _, _, true_answer = extract_numbers_from_problem(problem)
            _, _, pred_answer = extract_numbers_from_problem(generated_str)
            
            # Compare answers
            if pred_answer is not None and pred_answer == true_answer:
                correct += 1
        except (ValueError, IndexError):
            # Handle any parsing errors
            pass
            
        total += 1
        
    accuracy = (correct / total * 100) if total > 0 else 0
    print(f"Evaluation Accuracy: {accuracy:.2f}% on {total} examples")
    return accuracy

def train_model():
    print(f"Training on {device}")
    
    # ----- Registry Setup -----
    registry = ModelRegistry()

    # Create a new model configuration and save it
    config = ModelConfig(
        d_model=768,
        d_head=64,
        n_heads=12,
        n_layers=2,
        n_ctx=2048,
        d_vocab=len(chars),
        attention_dir="causal",
        attn_only=True,  # defaults to False
        # tokenizer_name="EleutherAI/gpt-neox-20b",
        seed=398,
        use_attn_result=True,
        normalization_type=None,  # defaults to "LN", i.e. layernorm with weights & biases
        positional_embedding_type="shortformer",
    )

    # Define a model name based on configuration (for example, using the layer count)
    model_name = f"arithmetic_model_{config.n_layers}layers"

    # Initialize the model using only the fields needed for HookedTransformerConfig
    ht_config = HookedTransformerConfig(**{
        k: v for k, v in asdict(config).items() if k in HookedTransformerConfig.__annotations__
    })
    model = HookedTransformer(ht_config).to(device)
    print(f"Moving model to device: {device}")

    # Initialize optimizer and criterion
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)

    # ----- Training Setup -----
    num_epochs = 10
    batch_size = 64
    log_interval = 10

    MIN_DIGITS = 3
    MAX_DIGITS = 3

    train_dataset = generate_addition_dataset(dataset_size=8000, min_digits=MIN_DIGITS, max_digits=MAX_DIGITS)
    test_dataset = generate_addition_dataset(dataset_size=2000, min_digits=MIN_DIGITS, max_digits=MAX_DIGITS)
    train_tensor = create_tensor_dataset(train_dataset)
    test_tensor = create_tensor_dataset(test_dataset)

    # ----- Training Loop -----
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 200
        for batch in range(num_batches):
            tokens, targets = get_batch(train_tensor, batch_size)
            optimizer.zero_grad()
            logits = model(tokens)
            # Trim logits to match target length (batch_size, seq_len-1, vocab_size)
            logits = logits[:, :-1, :]
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            if (batch + 1) % log_interval == 0:
                print(f"Epoch {epoch+1}, Batch {batch+1}/{num_batches}, Loss: {epoch_loss/log_interval:.4f}")
                epoch_loss = 0.0
        scheduler.step(loss)

    # Save the final model using the registry
    registry.save_model(model_name, model, config)

    # Load the best model from the registry
    model, config = registry.load_model(model_name)
    print(f"Loaded model: {model_name} with config: {config}")

    # ----- Testing & Evaluation -----
    with torch.no_grad():
        for i in range(5):
            problem = generate_addition_problem(min_digits=MIN_DIGITS, max_digits=MAX_DIGITS)
            prompt_end = problem.find("=.")
            prompt = problem[:prompt_end+2]  # Include the equals sign and period
            prompt_tokens = torch.tensor([tokenize(prompt)], dtype=torch.long).to(device)
            generated_tokens = generate_completion(model, prompt_tokens, max_gen=10)
            print(f"\nTest Case {i+1}:")
            print("Full Problem (ground truth):", problem)
            print("Prompt provided:", prompt)
            print("Model completion:", detokenize(generated_tokens))

    evaluate_model(model, test_dataset)
    
    return model, config

if __name__ == "__main__":
    train_model()