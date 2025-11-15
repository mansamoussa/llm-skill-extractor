import torch
from transformers import BertTokenizer, BertForTokenClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset # Assuming TensorDataset is created by preprocessing
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import Counter
import numpy as np
import gc
import os
import json # Needed to load label mappings

# TODO: Add any other necessary imports (e.g., for seqeval in final evaluation)
# from seqeval.metrics import classification_report

# --- Configuration ---
# Adjust these parameters as needed for experimentation
MODEL_NAME = 'bert-base-multilingual-cased'
NUM_EPOCHS = 3
LEARNING_RATE = 2e-5
BATCH_SIZE = 8
GRADIENT_ACCUMULATION_STEPS = 2
MAX_GRAD_NORM = 1.0 # Optional: Gradient clipping
VALIDATION_INTERVAL = 50
EARLY_STOPPING_PATIENCE = 5

# Define paths for data, models, and logs
# TODO: Verify these paths are correct for your project structure
DATA_DIR = '../data'
MODEL_DIR = '../model'
LOG_DIR = '../runs/student_run_fill_in'
LABEL_MAP_PATH = os.path.join(MODEL_DIR, 'id2label.json')
BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_student_model_fill_in.pt')
FINAL_MODEL_PATH = os.path.join(MODEL_DIR, 'final_student_model_fill_in.pt')

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Device Setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.cuda.empty_cache()

# --- Load Data and Mappings ---
print("Loading data and mappings...")
# TODO: Load your preprocessed train_dataset and test_dataset (PyTorch TensorDataset objects).
# These should contain input_ids, attention_mask, and labels for each sample.
# Example:
# train_dataset = torch.load(os.path.join(DATA_DIR, 'train_dataset.pt'))
# test_dataset = torch.load(os.path.join(DATA_DIR, 'test_dataset.pt'))
# --- Start Student Code ---
train_dataset = None # Replace with loading code
test_dataset = None # Replace with loading code
# --- End Student Code ---


# TODO: Load the id2label mapping from LABEL_MAP_PATH (JSON file).
# Then, derive the label2id mapping and the total number of labels (NUM_LABELS).
# Example:
# with open(LABEL_MAP_PATH, 'r') as f:
#     id2label_loaded = json.load(f)
#     id2label = {int(k): v for k, v in id2label_loaded.items()}
# label2id = {v: k for k, v in id2label.items()}
# NUM_LABELS = len(id2label)
# --- Start Student Code ---
id2label = None # Replace with loading code
label2id = None # Replace with derivation code
NUM_LABELS = -1  # Replace with derivation code
# --- End Student Code ---

# Basic check after loading
if train_dataset is None or test_dataset is None or NUM_LABELS == -1:
    print("Error: Data or label mapping not loaded correctly. Please fill in the loading code.")
    exit()
else:
    print(f"Loaded train_dataset with {len(train_dataset)} samples.")
    print(f"Loaded test_dataset with {len(test_dataset)} samples.")
    print(f"Loaded label mapping with {NUM_LABELS} labels.")


# --- Data Loaders ---
# Create DataLoaders to handle batching
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True
)
val_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    pin_memory=True
)

# --- Model Initialization ---
print(f"Initializing model: {MODEL_NAME}")
# TODO: Initialize the BertForTokenClassification model using MODEL_NAME.
# Ensure you pass the correct num_labels, id2label, and label2id.
# Example:
# model = BertForTokenClassification.from_pretrained(
#     MODEL_NAME,
#     num_labels=NUM_LABELS,
#     id2label=id2label,
#     label2id=label2id
# )
# --- Start Student Code ---
model = None # Replace with model initialization code
# --- End Student Code ---

if model is None:
    print("Error: Model not initialized. Please fill in the initialization code.")
    exit()

# Optional: Enable gradient checkpointing if needed for memory saving
# model.gradient_checkpointing_enable()

model.to(device) # Move model to the configured device

# --- Optimizer and Scheduler ---
# TODO: Initialize your chosen optimizer (e.g., AdamW). Pass the model parameters and learning rate.
# Example:
# optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
# --- Start Student Code ---
optimizer = None # Replace with optimizer initialization code
# --- End Student Code ---


# Calculate total training steps needed for the scheduler
num_training_steps = (len(train_loader) // GRADIENT_ACCUMULATION_STEPS) * NUM_EPOCHS
num_warmup_steps = int(0.1 * num_training_steps) # 10% warmup is common

# TODO: Initialize your chosen learning rate scheduler (e.g., linear warmup).
# Pass the optimizer, num_warmup_steps, and num_training_steps.
# Example:
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=num_warmup_steps,
#     num_training_steps=num_training_steps
# )
# --- Start Student Code ---
scheduler = None # Replace with scheduler initialization code
# --- End Student Code ---

if optimizer is None or scheduler is None:
    print("Error: Optimizer or Scheduler not initialized. Please fill in the initialization code.")
    exit()

print(f"Total training steps: {num_training_steps}, Warmup steps: {num_warmup_steps}")

# --- Loss Function (with Class Weights) ---
print("Calculating class weights...")
# TODO: Calculate class weights to handle label imbalance.
# HINT: Iterate through train_dataset, count the frequency of each label ID
# (make sure to ignore padding labels, often -100).
# Calculate the weight for each label (e.g., using inverse frequency).
# Store weights in a list `class_weights_list` where index corresponds to label ID.
# --- Start Student Code ---
class_weights_list = [1.0] * NUM_LABELS # Placeholder: Equal weights initially
# Add code here to iterate through train_dataset, count labels, and calculate weights.
# Remember to handle labels that might not appear in the training set (assign weight 1.0 or similar).

# --- End Student Code ---

# Convert weights to a PyTorch tensor and move to device
class_weights = torch.FloatTensor(class_weights_list).to(device)
print("Class weights calculated (or using default) and moved to device.")

# Initialize loss function with class weights and ignore_index for padding
loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights, ignore_index=-100)


# --- Evaluation Function ---
def evaluate_model(model, data_loader, loss_function, device, num_labels):
    model.eval()
    total_loss = 0
    total_correct_predictions = 0
    total_active_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids, labels, attention_mask = [b.to(device) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # --- Start Student Code ---
            # TODO: Calculate the loss ONLY for active (non-padding, non -100) tokens.
            # HINT: Create a mask for active labels (labels != -100).
            #       Apply the mask to logits and labels before passing to the loss function.
            #       Handle cases where a batch might contain only padding.
            batch_loss = 0.0 # Replace with calculated loss for the batch
            # TODO: Accumulate the total loss.
            # total_loss += ...

            # TODO: Calculate the number of correct predictions ONLY for active tokens.
            # HINT: Apply the active mask, get predictions (argmax), compare with true labels.
            batch_correct_predictions = 0 # Replace with calculated correct predictions for the batch
            batch_active_tokens = 0 # Replace with count of active tokens in the batch
            # TODO: Accumulate the total number of correct predictions and total active tokens.
            # total_correct_predictions += ...
            # total_active_tokens += ...
            # --- End Student Code ---

            total_loss += batch_loss # Accumulate loss calculated by student code
            total_correct_predictions += batch_correct_predictions # Accumulate correct preds
            total_active_tokens += batch_active_tokens # Accumulate active tokens

            del outputs, logits, input_ids, labels, attention_mask

    avg_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    accuracy = total_correct_predictions / total_active_tokens if total_active_tokens > 0 else 0

    model.train()
    return avg_loss, accuracy

# --- TensorBoard Setup ---
writer = SummaryWriter(log_dir=LOG_DIR)
print(f"TensorBoard logs will be saved to: {LOG_DIR}")

# --- Training Loop ---
global_step = 0
best_val_loss = float('inf')
patience_counter = 0

print(f"Starting training...")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
print(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")

for epoch in range(NUM_EPOCHS):
    model.train()
    epoch_train_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)
    # Reset gradients at the start of each epoch (safer)
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(progress_bar):
        input_ids, labels, attention_mask = [b.to(device) for b in batch]

        # Forward pass - Get model outputs (including loss when labels are provided)
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss # Loss is calculated by the Hugging Face model internally

        # Scale loss for gradient accumulation
        scaled_loss = loss / GRADIENT_ACCUMULATION_STEPS

        # --- Start Student Code ---
        # TODO: Perform the backward pass to compute gradients based on the scaled loss.
        # scaled_loss.???
        # --- End Student Code ---

        # Accumulate loss for logging (use the unscaled loss)
        epoch_train_loss += loss.item()

        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)

        # Optimizer Step (with Gradient Accumulation)
        if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            # --- Start Student Code ---
            # TODO: Update model weights using the optimizer.
            # optimizer.???
            # TODO: Update the learning rate using the scheduler.
            # scheduler.???
            # TODO: Reset gradients for the next accumulation cycle.
            # optimizer.???
            # --- End Student Code ---

            global_step += 1

            # Logging
            writer.add_scalar('Loss/train_step', loss.item(), global_step)
            writer.add_scalar('Learning_rate', scheduler.get_last_lr()[0], global_step)

            # Periodic Validation
            if global_step % VALIDATION_INTERVAL == 0:
                val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fct, device, NUM_LABELS)
                writer.add_scalar('Loss/validation_step', val_loss, global_step)
                writer.add_scalar('Accuracy/validation_step', val_accuracy, global_step)
                print(f"\nStep {global_step}: Val Loss={val_loss:.4f}, Val Acc={val_accuracy:.4f}")

                # Early Stopping & Model Saving
                if val_loss < best_val_loss:
                    print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save the best model state
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'val_loss': best_val_loss,
                        'num_labels': NUM_LABELS,
                        'id2label': id2label
                    }, BEST_MODEL_PATH)
                else:
                    patience_counter += 1
                    print(f"  Validation loss did not improve. Patience: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    print(f"\nEarly stopping triggered after {global_step} steps.")
                    break # Exit inner loop

                model.train() # Ensure model is back in training mode

        # Update progress bar description
        progress_bar.set_postfix({
            'train_loss': epoch_train_loss / (batch_idx + 1),
            'best_val_loss': best_val_loss
        })

        del outputs, loss, scaled_loss, input_ids, labels, attention_mask

    # --- End of Epoch ---
    if patience_counter >= EARLY_STOPPING_PATIENCE:
        print("Exiting training loop due to early stopping.")
        break # Exit outer loop

    print(f"\n--- End of Epoch {epoch + 1} ---")
    avg_epoch_train_loss = epoch_train_loss / len(train_loader)
    print(f"Average Training Loss: {avg_epoch_train_loss:.4f}")

    # Final validation for the epoch
    val_loss, val_accuracy = evaluate_model(model, val_loader, loss_fct, device, NUM_LABELS)
    writer.add_scalar('Loss/validation_epoch', val_loss, epoch + 1)
    writer.add_scalar('Accuracy/validation_epoch', val_accuracy, epoch + 1)
    print(f"End-of-Epoch Validation Loss: {val_loss:.4f}")
    print(f"End-of-Epoch Validation Accuracy: {val_accuracy:.4f}")

    # Check for saving best model again at epoch end
    if val_loss < best_val_loss:
         print(f"  Validation loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
         best_val_loss = val_loss
         patience_counter = 0 # Reset patience on improvement
         torch.save({
             'epoch': epoch,
             'model_state_dict': model.state_dict(),
             'optimizer_state_dict': optimizer.state_dict(),
             'scheduler_state_dict': scheduler.state_dict(),
             'val_loss': best_val_loss,
             'num_labels': NUM_LABELS,
             'id2label': id2label
         }, BEST_MODEL_PATH)

    gc.collect()
    torch.cuda.empty_cache()

# --- Training Finished ---
print("\nTraining completed!")
writer.close()

# --- Save Final Model ---
print(f"Saving final model state to {FINAL_MODEL_PATH}...")
# Ensure val_loss is defined even if training stopped early
final_val_loss = val_loss if 'val_loss' in locals() else float('inf')
torch.save({
    'epoch': epoch, # Last completed epoch
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'val_loss': final_val_loss,
    'num_labels': NUM_LABELS,
    'id2label': id2label
}, FINAL_MODEL_PATH)

print(f"Best validation loss achieved: {best_val_loss:.4f}")
print(f"Best model saved to: {BEST_MODEL_PATH}")
print(f"Final model saved to: {FINAL_MODEL_PATH}")
print(f"TensorBoard logs are located in: {LOG_DIR}")


# --- Final Evaluation ---
# TODO: Implement the final evaluation on the test set using the best model.
# 1. Load the best model state dict from BEST_MODEL_PATH into the model structure.
# 2. Use the `val_loader` (or create a dedicated `test_loader` if you split data differently)
# 3. Call the `evaluate_model` function to get loss and accuracy on the test set.
# 4. **Crucially:** Implement evaluation using `seqeval` to get Precision, Recall, F1-score.
#    - This involves getting predictions for the entire test set batch by batch.
#    - Convert prediction IDs and true label IDs back to strings (using id2label).
#    - **Important:** Align predictions and labels, removing padding tokens/subword tokens appropriately
#      before feeding sequences (as lists of label strings) to `seqeval`.
#    - Use `seqeval.metrics.classification_report` to print detailed results.
print("\nRunning final evaluation on test set using the best model...")
# --- Start Student Code ---
# Load best model
# checkpoint = torch.load(BEST_MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])

# Get predictions and true labels for the test set
# all_preds = []
# all_labels = []
# model.eval()
# with torch.no_grad():
#    for batch in val_loader: # Assuming val_loader uses the test set
#        # ... get predictions and labels, handle padding, convert IDs to label strings ...
#        # Append lists of label strings for each sequence in the batch to all_preds and all_labels

# Calculate and print seqeval report
# report = classification_report(all_labels, all_preds, mode='strict', scheme='IOB2') # Adjust scheme if needed
# print(report)

# --- End Student Code ---

print("\nScript finished.")

