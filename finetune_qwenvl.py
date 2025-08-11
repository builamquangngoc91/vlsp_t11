# Before running, ensure you have the required libraries installed:
# pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
# pip install --no-deps trl peft accelerate bitsandbytes
# pip install datasets transformers

import torch
from unsloth import FastVisionModel
from trl import SFTTrainer, SFTConfig
from unsloth.trainer import UnslothVisionDataCollator
from prepare_data import create_dataset

def main():
    # Load and prepare the dataset
    json_path = "dataset/train/vlsp_2025_train.json"
    image_dir = "dataset/train/train_images"
    converted_dataset = create_dataset(json_path, image_dir)

    # Load the model and tokenizer
    model, tokenizer = FastVisionModel.from_pretrained(
        "unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit",
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )

    # Add LoRA adapters
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    # Set up the trainer with improved parameters
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=converted_dataset,
        args=SFTConfig(
            per_device_train_batch_size=1,  # Reduced for memory efficiency
            gradient_accumulation_steps=8,  # Increased to maintain effective batch size
            warmup_steps=50,  # Increased warmup for stability
            max_steps=3000,  # Increased training steps
            num_train_epochs=3,  # Add epochs for better convergence
            learning_rate=1e-4,  # Slightly reduced learning rate for stability
            logging_steps=25,  # Less frequent logging
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",  # Cosine scheduling for better convergence
            seed=3407,
            output_dir="outputs",
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=4096,  # Keep max length for complex prompts
            save_steps=500,  # Save checkpoints more frequently
            eval_steps=500,  # Add evaluation steps
            save_total_limit=3,  # Limit saved checkpoints
            dataloader_num_workers=2,  # Parallel data loading
            fp16=True,  # Enable mixed precision training
        ),
    )

    # Train the model
    trainer.train()

    # Save the fine-tuned model
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

if __name__ == "__main__":
    main()
