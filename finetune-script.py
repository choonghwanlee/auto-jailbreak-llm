from datasets import load_dataset
from huggingface_hub import login
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, AutoConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel
import argparse
from trl import SFTTrainer


## code to fine-tune Phi3  

def main(args):
    if args.do_quantize:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, ## set this to false or true?
        )
    config = AutoConfig.from_pretrained(args.model)
    config.pad_token_id = 0

    model = AutoModelForCausalLM.from_pretrained( 
        args.model,
        quantization_config=bnb_config if args.quantize else None,
        device_map='auto',
        trust_remote_code=True,
        attn_implementation='flash_attention_2',
        config=config, 
    ) 

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.model_max_length = args.max_length
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side='right' ## for training only

    model.resize_token_embeddings(len(tokenizer))

    model = prepare_model_for_kbit_training(model)

    model.enable_input_require_grads()

    if args.do_lora:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=2*args.lora_r, ## it is advised to set alpha to 2x R
            target_modules='all-linear',
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM'
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    args = TrainingArguments(
            output_dir=args.output_dir, ## push to args
            evaluation_strategy="steps", 
            do_eval=True,
            optim=args.optim,
            per_device_train_batch_size=args.batch_size, ## push to args
            gradient_accumulation_steps=args.gradient_acc, ## push to args
            per_device_eval_batch_size=args.batch_size,
            log_level="debug",
            save_strategy="epoch",
            logging_steps=100,
            learning_rate=args.lr, ## push to args
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            eval_steps=100,
            num_train_epochs=args.train_epochs, ## push to args
            warmup_ratio=args.warmup_ratio, ## push to args
            lr_scheduler_type="linear",
            seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=args.train_data,
        eval_dataset=args.eval_data,
        dataset_text_field="text",
        tokenizer=tokenizer,
        packing=False,
        peft_config=lora_config if args.do_lora else None,
        max_seq_length=args.max_length, ##arbitrary choice
    )

    trainer.train()
    if args.do_lora:
        trainer.model.push_to_hub(f'{args.hf_dir}_lora', private=True)
    else:
        trainer.model.push_to_hub(args.hf_dir, private=True)
        tokenizer.push_to_hub(args.hf_dir, private=True)

    print(f"DONE! The fine-tuned model (or adapters) can be found on HuggingFace hub as {args.hf_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str)
    parser.add_argument("--model", type=str) 
    parser.add_argument("--train_data", type=dict)
    parser.add_argument("--eval_data", type=dict)
    parser.add_argument("--do_quantize", type=bool, default=False)
    parser.add_argument("--do_lora", type=bool, default=False)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default='./phi3-finetuned')
    parser.add_argument("--hf_dir", type=str, default='./phi3-finetuned')
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_acc", type=int, default=2)
    parser.add_argument("--optim", type=str, default='adamw_torch')
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr", type=float, default=2e-5)

    args = parser.parse_args()
    login(token=args.hf_token)
    main(args)