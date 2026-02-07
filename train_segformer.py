import os, json, math
import importlib.util
import torch
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
)
from nh_datasets.loader import build_dataset_from_py
from utils import (
    setup_devices_autodetect, safe_training_args, compute_mIoU,
    choose_resume_checkpoint, parse_args, discover_best_model_dir,
    take_first_n, ddp_barrier_safe
)
from loss import CompoundLoss

# --- Custom Trainer for OHEM/Dice ---
class SegformerTrainer(Trainer):
    def __init__(self, *args, num_classes=11, ignore_index=255, **kwargs):
        super().__init__(*args, **kwargs)
        self.criterion = CompoundLoss(num_classes=num_classes, ignore_index=ignore_index)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):          
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        # Upsample logits to match label size
        upsampled_logits = torch.nn.functional.interpolate(
            logits, size=labels.shape[-2:], mode="bilinear", align_corners=False
        )
        loss = self.criterion(upsampled_logits, labels)
        return (loss, outputs) if return_outputs else loss

# --- Helper to load config ---
def load_config_module(path):
    spec = importlib.util.spec_from_file_location("config_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def train(args, ddp_kwargs):
    config = load_config_module(args.config_file)
    
    image_processor = SegformerImageProcessor(do_resize=False, do_normalize=True, reduce_labels=False)
    
    train_ds = build_dataset_from_py(args.config_file, split=args.train_split, augment=True, image_processor=image_processor)
    val_ds = build_dataset_from_py(args.config_file, split=args.val_split, augment=False, image_processor=image_processor)
    
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        id2label=train_ds.id2label,
        label2id=train_ds.label2id,
        ignore_mismatched_sizes=True,
    )

    def collate_fn(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch], dim=0)
        labels = torch.stack([b["labels"] for b in batch], dim=0)
        return {"pixel_values": pixel_values, "labels": labels}

    # Read Safe Defaults or Config Values
    lr_scheduler = getattr(config, "lr_scheduler_type", "linear")
    warmup = getattr(config, "warmup_ratio", 0.0)
    smoothing = getattr(config, "label_smoothing_factor", 0.0)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="mIoU",
        greater_is_better=True,
        fp16=args.fp16,
        remove_unused_columns=False,
        dataloader_drop_last=True,
        
        # --- OPTIMIZATIONS ---
        lr_scheduler_type=lr_scheduler,
        warmup_ratio=warmup,
        label_smoothing_factor=smoothing,
        # ---------------------
        **ddp_kwargs,
    )

    def metrics_fn(eval_pred):
        return compute_mIoU(eval_pred, num_classes=args.num_classes, ignore_index=255)

    trainer = SegformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_fn,
        compute_metrics=metrics_fn,
        num_classes=args.num_classes,
        ignore_index=255
    )

    resume_from = choose_resume_checkpoint(args.resume, args.output_dir)
    trainer.train(resume_from_checkpoint=resume_from)
    trainer.save_model(args.output_dir)
    print("Training complete.")

def main():
    args = parse_args()
    mode, local_rank, world_size = setup_devices_autodetect()
    ddp_kwargs = {}
    if world_size > 1:
        ddp_kwargs.update(dict(ddp_find_unused_parameters=False, ddp_backend="nccl"))
    set_seed(args.seed)
    train(args, ddp_kwargs)

if __name__ == "__main__":
    main()