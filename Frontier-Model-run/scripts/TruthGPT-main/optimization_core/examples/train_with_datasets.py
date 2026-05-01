import yaml
from optimization_core.scripts.legacy.build_trainer import build_trainer
from trainers.trainer import TrainerConfig


def main():
    cfg_dict = yaml.safe_load(open("optimization_core/configs/llm_default.yaml", "r", encoding="utf-8"))
    training = cfg_dict.get("training", {})
    cfg = TrainerConfig(
        seed=cfg_dict.get("seed", 42),
        run_name=cfg_dict.get("run_name", "run"),
        output_dir=cfg_dict.get("output_dir", "runs/run"),
        model_name=cfg_dict.get("model", {}).get("name_or_path", "gpt2"),
        epochs=min(1, int(training.get("epochs", 1))),
        train_batch_size=int(training.get("train_batch_size", 8)),
        eval_batch_size=int(training.get("eval_batch_size", 8)),
        grad_accum_steps=int(training.get("grad_accum_steps", 1)),
        learning_rate=float(training.get("learning_rate", 5e-5)),
    )

    trainer = build_trainer(cfg, cfg_dict, train_texts=[], val_texts=[], max_seq_len=int(cfg_dict.get("data", {}).get("max_seq_len", 512)))
    trainer.train()


if __name__ == "__main__":
    main()




