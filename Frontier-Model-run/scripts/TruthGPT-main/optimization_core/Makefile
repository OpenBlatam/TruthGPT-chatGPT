.PHONY: help install train train-lora train-perf train-debug validate benchmark demo test clean visualize compare monitor health

help:
	@echo "TruthGPT Optimization Core - Comandos disponibles:"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Instalar dependencias"
	@echo "  make validate      - Validar configuración YAML"
	@echo "  make health        - Verificar entorno y configuración"
	@echo ""
	@echo "Training:"
	@echo "  make train         - Entrenar modelo (GPU, default config)"
	@echo "  make train-lora    - Entrenar con preset LoRA fast"
	@echo "  make train-perf    - Entrenar con preset performance max"
	@echo "  make train-debug   - Entrenar con preset debug"
	@echo "  make train-cpu     - Entrenar modelo (CPU)"
	@echo ""
	@echo "Utilities:"
	@echo "  make benchmark     - Benchmark de tokens/s"
	@echo "  make demo          - Demo interactiva Gradio"
	@echo "  make visualize     - Visualizar resultados de entrenamiento"
	@echo "  make compare       - Comparar múltiples runs"
	@echo "  make monitor       - Monitorear entrenamiento en tiempo real"
	@echo "  make cleanup-dry  - Ver qué se limpiaría (dry run)"
	@echo "  make cleanup       - Limpiar runs antiguos y checkpoints"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Ejecutar tests básicos"
	@echo "  make clean         - Limpiar checkpoints y caché"
	@echo ""

install:
	pip install -r requirements_advanced.txt

validate:
	python validate_config.py configs/llm_default.yaml

train:
	CUDA_VISIBLE_DEVICES=0 python train_llm.py --config configs/llm_default.yaml

train-lora:
	CUDA_VISIBLE_DEVICES=0 python train_llm.py --config configs/presets/lora_fast.yaml

train-perf:
	CUDA_VISIBLE_DEVICES=0 python train_llm.py --config configs/presets/performance_max.yaml

train-debug:
	CUDA_VISIBLE_DEVICES=0 python train_llm.py --config configs/presets/debug.yaml

train-cpu:
	CUDA_VISIBLE_DEVICES="" python train_llm.py --config configs/llm_default.yaml

benchmark:
	python examples/benchmark_tokens_per_sec.py --model gpt2 --dtype bf16

demo:
	python demo_gradio_llm.py

visualize:
	python utils/visualize_training.py runs --summary

compare:
	python utils/compare_runs.py --runs-dir runs

cleanup-dry:
	python utils/cleanup_runs.py --days 30 --keep-checkpoints 3

cleanup:
	python utils/cleanup_runs.py --days 30 --keep-checkpoints 3 --execute

monitor:
	python utils/monitor_training.py runs

health:
	python utils/health_check.py

test:
	python -c "from trainers.trainer import GenericTrainer; print('✅ Import OK')"
	python -c "from build_trainer import build_trainer; print('✅ Builder OK')"
	python validate_config.py configs/llm_default.yaml

clean:
	rm -rf runs/*/step_*.pt
	rm -rf __pycache__ */__pycache__ */*/__pycache__
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

