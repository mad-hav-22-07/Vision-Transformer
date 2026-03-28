"""Quick debug script to get the full traceback."""
import traceback
try:
    from src.train import main
    import sys
    sys.argv = ["train", "--config", "configs/train_config.yaml", "--epochs", "2", "--batch_size", "4"]
    main()
except Exception as e:
    traceback.print_exc()
