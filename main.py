from src.training.trainer import Trainer
from src.config import Config
from src.utils import save_model
from src.training.eval import Evaluator

if __name__ == "__main__":
    trainer = Trainer(Config())
    trainer.train()
    save_model(trainer.model, "model.pth")
    print("Model saved successfully")
    evaluator = Evaluator(trainer.model, "model.pth")
    evaluator.evaluate()