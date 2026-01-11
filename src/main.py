from training.trainer import Trainer
from config import Config
from utils import save_model

if __name__ == "__main__":
    trainer = Trainer(Config())
    trainer.train()
    save_model(trainer.model, "model.pth")
    print("Model saved successfully")