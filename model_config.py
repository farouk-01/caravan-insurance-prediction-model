from dataclasses import dataclass, asdict
from typing import Optional

@dataclass
class ModelConfig:
    threshold:float = 0.1
    class_weight:Optional[float] = None
    learning_rate:float = 0.01
    iterations:int = 1000
    l2_reg:bool = False
    l1_reg:bool = False
    lambda_const:Optional[float] = None

    def as_dict(self):
        return asdict(self)

    def copy_with(self, **updates):
        return ModelConfig(**{**self.as_dict(), **updates})

    def __str__(self):
        return (
            f"ModelConfig("
            f"learning_rate={self.learning_rate:.4f}, "
            f"class_weight={self.class_weight:.4f}, "
            f"threshold={self.threshold:.4f}), "
            f"l2_reg={self.l2_reg})"
            f"l1_reg={self.l1_reg})"
            f"lambda_const={self.lambda_const:.4f})"
        )