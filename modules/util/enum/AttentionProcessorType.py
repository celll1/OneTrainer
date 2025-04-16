from enum import Enum


class AttentionProcessorType(Enum):
    NONE = "none"
    SAGE = "sage"
    FLASH_ATTENTION_2 = "flash_attention_2"

    def __str__(self):
        return self.value
