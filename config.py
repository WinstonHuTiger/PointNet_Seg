from colossalai.amp import AMP_TYPE

BATCH_SIZE = 16
NUM_EPOCHS = 2000

CONFIG = dict(fp16=dict(mode=AMP_TYPE.TORCH))
