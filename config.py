BREAST={}
BREAST['crop']=[0.01, 0.01, 0.04, 0.04]
BREAST['binary']={'thresh': 0.01, 'maxval': 1.0}
BREAST['mask']={'ksize': (23, 23), 'operation': 'open'}

CARDIO={}
CARDIO['crop']=[0.06, 0.06, 0.2, 0.12]

PNEU={}
PNEU['crop']=[0.05, 0.05, 0.05, 0.17]

IMG_SHAPE=224*2
NBR_CLASSES=2
SPLIT=0.8

LR=0.005

TRAIN_WITH_DATAGEN=True
TRAIN_WITH_MIXUP=True

COMMON={}
COMMON['crop']=[0.03, 0.03, 0.03, 0.03]

DATASET=['atelectasis', 'cap', 'covid', 'normal', 'pneu']
