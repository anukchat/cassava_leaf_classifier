MODEL_NAME = "cassava_efficient_net.ckpt"
TRAIN_IMAGES_DIR = '../input/cassava-leaf-disease-classification/train_images'
TEST_IMAGES_DIR = '../input/cassava-leaf-disease-classification/test_images'
TRAIN_CSV = '../input/cassava-leaf-disease-classification/train.csv'
PRETRAINED_PATH = 'ml/trained_model/efficientnet-b5.pth'
BATCH_SIZE = 8
IMG_SIZE = 512
CLASSES = 5
CLASS_CATEGORIES = ('Cassava Bacterial Blight (CBB)',
                    'Cassava Brown Streak Disease (CBSD)',
                    'Cassava Green Mottle (CGM)',
                    'Cassava Mosaic Disease (CMD)',
                    'Healthy')
