from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd 

train_datagen = ImageDataGenerator(rescale=1/255,
    horizontal_flip=True,
    brightness_range=[0.5,1.5])
validation_datagen = ImageDataGenerator(rescale=1/255)

train_df = pd.read_csv('/data/embryo/method_3/train_grade.csv', dtype= str)
validation_df = pd.read_csv('/data/embryo/method_3/validation_grade.csv', dtype= str)


train_generator = train_datagen.flow_from_dataframe(
    dataframe = train_df,
    directory= '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'Second',
    batch_size=32,
    class_mode = 'categorical')

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe = validation_df,
    directory = '/data/embryo/method_3/',
    x_col = 'id',
    y_col = 'Second',
    batch_size=16,
    class_mode = 'categorical',
    shuffle=False)

labels = validation_generator.class_indices
labels = list(labels.keys())
print(labels)