import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
# Путь к папке с изображениями для обучения
train_dir = 'contour_shapes'

# Создание генератора изображений
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,  # Диапазон градусов для случайных поворотов
    width_shift_range=0.2,  # Случайное горизонтальное смещение
    height_shift_range=0.2,  # Случайное вертикальное смещение
    shear_range=0.2,  # Интенсивность сдвига
    zoom_range=0.2,  # Интенсивность зума
    horizontal_flip=True,  # Случайное отражение входов по горизонтали
)


train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),
    color_mode='grayscale',
    class_mode='categorical',
    subset='validation'
)


model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Добавление слоя Dropout
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))  # Добавление слоя Dropout
model.add(Flatten())
model.add(Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Добавление слоя Dropout
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.5))  # Добавление слоя Dropout
model.add(Dense(3, activation='softmax'))  # 3 класса: круг, квадрат, треугольник

# Компиляция модели
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])

# Создание callback для ранней остановки
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# Обучение модели
model.fit(train_generator,
          epochs=3,
          verbose=1,
          validation_data=validation_generator,
          callbacks=[early_stopping])
model.save('model3.h5')