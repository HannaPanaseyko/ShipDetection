import tensorflow as tf

model = tf.keras.models.load_model('G:\Development\Python\ShipDetectionTF\ExampleMasks\TrainedModel_768.keras')
tf.saved_model.save(model, "Models/TrainedModel_768")