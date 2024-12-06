from tensorflow.keras.models import load_model, save_model

# Load the model
model = load_model('lstm_model.h5')

# Resave the model in a newer format
model.save('lstm_model_updated.h5')
