import h5py
with h5py.File(r"lstm_model.h5", "r") as f:
    print(f.attrs["keras_version"])  # Prints Keras version used
    print(f.attrs["backend"])       # Prints TensorFlow backend info
