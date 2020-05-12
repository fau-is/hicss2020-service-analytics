from keras import backend
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

# confirm that tensorflow sees the GPU
assert 'GPU' in str(device_lib.list_local_devices())

# confirm that keras sees the GPU
assert len(backend.tensorflow_backend._get_available_gpus()) > 0