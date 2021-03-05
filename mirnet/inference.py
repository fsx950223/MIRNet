import numpy as np
from PIL import Image
import tensorflow as tf
from .model import mirnet_model
from .utils import closest_number


class Inferer:

    def __init__(self):
        self.model = None

    @staticmethod
    def download_weights(file_id: str):
        tf.keras.utils.get_file(
            'low_light_weights_best.h5', 'https://drive.google.com/uc?id={}'.format(file_id),
        )

    def build_model(self, num_rrg: int, num_mrb: int, channels: int, weights_path: str):
        self.model = mirnet_model(
            image_size=None, num_rrg=num_rrg,
            num_mrb=num_mrb, channels=channels
        )
        self.model.load_weights(weights_path)

    def _predict(self, original_image, image_resize_factor: float = 1.):
        height, width = original_image.shape[0:2]
        target_height, target_width = (
            closest_number(height // image_resize_factor, 4),
            closest_number(width // image_resize_factor, 4)
        )
        original_image = tf.image.resize(original_image, (target_height, target_width), antialias=True)
        # image = tf.keras.preprocessing.image.img_to_array(original_image)
        image = tf.image.convert_image_dtype(original_image, tf.float32)
        # image = image.astype('float32') / 255.0
        image = tf.expand_dims(image, axis=0)
        output = self.model(image)
        output_image = output[0] * 255.0
        output_image = tf.clip_by_value(output_image, 0, 255)
        output_image = tf.reshape(output_image,
            (output_image.shape[0], output_image.shape[1], 3)
        )

        return output_image

    def infer(self, image_path, image_resize_factor: float = 1.):
        original_image = np.array(Image.open(image_path))
        output_image = self._predict(original_image, image_resize_factor)
        output_image = Image.fromarray(np.uint8(output_image))
        original_image = Image.fromarray(np.uint8(original_image))
        return original_image, output_image

    def infer_streamlit(self, image_pil, image_resize_factor: float = 1.):
        original_image = image_pil
        output_image = self._predict(original_image, image_resize_factor)
        return original_image, output_image
