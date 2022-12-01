import tensorflow as tf

base_dir = "/media/cbanbury/T7/jquaye/resources/"

assets = [
  ("http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz", base_dir + "speech_commands")
, ("https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/multilingual_context_73_0.8011.tar.gz", base_dir + "embedding_model")
, ("https://github.com/harvard-edge/multilingual_kws/releases/download/v0.1-alpha/unknown_files.tar.gz", base_dir + "unknown_files")
]

for asset,cache in assets:
    tf.keras.utils.get_file(origin=asset, untar=True, cache_subdir=cache)

base_model = tf.keras.models.load_model( base_dir + "embedding_model/multilingual_context_73_0.8011")
embedding = tf.keras.models.Model(
    name="embedding_model",
    inputs=base_model.inputs,
    outputs=base_model.get_layer(name="dense_2").output,
)
embedding.trainable = False