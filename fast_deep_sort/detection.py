import tensorflow as tf
import pathlib
import types
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile

def load_detector(model_name):



    tf_version = tf.__version__
    #tf2
    if tf_version.startswith('2.'): 
        # from https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb
        base_url = 'http://download.tensorflow.org/models/object_detection/'
        model_file = model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(
            fname=model_name, 
            origin=base_url + model_file,
            untar=True)

        model_dir = pathlib.Path(model_dir)/"saved_model"

        model = tf.saved_model.load(str(model_dir))
        model = model.signatures['serving_default']

        return model

    if tf_version.startswith('1.'):
        # from https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/camera.html
        MODEL_NAME = model_name
        MODEL_FILE = MODEL_NAME + '.tar.gz'
        DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'
        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

        # Number of classes to detect
        NUM_CLASSES = 90

        # Download Model
        opener = urllib.request.URLopener()
        opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
        tar_file = tarfile.open(MODEL_FILE)
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)
            if 'frozen_inference_graph.pb' in file_name:
                tar_file.extract(file, os.getcwd())


        # Load a (frozen) Tensorflow model into memory.
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        # Detection
        def __call__(self, image_batch_np):
            with detection_graph.as_default():
                with tf.Session(graph=detection_graph) as sess:
                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = image_batch_np
                    # Extract image tensor
                    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
                    # Extract detection boxes
                    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
                    # Extract detection scores
                    scores = detection_graph.get_tensor_by_name('detection_scores:0')
                    # Extract detection classes
                    classes = detection_graph.get_tensor_by_name('detection_classes:0')
                    # Extract number of detectionsd
                    num_detections = detection_graph.get_tensor_by_name(
                        'num_detections:0')
                    # Actual detection.
                    (boxes, scores, classes, num_detections) = sess.run(
                        [boxes, scores, classes, num_detections],
                        feed_dict={image_tensor: image_np_expanded})
                    return {'boxes':boxes, 'scores':scores, 'classes':classes, 'num_detections':num_detections}


        # from https://stackoverflow.com/questions/972/adding-a-method-to-an-existing-object-instance
        detection_graph.__call__ = types.MethodType( __call__, detection_graph )
        return detection_graph
