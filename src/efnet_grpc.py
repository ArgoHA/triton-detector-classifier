import tritonclient.grpc as grpcclient
import numpy as np
import cv2
import sys


class Efnet_grpc():
    def __init__(self, classes,
                 url="localhost:8001",
                 model_name="efnet",
                 input_width=224,
                 input_height=224,
                 model_version="",
                 verbose=False, conf_thresh=0.7) -> None:

        self.model_name = model_name
        self.input_width = input_width
        self.input_height = input_height
        self.batch_size = 1
        self.conf_thresh = conf_thresh
        self.input_shape = [self.batch_size, self.input_height, self.input_width, 3]
        self.input_name = 'input_1' # name of input layer (from NN architecture)
        self.output_name = 'pred' # name of output layer
        self.output_size = 5
        self.triton_client = None
        self.classes = classes
        self.init_triton_client(url)
        self.test_predict()


    def init_triton_client(self, url):
        try:
            triton_client = grpcclient.InferenceServerClient(
                url=url,
                verbose=False,
                ssl=False,
            )
        except Exception as e:
            print("channel creation failed: " + str(e))
            sys.exit()
        self.triton_client = triton_client


    def test_predict(self):
        input_images = np.zeros((*self.input_shape,), dtype=np.float32)
        _ = self.predict(input_images)


    def predict(self, input_images):
        inputs = []
        outputs = []

        inputs.append(grpcclient.InferInput(self.input_name, [*input_images.shape], "FP32"))
        # Initialize the data
        inputs[-1].set_data_from_numpy(input_images)
        outputs.append(grpcclient.InferRequestedOutput(self.output_name))

        # Test with outputs
        results = self.triton_client.infer(
            model_name=self.model_name,
            inputs=inputs,
            outputs=outputs)

        # Get the output arrays from the results
        return results.as_numpy(self.output_name)


    def preprocessing(self, image):
        image = cv2.resize(image, dsize=(self.input_width, self.input_height), interpolation=cv2.INTER_CUBIC)
        image_fl = image.astype(np.float32)
        final_image = np.expand_dims(image_fl, axis=0)
        return final_image


    def classifier_pred(self, image):
        proc_image = self.preprocessing(image)
        pred = self.predict(proc_image)

        if np.max(pred) > self.conf_thresh:
            pred = np.argmax(pred)
        else:
            pred = len(self.classes) - 1
        return pred
