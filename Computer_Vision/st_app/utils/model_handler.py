from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_objects(self, image, confidence_threshold = 0.78):
        results = self.model.predict(image, conf = confidence_threshold)
        annotated_image = results[0].plot()  # YOLO method to get image with bounding boxes
        return results[0].boxes, results[0].names, annotated_image  # Return bounding boxes, names, and annotated image

    def detect_batch_objects(self, images, confidence_threshold=0.78, batch_size=8):
        """
        Detect objects in a batch of images.

        Args:
            images: List of images to process.
            confidence_threshold: Confidence threshold for predictions.
            batch_size: Number of images to process at a time.

        Returns:
            A list of results for each image in the batch.
        """
        all_results = []
        num_images = len(images)

        for i in range(0, num_images, batch_size):
            batch = images[i:i + batch_size]  # Create a batch of size `batch_size`
            results = self.model.predict(batch, conf=confidence_threshold)
            all_results.extend(results)  # Append results of the batch to the list

        return all_results
