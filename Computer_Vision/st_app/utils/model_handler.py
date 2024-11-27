from ultralytics import YOLO

class YOLOModel:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_objects(self, image, confidence_threshold = 0.78):
        results = self.model.predict(image, conf = confidence_threshold)
        annotated_image = results[0].plot()  # YOLO method to get image with bounding boxes
        return results[0].boxes, results[0].names, annotated_image  # Return bounding boxes, names, and annotated image
