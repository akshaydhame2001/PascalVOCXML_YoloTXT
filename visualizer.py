import cv2
import os
import xml.etree.ElementTree as ET

def parse_pascal_voc(xml_file):
    """Parse Pascal VOC XML annotations."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        bndbox = obj.find("bndbox")
        xmin = int(bndbox.find("xmin").text)
        ymin = int(bndbox.find("ymin").text)
        xmax = int(bndbox.find("xmax").text)
        ymax = int(bndbox.find("ymax").text)
        boxes.append((name, xmin, ymin, xmax, ymax))
    return boxes

def parse_yolo_txt(txt_file, img_width, img_height):
    """Parse YOLO TXT annotations."""
    boxes = []
    with open(txt_file, "r") as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            xmin = int((x_center - width / 2) * img_width)
            ymin = int((y_center - height / 2) * img_height)
            xmax = int((x_center + width / 2) * img_width)
            ymax = int((y_center + height / 2) * img_height)
            boxes.append((int(class_id), xmin, ymin, xmax, ymax))
    return boxes

def visualize_annotations(image_path, annotations, output_path=None):
    """Draw bounding boxes and annotations on the image."""
    image = cv2.imread(image_path)
    for ann in annotations:
        label, xmin, ymin, xmax, ymax = ann
        # Draw bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Add label text
        cv2.putText(image, str(label), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("Annotated Image", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage:
# Pascal VOC
image_file = "31.jpg"
#xml_file = "31.xml"
#annotations = parse_pascal_voc(xml_file)
#visualize_annotations(image_file, annotations, "output_voc.jpg")

# YOLO
txt_file = "31.txt"
image = cv2.imread(image_file)
img_height, img_width = image.shape[:2]
annotations = parse_yolo_txt(txt_file, img_width, img_height)
visualize_annotations(image_file, annotations, "output_txt.jpg")
