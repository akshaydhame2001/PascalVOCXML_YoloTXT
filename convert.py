import os
import xml.etree.ElementTree as ET
from pathlib import Path

def convert_voc_to_yolo(xml_dir, img_dir, output_dir, classes):
    """
    Convert Pascal VOC XML annotations to YOLOv8 TXT format.

    Parameters:
    - xml_dir: Directory containing Pascal VOC XML annotation files.
    - img_dir: Directory containing images corresponding to the annotations.
    - output_dir: Directory to save YOLOv8 TXT annotation files.
    - classes: List of class names.
    """
    os.makedirs(output_dir, exist_ok=True)

    total_files = len([file for file in os.listdir(xml_dir) if file.endswith(".xml")])
    converted_files = 0
    failed_files = []

    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith(".xml"):
            continue

        try:
            # Parse XML file
            xml_path = os.path.join(xml_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image dimensions
            size = root.find("size")
            img_width = int(size.find("width").text)
            img_height = int(size.find("height").text)

            # Prepare YOLO annotation content
            yolo_annotations = []
            for obj in root.findall("object"):
                class_name = obj.find("name").text
                if class_name not in classes:
                    continue
                class_id = classes.index(class_name)

                bndbox = obj.find("bndbox")
                xmin = int(bndbox.find("xmin").text)
                ymin = int(bndbox.find("ymin").text)
                xmax = int(bndbox.find("xmax").text)
                ymax = int(bndbox.find("ymax").text)

                # Convert to YOLO format (normalized x_center, y_center, width, height)
                x_center = ((xmin + xmax) / 2) / img_width
                y_center = ((ymin + ymax) / 2) / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            if yolo_annotations:
                # Save YOLO annotations to TXT file
                txt_file = os.path.join(output_dir, Path(xml_file).stem + ".txt")
                with open(txt_file, "w") as f:
                    f.write("\n".join(yolo_annotations))
                converted_files += 1
                print(f"[SUCCESS] Converted: {xml_file} -> {txt_file}")
            else:
                failed_files.append(xml_file)
                print(f"[WARNING] No valid annotations in: {xml_file}")

        except Exception as e:
            failed_files.append(xml_file)
            print(f"[ERROR] Failed to process {xml_file}: {e}")

    print("\n--- Conversion Summary ---")
    print(f"Total XML files: {total_files}")
    print(f"Successfully converted: {converted_files}")
    print(f"Failed conversions: {len(failed_files)}")
    if failed_files:
        print("Files failed to convert:")
        for file in failed_files:
            print(f" - {file}")

# Example usage
xml_directory = "E:\Download\drone\datasets\RealWorld\Drone_TrainSet_XMLs"
image_directory = "E:\Download\drone\datasets\RealWorld\Drone_TrainSet"
output_directory = os.path.join(os.getcwd(), "output")

class_list = ["drone"]  # Replace with your class names

convert_voc_to_yolo(xml_directory, image_directory, output_directory, class_list)
