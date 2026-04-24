import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
import time
import re
# from google.colab.patches import cv2_imshow
# from google.colab import files
import math
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize, remove_small_objects, binary_erosion
from PIL import Image, ImageDraw, ImageFont
from font_roboto import Roboto


component_model_path = "./model/YOLO_element_best.pt"  # YOLO model for components
text_model_path      = "./model/YOLO_text_best.pt"     # YOLO model for text regions
crnn_model_path      = "./model/crnn_inference_model.h5"  # CRNN model

component_model = YOLO(component_model_path)
text_model      = YOLO(text_model_path)
crnn_model      = tf.keras.models.load_model(crnn_model_path, compile=False)

# Character set used during CRNN training (must match training configuration)
CHARSET = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZΩµ-.∠"




def load_image(img_url):
    """
    Upload an image and crop ~5% from each side.
    This cropped image will be used for all subsequent processing.
    """
    # cropped = '' 
    # image_path = 'helo'
    # uploaded = files.upload()
    # image_path = list(uploaded.keys())[0]
    image_path = img_url
    image = cv2.imread(image_path)
    # image = cv2.imread(image_path)
    h, w = image.shape[:2]
    crop_p = 0.02  # 5% crop
    x_start = int(w * crop_p)
    y_start = int(h * crop_p)
    x_end   = int(w * (1 - crop_p))
    y_end   = int(h * (1 - crop_p))
    cropped = image[y_start:y_end, x_start:x_end]
    return cropped, image_path

def apply_nms(boxes, confidences, iou_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to avoid overlapping bounding boxes.
    Returns indices of the boxes to keep.
    """
    rects = [ [x1, y1, x2 - x1, y2 - y1] for x1, y1, x2, y2, _, _ in boxes ]  # Convert to x, y, w, h
    scores = [conf for _, _, _, _, conf, _ in boxes]
    indices = cv2.dnn.NMSBoxes(rects, scores, score_threshold=0.3, nms_threshold=iou_threshold)

    # Check if indices is not None and reshape it to a 2D array if necessary
    if indices is not None and len(indices) > 0:
        # Reshape indices to ensure it's a 2D array of shape (N, 1) or similar
        # This handles cases where NMSBoxes might return a flat array of scalars
        indices = indices.reshape(-1, 1)
        # Now iterate through the reshaped indices
        return [i[0] for i in indices]
    else:
        return []


def decode_prediction(pred):
    """
    Decode the prediction output from CRNN.
    """
    pred = np.squeeze(pred)
    best_indices = list(np.argmax(pred, axis=1))
    decoded, prev = "", -1
    for idx in best_indices:
        if idx != prev and idx < len(CHARSET):
            decoded += CHARSET[idx]
        prev = idx
    return decoded

def preprocess_for_crnn(image, img_size=(128, 32)):
    """
    Preprocess an image for CRNN inference.
    """
    image = cv2.fastNlMeansDenoising(image, None, h=5, templateWindowSize=7, searchWindowSize=21)
    image = cv2.resize(image, img_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=-1)  # add channel dimension (grayscale)
    image = np.expand_dims(image, axis=0)     # add batch dimension
    return image



def detect_components(image):
    """
    Detect components using the YOLO component model.
    Applies NMS to avoid duplicate detections.
    Returns a list of tuples: (x1, y1, x2, y2, confidence, class_id)
    """
    image_resized = cv2.resize(image, (640, 640))
    results = component_model(image_resized)
    boxes   = results[0].boxes.xywh.cpu().numpy()
    confs   = results[0].boxes.conf.cpu().numpy()
    cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    orig_h, orig_w = image.shape[:2]
    raw_boxes = []

    for i, box in enumerate(boxes):
        x_center, y_center, w_box, h_box = box
        x1 = int((x_center - w_box/2) * orig_w / 640)
        y1 = int((y_center - h_box/2) * orig_h / 640)
        x2 = int((x_center + w_box/2) * orig_w / 640)
        y2 = int((y_center + h_box/2) * orig_h / 640)
        raw_boxes.append((x1, y1, x2, y2, confs[i], cls_ids[i]))

    # Apply Non-Maximum Suppression
    keep_indices = apply_nms(raw_boxes, [b[4] for b in raw_boxes])
    component_boxes = [raw_boxes[i] for i in keep_indices]
    return component_boxes


def detect_text_regions(image):
    """
    Detect text regions using the YOLO text model.
    Returns a list of tuples: (x1, y1, x2, y2, confidence)
    """
    image_resized = cv2.resize(image, (640, 640))
    results = text_model(image_resized)
    boxes   = results[0].boxes.xywh.cpu().numpy()
    confs   = results[0].boxes.conf.cpu().numpy()
    orig_h, orig_w = image.shape[:2]
    text_boxes = []

    for i, box in enumerate(boxes):
        x_center, y_center, w_box, h_box = box
        x1 = int((x_center - w_box/2) * orig_w / 640)
        y1 = int((y_center - h_box/2) * orig_h / 640)
        x2 = int((x_center + w_box/2) * orig_w / 640)
        y2 = int((y_center + h_box/2) * orig_h / 640)
        text_boxes.append((x1, y1, x2, y2, confs[i]))
    
    return text_boxes

def crop_region(image, box):
    """
    Crop a region based on the bounding box.
    """
    x1, y1, x2, y2 = box[:4]
    return image[y1:y2, x1:x2]

def assign_text_to_components(component_boxes, text_boxes, image):
    """
    For each detected text region, run OCR and assign the recognized text
    to the nearest component (using box centers).
    Returns a list of tuples: (component_box, assigned_text, distance)
    """
    text_data = []
    for tb in text_boxes:
        x1, y1, x2, y2, conf = tb
        cropped = crop_region(image, tb)
        if len(cropped.shape) == 3:
            cropped_gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        else:
            cropped_gray = cropped
        ocr_img = preprocess_for_crnn(cropped_gray)
        preds = crnn_model.predict(ocr_img)
        text = decode_prediction(preds[0])
        m = re.search(r"[\d\.Ωµ∠]+", text)
        value = m.group() if m else None
        center = ((x1+x2)//2, (y1+y2)//2)
        text_data.append({"box": (x1, y1, x2, y2), "center": center, "text": text, "value": value})

    component_assignments = []
    for comp in component_boxes:
        cx1, cy1, cx2, cy2, conf, cls = comp
        comp_center = ((cx1+cx2)//2, (cy1+cy2)//2)
        min_dist = float('inf')
        assigned_text = None
        for td in text_data:
            if td["value"] is not None:
                dist = np.sqrt((comp_center[0]-td["center"][0])**2 + (comp_center[1]-td["center"][1])**2)
                if dist < min_dist:
                    min_dist = dist
                    assigned_text = td["text"]
        component_assignments.append((comp, assigned_text, min_dist))
    return component_assignments, text_data
from skimage import measure
def apply_wire_cleanup(image):
    """
    Keeps only the largest connected wire-like structure and returns
    a clean white-background image with only main wires in black.
    """
    original_height, original_width = image.shape[:2]

    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    labels = measure.label(thresh, connectivity=2)
    properties = measure.regionprops(labels)

    area_threshold = max([prop.area for prop in properties]) if properties else 0

    mask = np.zeros_like(thresh)
    for prop in properties:
        if prop.area >= area_threshold:
            coords = prop.coords
            mask[coords[:, 0], coords[:, 1]] = 255

    # Create white background, place black wires where mask is white
    clean_output = np.ones((original_height, original_width, 3), dtype=np.uint8) * 255
    clean_output[mask == 255] = [0, 0, 0]

    return clean_output





# # =============================================================================
# # Pipeline 2: Wire (Node) Detection & Component-to-Wire Mapping
# # =============================================================================

# def detect_boxes(image, model):
#     """
#     Detect boxes using a given YOLO model.
#     Returns a list of boxes in (x1, y1, x2, y2) format.
#     """
#     image_resized = cv2.resize(image, (640, 640))
#     results = model(image_resized)
#     boxes   = results[0].boxes.xywh.cpu().numpy()
#     orig_h, orig_w = image.shape[:2]
#     processed_boxes = []
#     for box in boxes:
#         x_center, y_center, w_box, h_box = box
#         x1 = int((x_center - w_box/2) * orig_w / 640)
#         y1 = int((y_center - h_box/2) * orig_h / 640)
#         x2 = int((x_center + w_box/2) * orig_w / 640)
#         y2 = int((y_center + h_box/2) * orig_h / 640)
#         processed_boxes.append((x1, y1, x2, y2))
#     return processed_boxes

def remove_detected_regions(image, regions, fill_color=(255,255,255)):
    """
    Mask out the given rectangular regions from the image.
    """
    for region in regions:
        x1, y1, x2, y2 = region
        cv2.rectangle(image, (x1,y1), (x2,y2), fill_color, -1)
    return image

def isolate_wires(image):
    """
    Isolate wires via Canny edge detection and dilation.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    return edges_dilated

def find_endpoints_iterative(skel):
    """
    Iteratively find endpoints from a skeleton and merge closely located points.
    """
    endpoints = []
    junctions = []
    h, w = skel.shape
    for i in range(h):
        for j in range(w):
            if skel[i,j]:
                count = 0
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di==0 and dj==0:
                            continue
                        ni, nj = i+di, j+dj
                        if 0<=ni<h and 0<=nj<w and skel[ni,nj]:
                            count += 1
                if count == 1:
                    endpoints.append((j,i))  # (x, y)
                elif count >= 3:
                    junctions.append((j,i))
    filtered = []
    thresh = 5
    for ep in endpoints:
        if all(math.hypot(ep[0]-jp[0],ep[1]-jp[1])>=thresh for jp in junctions):
            filtered.append(ep)
    if filtered:
        filtered = np.array(filtered, dtype=np.float32)
        clustering = DBSCAN(eps=8, min_samples=1).fit(filtered)
        labels = clustering.labels_
        final_points = []
        for label in set(labels):
            pts = filtered[labels==label]
            rep = np.mean(pts, axis=0)
            final_points.append((int(round(rep[0])), int(round(rep[1]))))
        return final_points
    else:
        return []

def morphologically_filter(skel):
    """
    Remove small artifacts via binary erosion and small object removal.
    """
    # Change 'structure' to 'footprint'
    eroded = binary_erosion(skel, footprint=np.ones((3,3)))
    filtered = remove_small_objects(eroded, min_size=50)
    return filtered

def detect_and_label_wires(wires_img, base_image, area_threshold=100):
    """
    Detect wires (nodes) in the wires image by finding contours.
    Annotates (draws) both wire endpoints and wire numbers.
    Returns the annotated image and a list of wire info.
    """
    contours, _ = cv2.findContours(wires_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    annotated_img = base_image.copy()
    wire_info = []
    wire_id = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_threshold:
            approx = cv2.approxPolyDP(cnt, epsilon=2, closed=True)
            x, y, w, h = cv2.boundingRect(approx)
            current_wire = {'id': wire_id, 'bounding_box': (x, y, x+w, y+h), 'area': area, 'endpoints': []}
            wire_mask = np.zeros(wires_img.shape, dtype=np.uint8)
            cv2.drawContours(wire_mask, [cnt], -1, 255, thickness=cv2.FILLED)
            wire_mask_bool = wire_mask > 0
            filtered_skel = morphologically_filter(wire_mask_bool)
            skel = skeletonize(filtered_skel)
            endpoints = find_endpoints_iterative(skel)
            endpoints_info = []
            for i_ep, (ex, ey) in enumerate(endpoints):
                cv2.circle(annotated_img, (ex, ey), 4, (0,255,0), -1)
                label = f"W{wire_id}_{i_ep}"
                cv2.putText(annotated_img, label, (ex+5, ey-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                endpoints_info.append({'label': label, 'coord': (ex, ey)})
            current_wire['endpoints'] = endpoints_info
            cx = x + w//2
            cy = y + h//2
            cv2.putText(annotated_img, f"n{wire_id}", (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
            wire_info.append(current_wire)
            wire_id += 1
    return annotated_img, wire_info

def find_ground_wire(wire_info):
    """Identify the bottom-most wire based on bounding box position"""
    if not wire_info:
        return None
    ground_wire = max(wire_info, key=lambda w: w['bounding_box'][3])  # index 3 is y2 (bottom y)
    return ground_wire['id']

def map_components_to_wires(component_boxes, wire_info, ground_wire_id=None):
    """
    For each component (given as a 4-tuple box) compute midpoints of each side;
    then, using wire endpoints, select the two nearest wires.
    Returns a dictionary mapping: { component_id: (wire_id1, wire_id2) }.
    """
    wire_endpoints = []
    for wire in wire_info:
        for ep in wire['endpoints']:
            wire_id = wire['id']
            # Replace ground wire ID with 0 for endpoints
            if ground_wire_id is not None and wire_id == ground_wire_id:
                wire_id = 0
            wire_endpoints.append((wire_id, ep['coord']))
    mapping = {}
    # Add this print header
    print("\nComponent Midpoint-to-Wire Mapping:")
    print("Component | Side  | Midpoint (x,y) | Nearest Wire ID | Distance")
    print("---------------------------------------------------------------")

    for comp_id, box in enumerate(component_boxes, start=1):
        x1, y1, x2, y2 = box
        midpoints = {'left': (x1, (y1+y2)//2), 'top': ((x1+x2)//2, y1),
                    'right': (x2, (y1+y2)//2), 'bottom': ((x1+x2)//2, y2)}

        for side, mid in midpoints.items():
            min_dist = float('inf')
            best_wire = None
            for wire_id, coord in wire_endpoints:
                d = math.hypot(mid[0]-coord[0], mid[1]-coord[1])

                # Print details for each midpoint
                if d < min_dist:
                    min_dist = d
                    best_wire = wire_id

            # Add this print statement
            print(f"Comp {comp_id:2} | {side:6} | ({mid[0]:4}, {mid[1]:4}) | Wire {best_wire:3} | {min_dist:.2f} px")
    for comp_id, box in enumerate(component_boxes, start=1):
        x1, y1, x2, y2 = box
        midpoints = {
            'left': (x1, (y1+y2)//2),
            'top': ((x1+x2)//2, y1),
            'right': (x2, (y1+y2)//2),
            'bottom': ((x1+x2)//2, y2)
        }
        candidate_by_wire = {}
        for side, mid in midpoints.items():
            min_dist = float('inf')
            best_wire = None
            for wire_id, coord in wire_endpoints:
                d = math.hypot(mid[0]-coord[0], mid[1]-coord[1])
                if d < min_dist:
                    min_dist = d
                    best_wire = wire_id
            if best_wire is not None:
                if best_wire in candidate_by_wire:
                    candidate_by_wire[best_wire] = min(candidate_by_wire[best_wire], min_dist)
                else:
                    candidate_by_wire[best_wire] = min_dist
        candidates = list(candidate_by_wire.items())
        candidates.sort(key=lambda x: x[1])
        selected = tuple(wire_id for wire_id, d in candidates[:2])
        mapping[comp_id] = selected
    return mapping, ground_wire_id

def overlay_nodes(image, wire_info):
    """
    On a copy of the image, draw nodes (wire endpoints and wire numbers)
    using the wire_info obtained from detect_and_label_wires.
    """
    annotated = image.copy()
    for wire in wire_info:
        for ep in wire['endpoints']:
            ex, ey = ep['coord']
            cv2.circle(annotated, (ex, ey), 4, (0,255,0), -1)
            cv2.putText(annotated, ep['label'], (ex+5, ey-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
        x, y, x2, y2 = wire['bounding_box']
        cx = x + (x2 - x)//2
        cy = y + (y2 - y)//2
        cv2.putText(annotated, f"n{wire['id']}", (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return annotated

def annotate_final_combined(image, assignments, mapping):
    
    """
    Annotate the image using Pillow so that special Unicode characters are rendered properly.
    For each component, draw its bounding box and a label in the format:
        "[ClassLabel][Number], [OCR Value] ([node1],[node2])"
    """
    # Class ID to component prefix mapping
    class_to_prefix = {
        0: 'V',    # DC Voltage Source
        1: 'R',    # Resistor
        2: 'I',    # Current Source
        3: 'L',    # Inductor
        4: 'C',    # Capacitor
        5: 'ACV'   # AC Voltage Source
    }
    counts = {cls: 0 for cls in class_to_prefix.keys()}  # Track counts per class

    # Convert the OpenCV image (BGR) to a PIL Image (RGB)
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    draw = ImageDraw.Draw(pil_img)

    # font_path = os.path.join(os.path.dirname(__file__), '/static/font', 'DejaVu_Sans/DejaVuSans-Bold.ttf')
    # print(font_path)
    # font = ImageFont.load_default()

    font_path = "static/font/static/Roboto-Regular.ttf"
    font =  ImageFont.truetype(font_path, 50)

    print("HELLO****************************************")
    for idx, (comp, text, dist) in enumerate(assignments):
        x1, y1, x2, y2, conf, cls_id = comp
        cls_id = int(cls_id)

        # Get class-based prefix and increment count
        prefix = class_to_prefix.get(cls_id, '?')
        counts[cls_id] += 1
        component_label = f"{prefix}{counts[cls_id]}"

        # Get node connections using original component index
        component_index = idx + 1  # Mapping uses 1-based indices
        nodes_tuple = mapping.get(component_index, ())
        nodes_str = ",".join(str(n) for n in nodes_tuple) if nodes_tuple else ""

        # Prepare label text
        ocr_text = text if text is not None else "No Value"
        label = f"{component_label}, {ocr_text} ({nodes_str})"

        # Draw component bounding box and label
        draw.rectangle([x1, y1, x2, y2], outline="green", width=3)
        draw.text((x1, y1 - 30), label, font=font, fill="red")

    # Convert back to OpenCV format (BGR)
    final_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    

    save_dir='static/uploads'
    filename = f"annotated_{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, final_img)
    final_img_path = f"{save_path.replace(os.sep, '\\')}"

    return final_img, final_img_path

def generate_netlist(assignments, mapping, ground_wire_id=None):  # Modified signature
    netlist_lines = [
        
    ]
    """
    Generate SPICE-compatible netlist from component data and node connections.
    Returns:
        str: Netlist following SPICE syntax
        list: List of component entries for verification
    """
    # Component type to SPICE prefix mapping
    COMPONENT_MAP = {
        0: 'V',  # DC Voltage Source
        1: 'R',  # Resistor
        2: 'I',  # Current Source
        3: 'L',  # Inductor
        4: 'C',  # Capacitor
        5: 'V'   # AC Voltage Source (treated as V with AC spec)
    }

    # Unit conversion for common suffixes
    UNIT_CONVERSION = {
        'Ω': 1,     'kΩ': 1e3,   'MΩ': 1e6,
        'F': 1,     'µF': 1e-6,  'nF': 1e-9,  'pF': 1e-12,
        'H': 1,     'mH': 1e-3,  'µH': 1e-6,
        'A': 1,     'mA': 1e-3,  'µA': 1e-6,
        'V': 1,     'kV': 1e3,   'mV': 1e-3
    }

    component_counts = {cls: 0 for cls in COMPONENT_MAP.keys()}
    
    component_entries = []

    for idx, (comp, text, dist) in enumerate(assignments):
        x1, y1, x2, y2, conf, cls_id = comp
        component_index = idx + 1  # 1-based index for mapping

        # Get node connections
        node1, node2 = mapping.get(component_index, (0, 0))

        # Get component type and count
        cls_id = int(cls_id)
        prefix = COMPONENT_MAP.get(cls_id, 'X')
        component_counts[cls_id] += 1
        comp_id = f"{prefix}{component_counts[cls_id]}"

        # Process component value
        value = "1"  # Default value
        if text:
            # Extract numerical value with possible unit
            match = re.search(r"([\d\.]+)([a-zA-ZΩµ]*)\b", text)
            if match:
                num, unit = match.groups()
                multiplier = UNIT_CONVERSION.get(unit, 1)
                try:
                    value = str(float(num) * multiplier)
                except:
                    value = num

        # Special handling for different component types
        if cls_id == 0:  # DC Voltage Source
            netlist_line = f"{comp_id} {node1} {node2} DC {value}"
        elif cls_id == 5:  # AC Voltage Source
            netlist_line = f"{comp_id} {node1} {node2} AC {value} 50"  # Default 50Hz
        elif cls_id == 2:  # Current Source
            netlist_line = f"{comp_id} {node1} {node2} {value}"
        else:  # Passive components (R, L, C)
            netlist_line = f"{comp_id} {node1} {node2} {value}"

        component_entries.append({
            'id': comp_id,
            'type': prefix,
            'nodes': (node1, node2),
            'value': value,
            'raw_text': text
        })
        netlist_lines.append(netlist_line)

    

    return '\n'.join(netlist_lines), component_entries


# # def visualize_wire_mapping(base_image, wire_info, connection_threshold=10):
# #     """
# #     Create visualization showing:
# #     - Wire endpoints with labels
# #     - Wire-to-wire connections (nodes)
# #     - Wire ID numbers
# #     Returns annotated image and connection list
# #     """
# #     annotated = base_image.copy()
# #     h, w = annotated.shape[:2]

# #     # Collect all endpoints with wire IDs
# #     all_endpoints = []
# #     for wire in wire_info:
# #         for ep in wire['endpoints']:
# #             all_endpoints.append({
# #                 'coord': ep['coord'],
# #                 'wire_id': wire['id'],
# #                 'label': ep['label']
# #             })

# #     # Find connections between wires (shared nodes)
# #     connections = []
# #     for i in range(len(all_endpoints)):
# #         for j in range(i+1, len(all_endpoints)):
# #             dx = all_endpoints[i]['coord'][0] - all_endpoints[j]['coord'][0]
# #             dy = all_endpoints[i]['coord'][1] - all_endpoints[j]['coord'][1]
# #             distance = math.hypot(dx, dy)

# #             if distance < connection_threshold:
# #                 connections.append((
# #                     all_endpoints[i]['label'],
# #                     all_endpoints[j]['label']
# #                 ))

# #                 # Draw connection line
# #                 cv2.line(annotated,
# #                         all_endpoints[i]['coord'],
# #                         all_endpoints[j]['coord'],
# #                         (0,255,255), 2)  # Yellow connection lines

# #     # Draw endpoints and labels
# #     for ep in all_endpoints:
# #         x, y = ep['coord']
# #         # Draw endpoint circle
# #         cv2.circle(annotated, (x,y), 6, (0,255,0), -1)  # Green circles
# #         # Draw endpoint label
# #         cv2.putText(annotated, ep['label'], (x+10, y-10),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

# #     # Draw wire IDs in center of wire regions
# #     for wire in wire_info:
# #         x1, y1, x2, y2 = wire['bounding_box']
# #         cx = (x1 + x2) // 2
# #         cy = (y1 + y2) // 2
# #         cv2.putText(annotated, f"Wire {wire['id']}", (cx-30, cy),
# #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,255), 2)  # Magenta text
# #     # Highlight ground wire
# #     if ground_wire_id is not None:
# #         ground_wire = next(w for w in wire_info if w['id'] == ground_wire_id)
# #         x1, y1, x2, y2 = ground_wire['bounding_box']
# #         cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 3)  # Red rectangle
# #         cv2.putText(annotated, "GROUND", (x1, y1-10),
# #                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

# #     return annotated, connections
# import numpy as np
# # =============================================================================
# # Unified Pipeline: Multiple Outputs
# # =============================================================================



# # if __name__ == "__main__":
# #     main()

import uuid
import os

# Step 4: Simulate CamScanner "Magic Filter"
def magic_filter(image_file, save_dir='static/uploads'):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Load image
    image = Image.open(image_file)
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(thresh, -1, kernel)

    # Save image
    filename = f"{uuid.uuid4().hex[:8]}.jpg"
    save_path = os.path.join(save_dir, filename)
    cv2.imwrite(save_path, sharp)

    # Return the relative URL path to serve via Flask
    return f"{save_path.replace(os.sep, '\\')}"