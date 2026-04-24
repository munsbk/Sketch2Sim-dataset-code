import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

# === Import your full AI pipeline from model.py ===
from model import (
    load_image,
    magic_filter,
    detect_components,
    detect_text_regions,
    apply_wire_cleanup,
    remove_detected_regions,
    isolate_wires,
    detect_and_label_wires,
    overlay_nodes,
    assign_text_to_components,
    find_ground_wire,
    map_components_to_wires,
    annotate_final_combined,
    generate_netlist
)

# ==========================================================
# Flask App Configuration
# ==========================================================
app = Flask(__name__, template_folder="templates", static_folder="static")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ==========================================================
# Helper: Convert absolute path to web-accessible static URL
# ==========================================================
def to_web_path(path):
    """Convert Windows/local file path to a proper Flask static web path."""
    if not path:
        return None

    # Normalize backslashes to forward slashes
    path = os.path.normpath(path).replace("\\", "/")

    # Always ensure it points under /static/
    if path.startswith("static/"):
        return "/" + path
    elif "/static/" in path:
        return "/" + path.split("/static/")[-1].join(["static/", ""])
    else:
        # fallback, force into /static/uploads/
        filename = os.path.basename(path)
        return f"/static/uploads/{filename}"



# ==========================================================
# Core AI Processing Function
# ==========================================================
def main(image_path):
    """Run the full AI pipeline on an uploaded circuit image."""
    print(f"Processing image: {image_path}")

    # Step 1: Preprocess and enhance
    image, _ = load_image(image_path)
    processed_path = magic_filter(image_path)  # enhanced version
    image, _ = load_image(processed_path)

    # Step 2: Detect components and text
    component_boxes = detect_components(image)
    text_boxes = detect_text_regions(image)

    # Step 3: Detect and isolate wires
    wire_cleaned = apply_wire_cleanup(image)
    all_boxes = [c[:4] for c in component_boxes] + [t[:4] for t in text_boxes]
    masked_image = remove_detected_regions(wire_cleaned.copy(), all_boxes)
    wires_edges = isolate_wires(masked_image)
    nodes_img, wire_info = detect_and_label_wires(wires_edges, masked_image)

    # Step 4: Overlay nodes and assign OCR text
    final_img = overlay_nodes(image.copy(), wire_info)
    assignments, text_data = assign_text_to_components(component_boxes, text_boxes, image)

    # Step 5: Map components to wires
    ground_wire_id = find_ground_wire(wire_info)
    mapping, determined_ground_wire_id = map_components_to_wires(
        [c[:4] for c in component_boxes], wire_info, ground_wire_id
    )

    # Step 6: Annotate and generate netlist
    final_img, annotated_path = annotate_final_combined(final_img, assignments, mapping)
    netlist, _ = generate_netlist(assignments, mapping, determined_ground_wire_id)

    # Return both image paths for web display
    return [netlist, processed_path, annotated_path]


# ==========================================================
# Flask Routes
# ==========================================================
@app.route("/", methods=["GET", "POST"])
def index():
    """Home page with upload form."""
    if request.method == "POST":
        f = request.files.get("file")
        if not f or f.filename == "":
            return render_template("index.html", error="No file selected", active_page="home")

        # Save uploaded file
        filename = secure_filename(f.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        f.save(save_path)

        # Run AI pipeline
        result = main(save_path)

        # 🧩 Debug print — shows exactly what paths your model returns
        print("DEBUG RESULT PATHS:", result)

        # Convert to web paths (so Flask can show images)
        result[1] = to_web_path(result[1])  # Processed image
        result[2] = to_web_path(result[2])  # Annotated image

        # 🧩 Debug print after path conversion
        print("DEBUG WEB PATHS:", result)

        return render_template("index.html", result=result, success=True, active_page="home")

    # GET method: just render upload form
    return render_template("index.html", active_page="home")


@app.route("/About")
def about():
    return render_template("about.html", active_page="about")


@app.route("/Services")
def services():
    # You can later replace with services.html if added
    return render_template("about.html", active_page="services")


@app.route("/Contact")
def contact():
    # You can later replace with contact.html if added
    return render_template("about.html", active_page="contact")


# ==========================================================
# Run Server
# ==========================================================
if __name__ == "__main__":
    app.run(debug=True)
