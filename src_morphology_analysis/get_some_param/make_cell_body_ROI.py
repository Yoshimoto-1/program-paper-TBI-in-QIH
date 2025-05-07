import napari
import json
import os
from skimage.io import imread

def save_shapes_json(viewer, sample_name, save_path, layer_name):
    """Save a specific Shapes layer in Napari as JSON"""
    shapes_layer = None
    
    # Gets the Shapes layer that matches the specified layer_name.
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Shapes) and layer.name == layer_name:
            shapes_layer = layer
            break 

    if shapes_layer and len(shapes_layer.data) > 0:  
        roi_data = [shape.tolist() for shape in shapes_layer.data]  

        save_dict = {
            "sample_name": sample_name,
            "shapes_layer": layer_name,  
            "cell_body_rois": roi_data  
        }

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4)
        print(f"Saved ROI to: {save_path}")
    else:
        print(f"! No ROI created for {sample_name} in {layer_name}, skipping JSON save.")


# save directory
save_ROI_dir = r"ROI_json_cell_body"
os.makedirs(save_ROI_dir, exist_ok=True)

# image directory
image_dir = r"ROI_extracted_regions"

image_files = [os.path.join(image_dir, tif) for tif in os.listdir(image_dir) if tif.endswith(".tif")]

for image_path in image_files:
    image_file = os.path.basename(image_path)
    sample_name = image_file.split(".")[0]
    image = imread(image_path)

    # Napari
    viewer = napari.Viewer()
    viewer.add_image(image, name=f"{sample_name}_Original Image")

    # **Create a Shapeslayer (with ROIs already loaded and added)**
    save_shapes_layer_path = os.path.join(save_ROI_dir, f'{sample_name}_cell-body-roi.json')

    existing_rois = []
    # Load existing ROI file if available
    if os.path.exists(save_shapes_layer_path):
        with open(save_shapes_layer_path, "r") as f:
            roi_data = json.load(f)
            existing_rois = roi_data.get("roi", []) 


    # Add existing ROI if you have one
    shapes_layer = viewer.add_shapes(existing_rois, shape_type='polygon', name=f'{sample_name} cell body', opacity=0.3)
    shapes_layer.mode = 'add_polygon'

    napari.run()

    # Save JSON for each Shapes layer
    layer_name = f'{sample_name} cell body'
    save_shapes_layer_path = os.path.join(save_ROI_dir, f'{sample_name}_cell-body-roi.json')
    save_shapes_json(viewer, sample_name, save_shapes_layer_path, layer_name)
