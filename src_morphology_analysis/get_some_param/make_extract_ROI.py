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
            "rois": roi_data 
        }

        with open(save_path, "w") as f:
            json.dump(save_dict, f, indent=4)
        print(f"Saved ROI to: {save_path}")
    else:
        print(f"! No ROI created for {sample_name} in {layer_name}, skipping JSON save.")

processed_images_dir = r"fft" # plese set your "fft" folda directory in "morphology_analysis_2d.ipynb" 

processed_images = [tif for tif in os.listdir(processed_images_dir) if tif.endswith(".tif")]

# save_path
save_ROI_dir = r"ROI_json"

os.makedirs(save_ROI_dir, exist_ok=True)

for i, processed_image_file in enumerate(processed_images):

    image_path = os.path.join(processed_images_dir, processed_image_file)
    sample_name = processed_image_file.split("_")[0]
    image = imread(image_path)

    # Create a save directory
    save_dir = os.path.join(save_ROI_dir, sample_name)
    os.makedirs(save_dir, exist_ok=True)

    # Napari
    viewer = napari.Viewer()
    viewer.add_image(image, name=f"{sample_name}_Original Image")

    # Create a Shapes layer
    for j in range(20):
        save_shapes_layer_path = os.path.join(save_dir, f"{j}_rois.json")

        existing_rois = []
        # Load existing ROI file if available
        if os.path.exists(save_shapes_layer_path):
            with open(save_shapes_layer_path, "r") as f:
                roi_data = json.load(f)
                existing_rois = roi_data.get("rois", []) 
        # Add existing ROI if you have one
        viewer.add_shapes(existing_rois, shape_type='polygon', name=f'{j}_Shapes', opacity=0.3)

    napari.run()

    # Save JSON for each Shapes layer
    for j in range(20):
        layer_name = f"{j}_Shapes"
        save_shapes_layer_path = os.path.join(save_dir, f"{j}_rois.json")
        save_shapes_json(viewer, sample_name, save_shapes_layer_path, layer_name)
