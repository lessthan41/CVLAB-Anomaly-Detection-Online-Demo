import gradio as gr
import numpy as np
import PIL.Image as Image
import os
import glob
import time
import cv2
import random
from CSAD.main import main as CSAD
from InstAD.main import main as InstAD
from ShiZhi.main import main as ShiZhi

mvtec_ad_root = "/mnt/sda2/tokichan/MVTec-AD"
mvtec_loco_root = "/mnt/sda2/tokichan/MVTec_LOCO"
visa_root = "/mnt/sda2/tokichan/VisA_highshot"


def greet(model, dataset, class_name, image):
    if model == "CSAD":

        anomaly_map, anomaly_score = CSAD(class_name, image)
    elif model == "InstAD":
        anomaly_map, anomaly_score = InstAD(class_name, image)
    else:
        anomaly_map, anomaly_score = ShiZhi(class_name, image)

    result_image = []
    result_paths = []

    # Define the text bar content
    text_bar = gr.Textbox(
        label=f"Anomaly Score:",
        value=f"{anomaly_score} / 100",
        elem_id="anomaly_score",
    )
    
    return text_bar, result_image[0]

def change_dataset(model):
    if model == "CSAD":
        return gr.update(choices=["MVTec LOCO"], value="MVTec LOCO")
    elif model == "InstAD":
        return gr.update(choices=["VisA"], value="VisA")
    else:
        return gr.update(choices=["MVTec AD"], value="MVTec AD")

def change_classes(dataset):
    if dataset == "MVTec LOCO":
        return gr.update(choices=["breakfast_box","pushpins","juice_bottle","screw_bag","splicing_connectors"], value="breakfast_box")
    elif dataset == "VisA":
        return gr.update(choices=["candle","capsules","macaroni1","macaroni2","tubes"], value="candle")
    else:
        return gr.update(choices=["bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper"], value="bottle")

def show_class_images(dataset_name,class_name):
    num_images = 6
    if dataset_name == "MVTec LOCO":
        dataset_root = mvtec_loco_root
        # test_logical_images = glob.glob(os.path.join(dataset_root, class_name, "test", "structural_anomalies", "*.png"))
        # test_structural_images = glob.glob(os.path.join(dataset_root, class_name, "test", "logical_anomalies", "*.png"))
        # all_test_images = test_logical_images + test_structural_images

        # show_images = random.sample(all_test_images, num_images)
        # return [Image.open(image_path) for image_path in show_images]


    elif dataset_name == "VisA":
        dataset_root = visa_root
    else:
        dataset_root = mvtec_ad_root

    test_all_images = glob.glob(os.path.join(dataset_root, class_name, "test", "*/*.*"))
    test_good_images = glob.glob(os.path.join(dataset_root, class_name, "test", "good/*.*"))
    test_anomaly_images = [i for i in test_all_images if i not in test_good_images]

    show_images = random.sample(test_anomaly_images, num_images)
    return [Image.open(image_path) for image_path in show_images]

css="""
    #anomaly_score span {
        font-size: 25px;
    }
    #anomaly_score textarea {
        font-size: 25px;
    }
    h1 {
        font-size: 50px;
    }
    footer {
        display:none !important
    }
"""

with gr.Blocks(css=css) as demo:
    text_bar = gr.Textbox(
        label="Anomaly Score:", 
        value="- / 100",
        elem_id="anomaly_score",
    )
    
    model_switcher = gr.Dropdown(
        ["CSAD", "InstAD", "ShiZhi"], label="Model", info="Will not add more models later!"
    )
    dataset_switcher = gr.Dropdown(
        ["MVTec LOCO", "VisA", "MVTec AD"], label="Dataset", info="Will not add more datasets later!"
    )
    classes_switcher = gr.Dropdown(
        [], label="Dataset", info="Will not add more datasets later!"
    )

    class_test_gallery = gr.Gallery(
        label="Select image", show_label=False, elem_id="gallery"
    , columns=[3], rows=[2], object_fit="contain", height="auto")

    model_switcher.change(fn=change_dataset, inputs=model_switcher, outputs=dataset_switcher)
    dataset_switcher.change(fn=change_classes, inputs=dataset_switcher, outputs=classes_switcher)
    classes_switcher.change(fn=show_class_images, inputs=[dataset_switcher,classes_switcher], outputs=class_test_gallery)

    interface = gr.Interface(
        fn=greet,
        inputs=[model_switcher, dataset_switcher, classes_switcher, class_test_gallery],  # Place text bar above the image input
        outputs=[text_bar, "image"],
        title="CVLAB Anomaly Detection Online Demo",
    )

    demo.launch()
