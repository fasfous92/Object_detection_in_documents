import os
import json 
from prompt_toolkit import prompt 
from utils.Qwen_25_LLM import Qwen_llm
from utils.extract_pdf import extract_pdf_pages_to_jpg



BASE_MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"#  Model to use 
ADAPTER_PATH = "output/qwen2.5-vl-final-detector-all-linear" # Path to the adapter trained on our signature dataset. Set to None to use the base model without loading any adapter.
PROMPT="Detect all logos, signatures, and stamps. If none are found, respond with: None of the requested objects were found."
OUTPUT_JSON='output/output_json.json'



if __name__ == "__main__":
   
    #--1. Initialize the model
    model=Qwen_llm(base_model_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH)
    model.load() 
    
    #--2. Load the PDF, extract pages as images and store them 
    PDF_PATH = "pdf_sample.pdf"
    OUTPUT_PDF_IMAGE_DIR = "pdf_to_image"
    extract_pdf_pages_to_jpg(PDF_PATH,output_dir=OUTPUT_PDF_IMAGE_DIR)
    
    #--3. Run inference on each page image and store the results in a JSON file
    output_json=[]
    pdf_images_dir = os.path.join(OUTPUT_PDF_IMAGE_DIR, "pdf_sample_pages")
    for img_file,i in zip(os.listdir(pdf_images_dir), range(len(os.listdir(pdf_images_dir)))):
        img_path = os.path.join(pdf_images_dir, img_file)
        #-- Run prediction and parse the output into structured detections
        output_text,original_height,original_width=model.predict(img_path,PROMPT)
        output_structured=model.parse_qwen_detections(output_text, original_width, original_height)

        if len(output_structured) == 0: #if no object is detected, skip to the next page without adding an entry to the JSON (to keep it clean and focused on pages with detections)
            continue
        output_json.append({
            "page": i+1,
            "bboxes": output_structured
        })
        
        if i==2: # For testing purposes, we will only run inference on the first 3 pages to speed up the process. Remove this condition to run on all pages.
            break
        
    #save the predictions for all pages in a JSON file
    json.dump(output_json, open(OUTPUT_JSON, "w"), indent=4)
    
    
