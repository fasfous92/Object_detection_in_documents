import os
import fitz  # PyMuPDF

def extract_pdf_pages_to_jpg(pdf_path):
    # Check if the file exists
    if not os.path.exists(pdf_path):
        print(f"Error: The file '{pdf_path}' was not found.")
        return

    # 1. Create a folder named after the PDF (without the .pdf extension)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_folder = f"pdf_to_image/{base_name}_pages"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created folder: {output_folder}")

    # 2. Open the PDF
    pdf_document = fitz.open(pdf_path)
    total_pages = len(pdf_document)
    print(f"Found {total_pages} pages. Starting extraction...")

    # 3. Iterate through pages and convert to JPG
    for page_number in range(total_pages):
        page = pdf_document.load_page(page_number)
        
        # Optional: Increase resolution (2.0 means 200% zoom)
        # Higher numbers mean better quality but larger file sizes
        zoom = 2.0  
        matrix = fitz.Matrix(zoom, zoom)
        
        # Render the page to a pixmap (image)
        pix = page.get_pixmap(matrix=matrix)
        
        # Create the filename (e.g., page_01.jpg, page_02.jpg)
        # Using zfill to pad numbers with zeros so they sort correctly in folders
        file_name = f"page_{str(page_number + 1).zfill(3)}.jpg"
        output_file = os.path.join(output_folder, file_name)
        
        # Save the image
        pix.save(output_file)
        print(f"Saved: {output_file}")

    pdf_document.close()
    print("\nExtraction complete! All pages are in the newly created folder.")

# --- Run the script ---
# Replace this with the name or path of your actual PDF file
pdf_target = "pdf_to_test/Annual report sample.pdf" 
extract_pdf_pages_to_jpg(pdf_target)
