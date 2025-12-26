import fitz  # PyMuPDF
import io
from PIL import Image

# Test avec un PDF existant
pdf_path = r"F:\RECHERCHE\THESE\project\uploads\Memoire_Fanny.pdf"

print(f"Testing PyMuPDF extraction on: {pdf_path}")

images = []
try:
    doc = fitz.open(pdf_path)
    print(f"PDF has {len(doc)} pages")
    
    for page_index in range(len(doc)):
        page = doc[page_index]
        image_list = page.get_images(full=True)
        print(f"Page {page_index + 1}: {len(image_list)} images found")
        
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes))
            images.append(image)
            print(f"  - Image {img_index + 1}: {image.size} pixels, mode: {image.mode}")
            
    print(f"\nTotal images extracted: {len(images)}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
