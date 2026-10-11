import os
import zipfile
import shutil
import re
from PIL import Image

def build_representation_sheet():
    template_docx = r'Standard Templates\2. Design registration\Representation Sheet.docx'
    out_dir = r'DESIGN_REGISTRATION'
    os.makedirs(out_dir, exist_ok=True)
    out_docx = os.path.join(out_dir, 'Representation_Sheet_our_project.docx')
    out_pdf = os.path.join(out_dir, 'Representation_Sheet_our_project.pdf')
    
    # 7 generated design views
    view_files = [
        os.path.join(out_dir, 'fig1_front.png'),
        os.path.join(out_dir, 'fig2_back.png'),
        os.path.join(out_dir, 'fig3_left.png'),
        os.path.join(out_dir, 'fig4_right.png'),
        os.path.join(out_dir, 'fig5_top.png'),
        os.path.join(out_dir, 'fig6_bottom.png'),
        os.path.join(out_dir, 'fig7_perspective.png'),
    ]
    
    for vf in view_files:
        if not os.path.exists(vf):
            raise FileNotFoundError(f"Missing view image: {vf}")
            
    # Target Title and Applicants
    old_title = 'MANUAL AGRICULTURAL MATERIAL COLLECTOR WITH INCLINED BELT CONVEYOR'
    new_title = 'EDGE-AI SIGN LANGUAGE TRANSLATION APPARATUS'
    
    old_applicants = 'NISHANTH SHANMUGA SUNDARAM, SARVESHWARAN VALARMATHI MATHU, VELMURUGAN M LOGANATHAN, GOKUL VELLAISAMY, VASUDEVAN MANGOTTIRI, NATARAJAN NARAYANAN'
    new_applicants = 'KAYALVIZHI, VENMUGIL RAJAN S, BUVANESH C, HARISH P'
    
    old_date = 'Dated this 12th Day of March, 2026'
    new_date = 'Dated this 11th Day of October, 2026'
    
    # Unpack template to temp folder
    temp_extract = os.path.join(out_dir, 'temp_docx_extract')
    if os.path.exists(temp_extract):
        shutil.rmtree(temp_extract)
    os.makedirs(temp_extract, exist_ok=True)
    
    with zipfile.ZipFile(template_docx, 'r') as zip_ref:
        zip_ref.extractall(temp_extract)
        
    # Replace document.xml text
    doc_xml_path = os.path.join(temp_extract, 'word', 'document.xml')
    with open(doc_xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    content = content.replace(old_title, new_title)
    content = content.replace(old_applicants, new_applicants)
    content = content.replace(old_date, new_date)
    
    with open(doc_xml_path, 'w', encoding='utf-8') as f:
        f.write(content)
        
    # Media replacements
    media_map = [
        ('image1.png', view_files[0]), # Front
        ('image3.png', view_files[1]), # Back
        ('image4.png', view_files[2]), # Left
        ('image5.png', view_files[3]), # Right
        ('image6.png', view_files[4]), # Top
        ('image7.png', view_files[5]), # Bottom
        ('image8.png', view_files[6]), # Perspective
    ]
    
    for media_name, src_file in media_map:
        target_path = os.path.join(temp_extract, 'word', 'media', media_name)
        if os.path.exists(target_path):
            shutil.copyfile(src_file, target_path)
            print(f"Replaced {media_name} with {os.path.basename(src_file)}")
            
    # Repack into new docx
    if os.path.exists(out_docx):
        os.remove(out_docx)
    with zipfile.ZipFile(out_docx, 'w', zipfile.ZIP_DEFLATED) as zip_out:
        for root, dirs, files in os.walk(temp_extract):
            for file in files:
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, temp_extract)
                zip_out.write(full_path, rel_path)
                
    shutil.rmtree(temp_extract)
    print(f"Successfully created: {out_docx}")
    
    # Export to PDF via Word COM
    try:
        import win32com.client
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        doc = word.Documents.Open(os.path.abspath(out_docx))
        doc.SaveAs(os.path.abspath(out_pdf), FileFormat=17) # 17 = wdFormatPDF
        doc.Close()
        word.Quit()
        print(f"Successfully converted to PDF: {out_pdf}")
    except Exception as e:
        print(f"Word COM PDF conversion notice: {e}")

if __name__ == '__main__':
    build_representation_sheet()
