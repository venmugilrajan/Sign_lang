"""
Convert PDF -> DOCX via Word COM, then use python-docx to do XML-level text 
replacement across ALL elements (paragraphs, tables, shapes, textboxes).
"""
import os, sys, time, re
sys.stdout.reconfigure(encoding='utf-8')

SRC_PDF = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1 pdf.pdf')
TEMP_DOCX = os.path.abspath('r:/Projects/Sign Language/PATENT/_temp_form1_raw.docx')
DST_DOCX = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1_our_project.docx')
DST_PDF = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1_our_project.pdf')

# Clean
for f in [TEMP_DOCX, DST_DOCX, DST_PDF]:
    if os.path.exists(f):
        os.remove(f)

# ========== STEP 1: PDF -> raw DOCX via Word COM ==========
import win32com.client
word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = False
doc = word.Documents.Open(SRC_PDF)
doc.SaveAs2(TEMP_DOCX, FileFormat=16)
doc.Close()
word.Quit()
print("Step 1 done: PDF -> raw DOCX")
time.sleep(1)

# ========== STEP 2: XML-level replacement via python-docx / lxml ==========
from lxml import etree
import zipfile, shutil

REPLACEMENTS = [
    # Title
    ('ORAL DENTAL COMPOSITION OF WHEATGRASS (TRITICUM AESTIVUM) PHYTOEXTRACT FOR DENTAL PAIN MANAGEMENT',
     'A REAL-TIME VISION-BASED DUAL-HAND SIGN LANGUAGE RECOGNITION SYSTEM USING GEOMETRIC LANDMARKS AND MULTI-LAYER PERCEPTRON'),
    
    # Names
    ('SHOBANA KRISHNAKUMAR', 'KAYALVIZHI'),
    ('SHOBANA', 'KAYALVIZHI'),
    ('KRISHNAKUMAR', ''),
    ('SABAH SALEEM', 'VENMUGIL RAJAN S'),
    ('MITTHRA SURESH', 'BUVANESH C'),
    ('M MADHANA', ''),
    ('MADHUBALA', 'HARISH P'),
    ('PAVITHRA K', ''),
    ('K VIJAY', ''),
    ('VENKATESH', ''),
    
    # Department
    ('Conservative Dentistry and', 'Computer Science and'),
    ('Conservative Dentistry a nd', 'Computer Science and'),
    ('Endodontics,', 'Engineering,'),
    ('Endodontics', 'Engineering'),
    
    # Date
    ('15th day of July, 2026', '2nd day of October, 2026'),
    
    # Counts
    ('No. of pages ( 20 )', 'No. of pages ( 8 )'),
    ('20+2+1+7 = 30', '8+2+1+2 = 13'),
    ('No. of drawings ( 10 )', 'No. of drawings ( 2 )'),
    ('pages ( 7 )', 'pages ( 2 )'),
]

def replace_in_xml(xml_bytes, replacements):
    """Replace text in all <w:t> elements across the entire XML."""
    tree = etree.fromstring(xml_bytes)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    
    # Strategy: For each paragraph-like element, concatenate all <w:t> text,
    # do replacements, then redistribute back to <w:t> elements.
    # This handles text split across multiple runs.
    
    # First pass: simple per-run replacement
    count = 0
    for t_elem in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t_elem.text:
            original = t_elem.text
            for old, new in replacements:
                if old in t_elem.text:
                    t_elem.text = t_elem.text.replace(old, new)
                    count += 1
    
    # Second pass: handle text split across runs within a paragraph
    for p_elem in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        runs = p_elem.findall('.//w:r', ns)
        if not runs:
            continue
        
        # Get all text
        t_elems = []
        for r in runs:
            t = r.find('w:t', ns)
            if t is not None and t.text:
                t_elems.append(t)
        
        if not t_elems:
            continue
        
        full_text = ''.join(t.text for t in t_elems)
        new_text = full_text
        changed = False
        for old, new_val in replacements:
            if old in new_text:
                new_text = new_text.replace(old, new_val)
                changed = True
        
        if changed:
            # Put all text in the first <w:t>, clear the rest
            t_elems[0].text = new_text
            t_elems[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            for t in t_elems[1:]:
                t.text = ''
            count += 1
    
    return etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True), count

# Process the DOCX (which is a ZIP)
shutil.copy2(TEMP_DOCX, DST_DOCX)

total = 0
with zipfile.ZipFile(TEMP_DOCX, 'r') as zin:
    with zipfile.ZipFile(DST_DOCX, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            
            # Process all XML files that could contain text
            if item.filename.endswith('.xml') or item.filename.endswith('.rels'):
                try:
                    if b'<w:t' in data or b'w:t>' in data:
                        new_data, count = replace_in_xml(data, REPLACEMENTS)
                        if count > 0:
                            print(f"  Modified {item.filename} ({count} replacements)")
                            total += count
                            data = new_data
                except Exception as e:
                    pass  # Not a Word XML file, skip
            
            zout.writestr(item, data)

print(f"\nStep 2 done: {total} XML-level replacements")

# Clean temp
os.remove(TEMP_DOCX)

# ========== STEP 3: Convert final DOCX to PDF ==========
time.sleep(1)
word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = False
doc = word.Documents.Open(DST_DOCX)
doc.SaveAs2(DST_PDF, FileFormat=17)
doc.Close()
word.Quit()
print("Step 3 done: DOCX -> PDF")

# ========== VERIFY ==========
print("\n=== VERIFICATION ===")
import pypdf
reader = pypdf.PdfReader(DST_PDF)
print(f"PDF pages: {len(reader.pages)}")
all_text = ' '.join(p.extract_text() or '' for p in reader.pages)

for name in ['KAYALVIZHI', 'VENMUGIL RAJAN', 'BUVANESH', 'HARISH', 'SIGN LANGUAGE', 'Computer Science']:
    print(f"  {name}: {'FOUND OK' if name in all_text else 'NOT FOUND (BAD)'}")
for name in ['SHOBANA', 'SABAH', 'MITTHRA', 'MADHUBALA', 'ORAL DENTAL', 'Conservative Dentistry']:
    print(f"  {name}: {'STILL PRESENT (BAD)' if name in all_text else 'REMOVED OK'}")

print("\nDONE!")
