"""
Generate Form 1 for our project:
Convert template PDF -> DOCX via Word COM, apply XML text replacements,
apply faculty/student slot fixes, and convert final DOCX -> PDF.
"""
import os, sys, time, re
sys.stdout.reconfigure(encoding='utf-8')

SRC_PDF = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1 pdf.pdf')
TEMP_DOCX = os.path.abspath('r:/Projects/Sign Language/PATENT/_temp_form1_raw.docx')
DST_DOCX = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1_our_project.docx')
DST_PDF = os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1_our_project.pdf')

# Clean existing
for f in [TEMP_DOCX, DST_DOCX, DST_PDF]:
    if os.path.exists(f):
        try:
            os.remove(f)
        except Exception:
            pass

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

NEW_TITLE = 'AN EMBEDDED EDGE-AI KIOSK APPARATUS AND WIRELESS SENSOR GLOVE SYSTEM FOR REAL-TIME BIDIRECTIONAL SIGN LANGUAGE TRANSLATION USING MULTI-MODAL SENSOR FUSION AND NEURAL INFERENCE'

REPLACEMENTS = [
    # Title replacements
    ('ORAL DENTAL COMPOSITION OF WHEATGRASS (TRITICUM AESTIVUM) PHYTOEXTRACT FOR DENTAL PAIN MANAGEMENT', NEW_TITLE),
    ('A REAL-TIME VISION-BASED DUAL-HAND SIGN LANGUAGE RECOGNITION SYSTEM USING GEOMETRIC LANDMARKS AND MULTI-LAYER PERCEPTRON', NEW_TITLE),
    
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
    ('15th day of July, 2026', '11th day of October, 2026'),
    ('2nd day of October, 2026', '11th day of October, 2026'),
    
    # Counts
    ('No. of pages ( 20 )', 'No. of pages ( 12 )'),
    ('No. of pages ( 8 )', 'No. of pages ( 12 )'),
    ('20+2+1+7 = 30', '12+2+1+4 = 19'),
    ('8+2+1+2 = 13', '12+2+1+4 = 19'),
    ('No. of drawings ( 10 )', 'No. of drawings ( 4 )'),
    ('No. of drawings ( 2 )', 'No. of drawings ( 4 )'),
    ('pages ( 7 )', 'pages ( 4 )'),
    ('pages ( 2 )', 'pages ( 4 )'),
]

def replace_in_xml(xml_bytes, replacements):
    """Replace text in all <w:t> elements across the entire XML."""
    tree = etree.fromstring(xml_bytes)
    ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
    count = 0

    # First pass: simple per-run replacement
    for t_elem in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t'):
        if t_elem.text:
            for old, new in replacements:
                if old in t_elem.text:
                    t_elem.text = t_elem.text.replace(old, new)
                    count += 1
    
    # Second pass: handle text split across runs within a paragraph
    for p_elem in tree.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
        runs = p_elem.findall('.//w:r', ns)
        if not runs:
            continue
        
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
            t_elems[0].text = new_text
            t_elems[0].set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
            for t in t_elems[1:]:
                t.text = ''
            count += 1
            
    # Fix KAYALVIZHI slot and student roles directly in the tree
    ns_w = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    all_t = list(tree.iter(ns_w + 't'))
    for i, t in enumerate(all_t):
        txt = t.text or ''
        if 'Faculty' in txt and 'Dept' in txt:
            for j in range(i-1, max(0, i-10), -1):
                prev = (all_t[j].text or '').strip()
                if prev == '':
                    all_t[j].text = 'KAYALVIZHI'
                    count += 1
                    break
            break

    for i, t in enumerate(all_t):
        txt = t.text or ''
        if 'VENMUGIL RAJAN S' in txt:
            for j in range(i+1, min(len(all_t), i+10)):
                jtxt = all_t[j].text or ''
                if 'Faculty' in jtxt:
                    all_t[j].text = jtxt.replace('Faculty', 'Student')
                    count += 1
                    break
            break
    
    return etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True), count

# Process the DOCX
total = 0
with zipfile.ZipFile(TEMP_DOCX, 'r') as zin:
    with zipfile.ZipFile(DST_DOCX, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith('.xml') or item.filename.endswith('.rels'):
                try:
                    if b'<w:t' in data or b'w:t>' in data:
                        new_data, count = replace_in_xml(data, REPLACEMENTS)
                        if count > 0:
                            total += count
                            data = new_data
                except Exception:
                    pass
            zout.writestr(item, data)

print(f"Step 2 done: {total} XML-level replacements")

# Clean temp
if os.path.exists(TEMP_DOCX):
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

print("\nForm 1 generation finished successfully!")
