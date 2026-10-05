"""
Fix KAYALVIZHI name in first inventor slot and swap Faculty/Student roles.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
from lxml import etree
import zipfile, shutil, io

SRC = 'r:/Projects/Sign Language/PATENT/Form 1_our_project.docx'
DST = 'r:/Projects/Sign Language/PATENT/Form 1_our_project_fixed.docx'

with zipfile.ZipFile(SRC, 'r') as zin:
    with zipfile.ZipFile(DST, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            
            if item.filename == 'word/document.xml':
                tree = etree.fromstring(data)
                ns_w = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
                
                # Collect all <w:t> elements
                t_elems = list(tree.iter(ns_w + 't'))
                
                # Find the first empty-ish text element before "Faculty, Dept. of"
                # That's where SHOBANA KRISHNAKUMAR was and is now blank
                for i, t in enumerate(t_elems):
                    text = t.text or ''
                    if 'Faculty' in text and 'Dept' in text:
                        # Go backwards to find the blank slot
                        for j in range(i-1, max(0, i-10), -1):
                            prev_text = (t_elems[j].text or '').strip()
                            if prev_text == '' or prev_text in ['', ' ']:
                                # Check if this is in a name-like position (after several blanks)
                                # The pattern is: blank, blank, blank, NAME, blank, blank, blank, Indian, ...
                                t_elems[j].text = 'KAYALVIZHI'
                                print('Inserted KAYALVIZHI at text element ' + str(j))
                                break
                        break
                
                # Also fix: The second inventor (VENMUGIL RAJAN S) is in a Faculty slot
                # but should be Student. Find "Faculty" near VENMUGIL and change to Student
                for i, t in enumerate(t_elems):
                    text = t.text or ''
                    if 'VENMUGIL RAJAN S' in text:
                        # Look backwards for "Faculty" nearby
                        for j in range(i+1, min(len(t_elems), i+10)):
                            jtext = t_elems[j].text or ''
                            if 'Faculty' in jtext:
                                t_elems[j].text = jtext.replace('Faculty', 'Student')
                                print('Changed Faculty to Student for VENMUGIL at element ' + str(j))
                                break
                        break
                
                data = etree.tostring(tree, xml_declaration=True, encoding='UTF-8', standalone=True)
            
            zout.writestr(item, data)

# Replace original
shutil.move(DST, SRC)
print('Fixed DOCX saved')

# Reconvert to PDF
import win32com.client, time
time.sleep(0.5)
word = win32com.client.Dispatch('Word.Application')
word.Visible = False
word.DisplayAlerts = False
doc = word.Documents.Open(os.path.abspath(SRC))
doc.SaveAs2(os.path.abspath('r:/Projects/Sign Language/PATENT/Form 1_our_project.pdf'), FileFormat=17)
doc.Close()
word.Quit()
print('PDF re-exported')

# Verify
import pypdf
reader = pypdf.PdfReader('r:/Projects/Sign Language/PATENT/Form 1_our_project.pdf')
all_text = ' '.join(p.extract_text() or '' for p in reader.pages)
for name in ['KAYALVIZHI', 'VENMUGIL RAJAN', 'BUVANESH', 'HARISH', 'SIGN LANGUAGE', 'Computer Science']:
    status = 'FOUND' if name in all_text else 'NOT FOUND'
    print(name + ': ' + status)
print('DONE!')
