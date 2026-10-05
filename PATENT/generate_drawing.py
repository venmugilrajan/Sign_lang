import docx
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_drawings():
    doc = docx.Document()

    # Rule 9 Margins
    for s in doc.sections:
        s.top_margin = Cm(4.0)
        s.bottom_margin = Cm(3.0)
        s.left_margin = Cm(4.0)
        s.right_margin = Cm(3.0)

    # --- SHEET 1 OF 2 ---
    p_hdr1 = doc.add_paragraph()
    p_hdr1.paragraph_format.line_spacing = 1.0
    p_hdr1.paragraph_format.space_after = Pt(4)
    r1 = p_hdr1.add_run('Applicants: Bannari Amman Institute of Technology                             Total No. of sheets: 02\n'
                        '                                                                                                     Sheet No.: 01 of 02')
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(10)
    r1.font.bold = True

    # Image 1 (Figure 1)
    p_img1 = doc.add_paragraph()
    p_img1.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img1.paragraph_format.space_before = Pt(4)
    p_img1.paragraph_format.space_after = Pt(4)
    run_img1 = p_img1.add_run()
    run_img1.add_picture('r:/Projects/Sign Language/PATENT/fig1_flowchart.png', width=Cm(12.5))

    # Agent Signature Sheet 1
    p_sig1 = doc.add_paragraph()
    p_sig1.paragraph_format.line_spacing = 1.0
    p_sig1.paragraph_format.space_before = Pt(4)
    r_sig1 = p_sig1.add_run('Vinoth Kumar J\nPatent Agent (IN/PA 6111)\nAgent for the Applicant')
    r_sig1.font.name = 'Times New Roman'
    r_sig1.font.size = Pt(10)
    r_sig1.font.bold = True

    # Page Break
    doc.add_page_break()

    # --- SHEET 2 OF 2 ---
    p_hdr2 = doc.add_paragraph()
    p_hdr2.paragraph_format.line_spacing = 1.0
    p_hdr2.paragraph_format.space_after = Pt(4)
    r2 = p_hdr2.add_run('Applicants: Bannari Amman Institute of Technology                              Total No. of sheets: 02\n'
                        '                                                                                                                     Sheet No.: 02 of 02')
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10)
    r2.font.bold = True

    # Image 2 (Figure 2)
    p_img2 = doc.add_paragraph()
    p_img2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img2.paragraph_format.space_before = Pt(4)
    p_img2.paragraph_format.space_after = Pt(4)
    run_img2 = p_img2.add_run()
    run_img2.add_picture('r:/Projects/Sign Language/PATENT/fig2_architecture.png', width=Cm(12.5))

    # Agent Signature Sheet 2
    p_sig2 = doc.add_paragraph()
    p_sig2.paragraph_format.line_spacing = 1.0
    p_sig2.paragraph_format.space_before = Pt(4)
    r_sig2 = p_sig2.add_run('Vinoth Kumar J\nPatent Agent (IN/PA 6111)\nAgent for the Applicant')
    r_sig2.font.name = 'Times New Roman'
    r_sig2.font.size = Pt(10)
    r_sig2.font.bold = True

    target_path = 'r:/Projects/Sign Language/PATENT/Drawing_our_project.docx'
    doc.save(target_path)
    print('Drawing docx saved')

if __name__ == '__main__':
    create_drawings()
