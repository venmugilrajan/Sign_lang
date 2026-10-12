import os
import docx
from docx.shared import Pt, Cm, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT

def create_drawings():
    doc = docx.Document()

    # Standard A4 Margins matching Indian Patent Office specifications
    # Page width = 21.0 cm, height = 29.7 cm
    # Printable area: 21.0 - 2.0 - 2.0 = 17.0 cm width
    for s in doc.sections:
        s.top_margin = Cm(1.2)
        s.bottom_margin = Cm(1.2)
        s.left_margin = Cm(2.0)
        s.right_margin = Cm(2.0)
        s.page_width = Cm(21.0)
        s.page_height = Cm(29.7)

    figures = [
        ('r:/Projects/Sign Language/PATENT/fig1_flowchart.png', '01'),
        ('r:/Projects/Sign Language/PATENT/fig2_architecture.png', '02'),
        ('r:/Projects/Sign Language/PATENT/fig3_physical_system.png', '03'),
        ('r:/Projects/Sign Language/PATENT/fig4_kinematics.png', '04'),
    ]

    total_sheets = f"{len(figures):02d}"
    sig_img_path = r'r:/Projects/Sign Language/PATENT/agent_signature.png'

    for idx, (fig_path, sheet_no) in enumerate(figures):
        if idx > 0:
            doc.add_page_break()

        # ─── HEADER: Line 1 (Applicant on left, Total sheets on right) ───
        p_hdr1 = doc.add_paragraph()
        p_hdr1.paragraph_format.tab_stops.add_tab_stop(Cm(17.0), WD_TAB_ALIGNMENT.RIGHT)
        p_hdr1.paragraph_format.space_before = Pt(0)
        p_hdr1.paragraph_format.space_after = Pt(1)
        p_hdr1.paragraph_format.line_spacing = 1.0

        r_app = p_hdr1.add_run('Applicants: Bannari Amman Institute of Technology')
        r_app.font.name = 'Times New Roman'
        r_app.font.size = Pt(11)

        p_hdr1.add_run('\t')

        r_tot = p_hdr1.add_run(f'Total No. of sheets: {total_sheets}')
        r_tot.font.name = 'Times New Roman'
        r_tot.font.size = Pt(11)

        # ─── HEADER: Line 2 (Sheet No. aligned right under Total sheets) ───
        p_hdr2 = doc.add_paragraph()
        p_hdr2.paragraph_format.tab_stops.add_tab_stop(Cm(17.0), WD_TAB_ALIGNMENT.RIGHT)
        p_hdr2.paragraph_format.space_before = Pt(0)
        p_hdr2.paragraph_format.space_after = Pt(8)
        p_hdr2.paragraph_format.line_spacing = 1.0

        p_hdr2.add_run('\t')
        r_sht = p_hdr2.add_run(f'Sheet No.: {sheet_no} of {total_sheets}')
        r_sht.font.name = 'Times New Roman'
        r_sht.font.size = Pt(11)

        # ─── DRAWING IMAGE (Centered, perfectly scaled to avoid overflow) ───
        p_img = doc.add_paragraph()
        p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_img.paragraph_format.space_before = Pt(0)
        p_img.paragraph_format.space_after = Pt(4)
        run_img = p_img.add_run()
        run_img.add_picture(fig_path, width=Cm(13.6))

        # ─── AGENT SIGNATURE BLOCK (Signature image + details) ───
        # Signature image if available
        if os.path.exists(sig_img_path):
            p_sig_img = doc.add_paragraph()
            p_sig_img.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p_sig_img.paragraph_format.space_before = Pt(0)
            p_sig_img.paragraph_format.space_after = Pt(2)
            p_sig_img.paragraph_format.line_spacing = 1.0
            r_sig_img = p_sig_img.add_run()
            r_sig_img.add_picture(sig_img_path, width=Cm(3.7))

        # Agent designation lines
        sig_lines = [
            'Vinoth Kumar J',
            'Patent Agent (IN/PA 6111)',
            'Agent for the Applicant'
        ]
        for s_idx, s_text in enumerate(sig_lines):
            p_s = doc.add_paragraph()
            p_s.alignment = WD_ALIGN_PARAGRAPH.RIGHT
            p_s.paragraph_format.space_before = Pt(0)
            p_s.paragraph_format.space_after = Pt(1 if s_idx < len(sig_lines) - 1 else 0)
            p_s.paragraph_format.line_spacing = 1.0
            r_s = p_s.add_run(s_text)
            r_s.font.name = 'Times New Roman'
            r_s.font.size = Pt(11)

    target_docx = r'r:\Projects\Sign Language\PATENT\Drawing_our_project.docx'
    target_pdf = r'r:\Projects\Sign Language\PATENT\Drawing_our_project.pdf'
    
    doc.save(target_docx)
    print(f'Drawing docx saved: {target_docx}')

    # Word COM PDF Conversion
    try:
        import win32com.client
        word = win32com.client.DispatchEx("Word.Application")
        word.Visible = False
        wdoc = word.Documents.Open(os.path.abspath(target_docx))
        wdoc.SaveAs(os.path.abspath(target_pdf), FileFormat=17) # 17 = wdFormatPDF
        wdoc.Close()
        word.Quit()
        print(f"Successfully converted Drawing to PDF: {target_pdf}")
    except Exception as e:
        print(f"Word COM PDF conversion notice: {e}")

if __name__ == '__main__':
    create_drawings()
