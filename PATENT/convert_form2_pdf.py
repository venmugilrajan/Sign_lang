import os
import win32com.client

def docx_to_pdf(docx_path, pdf_path):
    docx_path = os.path.abspath(docx_path)
    pdf_path = os.path.abspath(pdf_path)
    word = win32com.client.DispatchEx("Word.Application")
    word.Visible = False
    word.DisplayAlerts = False
    try:
        doc = word.Documents.Open(docx_path)
        # 17 represents wdExportFormatPDF
        doc.ExportAsFixedFormat(pdf_path, 17)
        doc.Close(False)
        print(f"Exported PDF to: {pdf_path}")
    finally:
        word.Quit()

if __name__ == '__main__':
    docx_path = r"r:\Projects\Sign Language\PATENT\Form 2_our_project.docx"
    pdf_path = r"r:\Projects\Sign Language\PATENT\Form 2_our_project.pdf"
    docx_to_pdf(docx_path, pdf_path)
