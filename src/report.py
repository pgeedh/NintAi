from fpdf import FPDF
import os
from datetime import datetime

class BikeFitPDF(FPDF):
    def header(self):
        # Logo could go here
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NintAi Professional Bike Fit Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_report(image_path, angles_dict, recommendations, output_path="output/report.pdf", ai_text=None):
    """
    Generates a PDF report for the bike fit analysis.
    
    Args:
        image_path (str): Path to the annotated image to include.
        angles_dict (dict): Dictionary of calculated angles.
        recommendations (list): List of recommendation strings.
        output_path (str): Destination for the PDF file.
        ai_text (str, optional): Detailed analysis from Gemini AI.
    """
    pdf = BikeFitPDF()
    pdf.add_page()
    
    # Title / Metadata
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(10)

    # 1. Annotated Image
    if os.path.exists(image_path):
        # Scale image to fit width (A4 width is 210mm, margins ~10mm each side -> 190mm)
        pdf.image(image_path, x=10, y=None, w=190)
    else:
        pdf.cell(200, 10, txt="[Image not found]", ln=True)
    
    pdf.ln(10)

    # 2. Bio-Mechanical Analysis
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Bio-Mechanical Analysis", ln=True)
    pdf.set_font("Arial", size=12)
    
    # Table Header
    pdf.set_fill_color(240, 240, 240)
    pdf.cell(60, 10, "Metric", 1, 0, 'B', True)
    pdf.cell(60, 10, "Observed Angle", 1, 0, 'B', True)
    pdf.cell(70, 10, "Target Range", 1, 1, 'B', True)
    
    # Data Rows
    def add_row(metric, value, target):
        pdf.cell(60, 10, metric, 1)
        pdf.cell(60, 10, f"{int(value)} degrees", 1)
        pdf.cell(70, 10, target, 1, 1)

    add_row("Knee Extension", angles_dict.get('knee', 0), "25 - 35 (Flexion)")
    add_row("Hip Flexion", angles_dict.get('hip', 0), "45 - 55")
    add_row("Elbow Flexion", angles_dict.get('elbow', 0), "90 - 120")
    
    pdf.ln(10)

    # 3. Recommendations
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recommendations", ln=True)
    pdf.set_font("Arial", size=12)
    
    if recommendations:
        for rec in recommendations:
            pdf.multi_cell(0, 10, f"- {rec}")
    else:
        pdf.cell(0, 10, "Your fit looks optimal! No major adjustments detected.")

    # 4. AI Analysis (Gemini)
    if ai_text:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "AI Professional Insight (Gemini)", ln=True)
        pdf.set_font("Arial", size=11)
        # Handle unicode encoding issues in basic fpdf
        try:
            pdf.multi_cell(0, 6, ai_text.encode('latin-1', 'replace').decode('latin-1'))
        except:
            pdf.multi_cell(0, 6, "AI Analysis text could not be rendered due to font encoding issues.")

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pdf.output(output_path)
    print(f"PDF Report generated: {output_path}")
