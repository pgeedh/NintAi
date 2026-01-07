from fpdf import FPDF
import os
from datetime import datetime

class BikeFitPDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'NintAi - Professional Analysis', 0, 1, 'C')
        self.set_font('Arial', 'I', 10)
        self.cell(0, 5, 'Nintai (Japanese for Perseverance)', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 5, 'Created with love for the passion of Triathlon by Pruthvi', 0, 1, 'C')
        self.cell(0, 5, f'Page {self.page_no()}', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.set_fill_color(200, 220, 255)
        self.cell(0, 10, title, 0, 1, 'L', True)
        self.ln(4)

def generate_quad_report(img_tdc, img_bdc, img_front, img_over, clinical_data, output_path, ai_text=None):
    """
    Generates a 4-Phase Quad-View Report (2x2 Grid).
    """
    pdf = BikeFitPDF()
    pdf.add_page()
    
    # 1. Header
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    pdf.ln(2)

    # 2. Quad View (Visual Assessment)
    pdf.chapter_title("Visual Assessment: Quad-Phase Analysis")
    
    # Grid Layout
    # A4 width ~190mm clean. 
    # Image width = 90mm. Gap = 5mm.
    # Row 1 Y
    y1 = pdf.get_y()
    
    # Top Left (TDC)
    if img_tdc and os.path.exists(img_tdc):
        pdf.image(img_tdc, x=10, y=y1, w=90)
        # pdf.text(10, y1-1, "Top of Stroke") # Integrated in image now
        
    # Top Right (Front)
    if img_front and os.path.exists(img_front):
        pdf.image(img_front, x=105, y=y1, w=90)
        
    # Row 2 Y (Approx height of image 90mm * 9/16 = 50mm + buffer)
    y2 = y1 + 55 
    
    # Bottom Left (BDC)
    if img_bdc and os.path.exists(img_bdc):
        pdf.image(img_bdc, x=10, y=y2, w=90)
        
    # Bottom Right (Overall)
    if img_over and os.path.exists(img_over):
        pdf.image(img_over, x=105, y=y2, w=90)
        
    pdf.ln(110) # Move past grid

    # 3. Clinical Table
    pdf.chapter_title("Dynamic Kinematics Table")
    
    pdf.set_font("Arial", 'B', 10)
    pdf.set_fill_color(240, 240, 240)
    
    col_w = [60, 30, 50, 40]
    pdf.cell(col_w[0], 8, "Metric", 1, 0, 'C', True)
    pdf.cell(col_w[1], 8, "Value", 1, 0, 'C', True)
    pdf.cell(col_w[2], 8, "Reference Range", 1, 0, 'C', True)
    pdf.cell(col_w[3], 8, "Status", 1, 1, 'C', True)
    
    stats = clinical_data['stats']
    
    def add_row(metric, value, target_min, target_max, unit="deg"):
        pdf.set_font("Arial", '', 10)
        pdf.cell(col_w[0], 8, metric, 1)
        pdf.cell(col_w[1], 8, f"{value:.1f} {unit}", 1, 0, 'C')
        pdf.cell(col_w[2], 8, f"{target_min}-{target_max} {unit}", 1, 0, 'C')
        
        status = "Optimal"
        pdf.set_text_color(0, 150, 0)
        if value < target_min:
            status = "Low / Closed"
            pdf.set_text_color(200, 0, 0)
        elif value > target_max:
            status = "High / Open"
            pdf.set_text_color(200, 0, 0)
        
        pdf.cell(col_w[3], 8, status, 1, 1, 'C')
        pdf.set_text_color(0, 0, 0)

    add_row("Knee Extension (BDC)", stats.get('knee_ext_max',0), 140, 150)
    add_row("Knee Flexion (TDC)", stats.get('knee_flex_min',0), 70, 110)
    add_row("Hip Flexion (Closed)", stats.get('hip_closed_min',0), 45, 55)
    add_row("Torso Angle (Avg)", stats.get('back_avg',0), 40, 50)
    add_row("Shoulder Angle", stats.get('arm_avg',0), 80, 100)
    
    if 'neck_avg' in stats and stats['neck_avg'] > 0:
        add_row("Neck/Back Arch", stats['neck_avg'], 130, 160)
    if 'wrist_tilt_avg' in stats and stats['wrist_tilt_avg'] > 0:
        add_row("Wrist Tilt", stats['wrist_tilt_avg'], 0, 20)
    
    pdf.ln(5)

    # 4. AI Analysis
    if ai_text:
        pdf.add_page()
        pdf.chapter_title("AI Coach Verdict")
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 6, "Analysis powered by NintAi.")
        pdf.ln(2)
        try:
             clean_text = ai_text.encode('latin-1', 'replace').decode('latin-1')
             pdf.multi_cell(0, 6, clean_text)
        except: pass

    pdf.output(output_path)
    print(f"Report saved: {output_path}")

# Stubs
def generate_report(*args, **kwargs): pass
def generate_clinical_report(*args, **kwargs): pass
def generate_clinical_report_3phase(*args, **kwargs): pass
