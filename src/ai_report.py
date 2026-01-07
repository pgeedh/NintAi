import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def generate_ai_analysis(angles, recommendations, api_key=None):
    """
    Uses Google Gemini (2.0 Flash Exp) to generate a detailed bio-mechanical analysis.
    User requested "Gemini 3 Flash", closest available is 2.0 Flash Exp.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        return "AI Analysis Unavailable: No Google API Key provided."

    try:
        genai.configure(api_key=key)
        
        # User requested "Gemini 3 Flash". Switching to `gemini-2.0-flash-exp`.
        model = genai.GenerativeModel('gemini-2.0-flash-exp')

        prompt = f"""
        You are an expert professional bike fitter and biomechanist (like those at MyVeloFit or Retul).
        
        Please analyze the following data for a cyclist:
        
        **Observed Angles:**
        - Knee Extension (Max / BDC): {angles.get('knee_ext_max', 'N/A')} degrees (Optimal: 140-150 deg)
        - Knee Flexion (Min / TDC): {angles.get('knee_flex_min', 'N/A')} degrees
        - Hip Flexion (Closed / TDC): {angles.get('hip_closed_min', 'N/A')} degrees (Optimal: 45-55 deg)
        - Back Angle (Avg): {angles.get('back_avg', 'N/A')} degrees (Optimal: 40-50 deg for road)
        - Shoulder/Arm Angle (Avg): {angles.get('arm_avg', 'N/A')} degrees (Optimal: 80-100 deg)
        - Neck/Back Arch: {angles.get('neck_avg', 'N/A')} degrees
        - Wrist Tilt: {angles.get('wrist_tilt_avg', 'N/A')} degrees

        **Context:**
        - The user is a triathlete/cyclist seeking "Professional Deep Dive" feedback.
        
        **Your Task:**
        1. Evaluate the rider's position based on these angles.
        2. Explain *why* the recommended adjustments (based on deviations from optimal) are necessary in terms of power, aerodynamics, and comfort.
        3. Provide tips for common issues (knee pain, back pain, numbness).
        4. Keep the tone professional, encouraging, and succinct (max 300 words).
        
        **IMPORTANT:**
        End your response with this exact line:
        "Created with love for the passion of Triathlon by Pruthvi"
        """
        
        response = model.generate_content(prompt)
        text = response.text
        
        # Ensure the watermark is there if the AI missed it (safety)
        if "Created with love for the passion of Triathlon by Pruthvi" not in text:
             text += "\n\nCreated with love for the passion of Triathlon by Pruthvi"
             
        return text
    except Exception as e:
        return f"AI Analysis Failed: {str(e)}"
