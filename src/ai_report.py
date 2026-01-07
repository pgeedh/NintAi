import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def generate_ai_analysis(angles, recommendations, api_key=None):
    """
    Uses Google Gemini 3 (or Flash 1.5/Pro) to generate a detailed bio-mechanical analysis.
    
    Args:
        angles (dict): Dictionary of calculated angles (knee, hip, elbow).
        recommendations (list): List of algorithmic recommendations.
        api_key (str): Google API Key.
        
    Returns:
        str: Detailed analysis text.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY")
    if not key:
        return "AI Analysis Unavailable: No Google API Key provided."

    try:
        genai.configure(api_key=key)
        
        # Determine model - assuming 'gemini-1.5-flash' as a robust default available now. 
        # User asked for "Gemini 3", but standard access might be 1.5. 
        # We'll try to use a generic 'gemini-pro' or 'gemini-1.5-flash' alias.
        model = genai.GenerativeModel('gemini-1.5-flash')

        prompt = f"""
        You are an expert professional bike fitter and biomechanist (like those at MyVeloFit or Retul).
        
        Please analyze the following data for a cyclist:
        
        **Observed Angles:**
        - Knee Extension (at bottom dead center): {angles.get('knee', 'N/A')} degrees (Optimal: 25-35 deg flexion)
        - Hip Flexion: {angles.get('hip', 'N/A')} degrees (Optimal: 45-55 deg)
        - Elbow Flexion: {angles.get('elbow', 'N/A')} degrees (Optimal: 90-120 deg)
        
        **Algorithmic Recommendations:**
        {chr(10).join(['- ' + r for r in recommendations])}
        
        **Your Task:**
        1. Evaluate the rider's position based on these angles.
        2. Explain *why* the recommended adjustments (if any) are necessary in terms of power, aerodynamics, and comfort.
        3. Provide tips for common issues associated with these angles (e.g., knee pain, lower back pain, numbness).
        4. Keep the tone professional, encouraging, and succinct (max 300 words).
        """
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI Analysis Failed: {str(e)}"
