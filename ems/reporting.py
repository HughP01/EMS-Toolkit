import google.generativeai as genai
import pandas as pd
import os

genai.configure(api_key=os.getenv('GEMINI_API_KEY')) #Have API key saved as GEMINI_API_KEY in sys variables
