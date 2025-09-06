import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import requests
import json
import time

# Function to get a mood score from a user prompt using a large language model.
# In a real-world application, this would be integrated with a backend service.
# For demonstration, we'll use a direct API call.
def get_mood_from_prompt(prompt):
    """
    Analyzes the sentiment of a user prompt using a large language model API.
    Returns a mood score from 1 (very negative) to 5 (very positive).
    """
    api_key = "AIzaSyAQrAG8xX5JgIpLOsMJvapYsoE0zSZKwH4" # API key is automatically provided by the Canvas environment.
    model_name = "gemini-2.5-flash-preview-05-20"
    api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # The system instruction guides the model on how to respond.
    system_instruction = {
        "parts": [{ "text": "Analyze the sentiment of the following user text and provide a single integer score from 1 to 5. Do not provide any other text. 1 = extremely negative, 2 = negative, 3 = neutral, 4 = positive, 5 = extremely positive." }]
    }

    payload = {
        "contents": [{ "parts": [{ "text": prompt }] }],
        "systemInstruction": system_instruction,
    }

    retries = 3
    for i in range(retries):
        try:
            response = requests.post(api_url, data=json.dumps(payload), headers={'Content-Type': 'application/json'})
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            result = response.json()
            # Extract the score from the model's response
            text = result['candidates'][0]['content']['parts'][0]['text']
            
            # The model might return a string, so we'll try to convert it to an integer
            score = int(text.strip())
            
            # Ensure the score is within our desired range
            return max(1, min(5, score))

        except (requests.exceptions.RequestException, KeyError, ValueError) as e:
            print(f"Error fetching mood for prompt: '{prompt}'. Error: {e}")
            if i < retries - 1:
                wait_time = 2 ** (i + 1)
                print(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # If all retries fail, return a default score
                print("All retries failed. Defaulting to a score of 3.")
                return 3
    return 3 # Fallback in case the loop completes without a successful return


# A list of more varied and nuanced user prompts.
user_prompts = [
    "I'm feeling so unmotivated today, it's a real struggle.",
    "I am feeling down today ,the whole day of mine was full  of misery",
    "Just finished an amazing book, I can't wait to start the next one!",
    "I am feeling lucky today,I got 500 rs note on road",
    "Work has been stressful, but I'm managing to keep my head above water.",
    "This weather is absolutely beautiful, it made my day.",
    "My phone broke and I lost all my contacts. This is the worst.",
    "Received some surprising news today, still trying to process it all.",
    "I had a fantastic day catching up with old friends.",
    "Nothing much to report, just a quiet, average day."
]

# A more robust function to generate mood data from prompts
def generate_mood_data_from_prompts(prompts):
    mood_data_list = []
    # Generate dates starting from a recent past date
    start_date = datetime.now() - timedelta(days=len(prompts) - 1)
    
    for i, prompt in enumerate(prompts):
        date = start_date + timedelta(days=i)
        
        # Use the LLM to get the mood score for each prompt
        mood_score = get_mood_from_prompt(prompt)

        # A reverse mapping to get the word from the score
        score_to_word = {
            1: 'Terrible',
            2: 'Bad',
            3: 'Neutral',
            4: 'Good',
            5: 'Excellent'
        }
        mood_word = score_to_word.get(mood_score, 'Neutral')
        
        print(f"Prompt: '{prompt}' -> Score: {mood_score} ({mood_word})")
        mood_data_list.append((date, mood_score, mood_word))
    
    return mood_data_list

# Generate mood data from the new, more complex prompts
mood_data = generate_mood_data_from_prompts(user_prompts)

# Separate data into dates, scores, and words
dates = [item[0] for item in mood_data]
scores = [item[1] for item in mood_data]
words = [item[2] for item in mood_data]

# --- Graphical Representation (Line Plot) ---
plt.style.use('fivethirtyeight')
plt.figure(figsize=(10, 6))
plt.plot(dates, scores, marker='o', linestyle='-', color='#6495ED')

# Format the x-axis to show dates nicely
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()

# Set y-axis ticks and labels
plt.yticks([1, 2, 3, 4, 5], ['Terrible', 'Bad', 'Neutral', 'Good', 'Excellent'])
plt.ylim(0.5, 5.5)

plt.title('Daily Mood Trend', fontsize=18)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Mood', fontsize=12)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Add annotations for each data point
for date, score, word in zip(dates, scores, words):
    plt.annotate(
        word,
        (date, score),
        textcoords="offset points",
        xytext=(0, 10),
        ha='center',
        fontsize=10,
        color='black'
    )

plt.tight_layout()
plt.savefig('mood_trend.png')
print("Mood trend graph saved as 'mood_trend.png'")
plt.close()

# --- Pictorial Representation (Bar Chart) ---
mood_counts = {
    'Terrible': words.count('Terrible'),
    'Bad': words.count('Bad'),
    'Neutral': words.count('Neutral'),
    'Good': words.count('Good'),
    'Excellent': words.count('Excellent'),
}
mood_labels = list(mood_counts.keys())
counts = list(mood_counts.values())

colors = ['#FF4500', '#FF8C00', '#FFD700', '#ADFF2F', '#32CD32']

plt.figure(figsize=(8, 6))
bars = plt.bar(mood_labels, counts, color=colors, edgecolor='black', alpha=0.8)

# Add the count on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2.0,
        yval,
        int(yval),
        ha='center',
        va='bottom',
        fontsize=12,
    )

plt.title('Mood Distribution', fontsize=18)
plt.xlabel('Mood', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(axis='y', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.savefig('mood_distribution.png')
print("Mood distribution chart saved as 'mood_distribution.png'")
plt.close()

# Instructions:
# 1. You will need to install the 'requests' library to make the API call. Run: `pip install requests`
# 2. Run the script from your terminal: `python mood_tracker.py`
# 3. This will create two image files: 'mood_trend.png' and 'mood_distribution.png'.
# 4. These images are referenced in the HTML file, so you can then open 'mood_dashboard.html'
#    in your browser to view the mood tracker.
# 5. You can easily update the 'user_prompts' list with your own text data to see new results.


