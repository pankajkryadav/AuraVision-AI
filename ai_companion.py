import requests
import json
import random

class AICompanion:
    def __init__(self, model_name="qwen3:0.6b", api_url="http://localhost:11434"):
        self.model_name = model_name
        self.api_url = api_url
        self.ollama_connected = False

    def check_connection(self):
        was_connected = self.ollama_connected
        try:
            r = requests.get(f"{self.api_url}/api/tags", timeout=1.0)
            if r.status_code == 200:
                self.ollama_connected = True
                if not was_connected:
                    print("Ollama connection verified. LLM mode active.")
                return True
        except Exception:
            pass
        self.ollama_connected = False
        if was_connected:
            print("Ollama connection unavailable. Smart Rule-Based Fallback active.")
        return False

    def generate_response(self, user_message, mood_context, sentiment_label, conversation_history=None):
        """
        Generates an AI response based on Ollama or the local rule-based engine.
        """
        # Proactively check connection status in case it was started after app launch
        self.check_connection()
        
        if self.ollama_connected:
            return self._generate_ollama_response(user_message, mood_context, sentiment_label, conversation_history)
        else:
            return self._generate_fallback_response(user_message, mood_context, sentiment_label)

    def _generate_ollama_response(self, user_message, mood_context, sentiment_label, conversation_history):
        # Build system instructions adapting the model's personality to user's emotional state
        system_instruction = (
            f"You are Aura, an empathetic AI Workspace Companion and Assistant. "
            f"The user's current facial emotion is: {mood_context}. "
            f"Their journal/input sentiment is: {sentiment_label}. "
            f"Adapt your tone accordingly: if they are Stressed or Tired, be highly supportive, encouraging, and soothing. "
            f"If they are Focused, be concise, professional, and avoid distracting them unless they ask a direct question or request code help. "
            f"If they are Happy, share their enthusiasm! "
            f"If the user asks a technical or coding question, write clean, well-commented code in standard markdown code blocks (using ```lang ... ```) and explain it clearly and thoroughly."
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        
        # Add historical context if provided
        if conversation_history:
            for chat in conversation_history[-6:]:  # Last 6 exchanges to avoid token overload
                messages.append({"role": "user" if chat["role"] == "user" else "assistant", "content": chat["message"]})
                
        messages.append({"role": "user", "content": user_message})
        
        try:
            r = requests.post(
                f"{self.api_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": messages,
                    "stream": False
                },
                timeout=30.0
            )
            if r.status_code == 200:
                return r.json().get("message", {}).get("content", "Sorry, I had trouble generating a reply.")
        except Exception as e:
            print(f"Ollama connection timed out or failed: {e}")
            
        return self._generate_fallback_response(user_message, mood_context, sentiment_label)

    def _generate_fallback_response(self, message, mood, sentiment):
        message = message.lower()
        
        # Categorized keyword lists for topic detection
        stress_triggers = ["stress", "anxious", "worry", "deadline", "exam", "grade", "fail", "hard", "tired", "sleep", "exhausted", "burnout"]
        work_triggers = ["work", "study", "code", "project", "presentation", "assignment", "report", "write", "tasks", "schedule"]
        happy_triggers = ["happy", "good", "great", "nice", "awesome", "won", "passed", "love", "excited", "glad"]
        greeting_triggers = ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "morning"]
        
        # Context-based canned sentences
        greetings = [
            "Hello there! I'm Aura. How is your work going?",
            "Hi! Hope your day is progressing smoothly. How can I help you?",
            "Hey! I'm here to help you stay focused and balanced. What's on your mind today?"
        ]
        
        # Responses matching mood states
        if any(w in message for w in greeting_triggers):
            base = random.choice(greetings)
            if mood == "Stressed" or mood == "Tired":
                return base + " I notice you're looking a bit drained. Remember to take a deep breath."
            elif mood == "Focused":
                return "Hey! I see you're locked in and focused. Let me know if you need anything specific; otherwise, keep up the great work!"
            return base

        # Topic: Stress, Tiredness
        if any(w in message for w in stress_triggers) or mood in ["Stressed", "Tired"] or sentiment == "Negative":
            responses = [
                "I hear you. Academic and project deadlines can be incredibly intense. Take a 5-minute screen break—stretch and drink some water.",
                "It's completely okay to feel overwhelmed. Focus on one small step at a time. What is the very next minor task you can tackle?",
                "Fatigue is your brain's way of asking for a brief pause. Let's do a quick breathing exercise: inhale for 4 seconds, hold for 4, exhale for 4. Better?",
                "Remember, your well-being comes first. A quick break now will actually boost your efficiency when you return to your work."
            ]
            return random.choice(responses)
            
        # Topic: Work / Productivity
        if any(w in message for w in work_triggers) or mood == "Focused":
            responses = [
                "Excellent. Let's block out distractions. I recommend setting a 25-minute Pomodoro timer. I will keep track of your focus.",
                "You're in the zone! Focus on structured intervals. Would you like me to log this productivity block in your database?",
                "Great! Breaking down your project into micro-deliverables is the key. Focus on finishing this current file, then celebrate.",
                "Workspace monitoring is running. You're doing great. Keep your posture straight!"
            ]
            return random.choice(responses)

        # Topic: Happiness / Success
        if any(w in message for w in happy_triggers) or mood == "Happy" or sentiment == "Positive":
            responses = [
                "That's fantastic! I love seeing that positive energy. Let's channel this momentum into your project!",
                "Incredible news! Celebrating small wins is highly important for cognitive health. Keep up this wonderful streak!",
                "I am so glad to hear that! Your mood trends on the dashboard are looking exceptionally bright today.",
                "Awesome! What was the highlight of your day today?"
            ]
            return random.choice(responses)

        # General Default Responses
        defaults = [
            f"I see you're currently in a **{mood}** state. It's good to be aware of our mental states as we study. What are you working on right now?",
            "As your workspace companion, I'm logging your focus intervals and mood. Feel free to write a journal entry to track your cognitive trends.",
            "That's interesting. Tell me more about your project. How are you designing the system architecture?",
            "I'm keeping an eye on your attention level. Let's try to complete this workspace session with high focus!"
        ]
        return random.choice(defaults)

    def generate_visual_advice(self, metrics):
        """
        Prompts Ollama to generate brief feedback based on real-time face metrics.
        """
        self.check_connection()
        
        emotion = metrics.get("emotion", "Focused")
        focus = metrics.get("focus_score", 100.0)
        blinks = metrics.get("blink_count", 0)
        fatigue = metrics.get("fatigue_alert", False)
        
        if not self.ollama_connected:
            # Fallback canned insights
            if fatigue:
                return "CRITICAL fatigue detected. Take a 5-minute break immediately! Close your eyes and rest."
            if emotion == "Stressed":
                return "High stress/blink rate detected. Try lowering your shoulders, taking a deep breath, and stretching."
            if emotion == "Tired":
                return "Drowsiness patterns identified. Consider stepping away for water or a quick stretch."
            if emotion == "Focused":
                return "Excellent focus! You are in the zone. Keep up the high attention block."
            return "Workspace environment looks stable. Remember to keep a healthy posture!"
            
        system_instruction = (
            "You are Aura, an empathetic AI Workspace Counselor. "
            "You analyze user's real-time workspace facial metrics and provide a single-sentence advice or observation. "
            "Keep response under 20 words, direct, encouraging, and supportive. Do not include quotes."
        )
        
        prompt = (
            f"Analyze these metrics and comment:\n"
            f"- Detected Facial State: {emotion}\n"
            f"- Focus Score: {focus:.0f}%\n"
            f"- Blink Count (last 10s): {blinks}\n"
            f"- Fatigue Alert: {'ACTIVE' if fatigue else 'NORMAL'}"
        )
        
        try:
            r = requests.post(
                f"{self.api_url}/api/chat",
                json={
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": False
                },
                timeout=5.0
            )
            if r.status_code == 200:
                content = r.json().get("message", {}).get("content", "Environment stable. Keep up the good work!").strip()
                # Clean up quotes if returned by the model
                content = content.replace('"', '').replace("'", "")
                return content
        except Exception as e:
            print(f"Error generating visual advice: {e}")
            
        return "Workspace environment looks stable. Remember to stay hydrated!"

