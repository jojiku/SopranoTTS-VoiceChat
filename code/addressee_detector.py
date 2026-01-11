# addressee_detector.py
import logging
import time
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

logger = logging.getLogger(__name__)
model_dir = "Silxxor/Lucy-addressee-detector"

class AddresseeDetector: 
    def __init__(self):
        """The brain police.  Decides if you are worthy of my attention."""

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            self.classifier = pipeline(
                "text-classification", 
                model=model_dir, 
                tokenizer=AutoTokenizer.from_pretrained(model_dir),
                device=0 if self.device == "cuda" else -1,
                top_k=None
            )
            logger.info("AddresseeDetector loaded.")
        except Exception as e:
            logger.error(f"Failed to load addressee model: {e}")
            self.classifier = None

        self.last_interaction_time = 0
        
        self.WAKE_WORDS = ["lucy", "computer", "assistant"]
        self.CONVERSATION_WINDOW = 10.0
        
        self.HIGH_THRESHOLD = 0.75
        self.LOW_THRESHOLD = 0.25

    def predict(self, text):
        """Returns probability (0.0 - 1.0) that text is addressed to Lucy."""
        if not self.classifier:
            return 1.0
        
        results = self.classifier(text)[0]
        
        for res in results:
            if res['label'] == 'LABEL_1':
                return res['score']
            if res['label'] == 'LABEL_0':
                return 1.0 - res['score']
        
        return 0.0

    def should_reply(self, text, time_since_ai_spoke):
        """
        Simple logic: 
        1. Wake word = YES
        2. Within 5s of last AI response = YES (unless model is VERY sure it's not for us)
        3. Cold start = trust the model
        """
        text_lower = text.lower().strip()
        
        # === WAKE WORDS:  Always yes ===
        for word in self.WAKE_WORDS: 
            if word.lower() in text_lower:
                return True
        
        base_score = self.predict(text)
        in_conversation = time_since_ai_spoke < 10.0
        
        if in_conversation:
            # Recently talked = assume they're still talking to us
            # ONLY reject if model is 99% confident it's NOT for us (score < 0.01)
            if base_score < 0.01:
                logger.info(f"[ACTIVE] score={base_score:. 2f} < 0.01 -> NO (model 99% confident not addressed)")
                return False
            else:  
                logger. info(f"[ACTIVE] score={base_score:.2f} within 10s -> YES")
                return True
        else:  
            # Cold start = need model confidence
            threshold = 0.5
            result = base_score >= threshold
            logger. info(f"[COLD] score={base_score:.2f} >= {threshold} -> {result}")
            return result
    
    def shutdown(self):
        """Release GPU memory and cleanup resources."""
        logger.info("🧠 AddresseeDetector:  Shutting down...")
        
        if self.classifier is not None:
            # Delete the pipeline and its underlying model
            del self.classifier
            self.classifier = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("🧠 AddresseeDetector: Shutdown complete.")

    def __del__(self):
        """Destructor - attempt cleanup if shutdown wasn't called."""
        try: 
            if hasattr(self, 'classifier') and self.classifier is not None: 
                self.shutdown()
        except Exception: 
            pass


if __name__=="__main__":
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

    classifier = AddresseeDetector()

    raw_data = [
        ("damn, this coffee is total shit", 0),
        ("Lucy, what time is it?", 1),
        ("fuck, I'm so tired", 0),
        ("hey Lucy, help me", 1),
        ("this code is fucking terrible", 0),
        ("Lucy, the code is total shit, help me figure it out", 1),
        ("hi, how are things?", 0),
        ("yo Lucy", 1),
        ("hey everyone / what's up guys", 0),
        ("well you know how it goes", 0),
        ("I think you should", 0),
        ("what the fuck is even happening", 0),
        ("hey assistant, how's it going?", 1),
        ("assistant, can you help me figure this out?", 1),
        ("alright assistant", 1),
        ("computer, help me", 1),
        ("hey computer", 1),
        ("you need to look at this", 1), 
        ("we need to fix this", 0),
        ("someone help", 0),
        ("does anyone know why?", 0),
        ("can someone explain?", 0),
        ("where did I put this?", 0),
        ("when was this last?", 0),
        ("who broke this?", 0),
        ("which option is better?", 0),
        ("why isn't this shit working?", 0),
        ("you won't believe it", 0),
        ("we need more time", 0),
        ("there's some problem here", 0),
        ("something's not right here", 0),
        ("everything's fucking broken", 0),
        ("show logs", 1),
        ("check the database", 1),
        ("run the tests again", 1),
        ("stop the server", 1),
        ("restart everything", 1),
        ("why did I do it this way?", 0),
        ("what was I even thinking?", 0),
        ("how did this ever work?", 0),
        ("where was I even going with this?", 0),
        ("when did this all break?", 0),
        ("well like there's this thing with...", 0),
        ("yeah, but the problem is that", 0),
        ("okay, basically", 0),
        ("I mean, the point is that", 0),
        ("well obviously we can't", 0),
        ("you know what I mean?", 1),
        ("can't believe this shit", 0),
        ("are you guys kidding me", 0),
        ("stop this", 0),
        ("fix this mess", 1),
        ("alright, fuck it, let's move on", 0),
        ("fuck it, I'll figure it out myself", 0),
        ("okay, let's say this works", 0),
        ("well alright then", 0),
        ("cool story bro", 0),
        ("she understands whether I'm talking to her or not", 0),
        ("yeah, she's really smart", 0),
        ("and she's funny too", 0),
        ("she's just awesome", 0),
        ("yeah, she's the best so far", 0)
    ]

    test_texts = [x[0] for x in raw_data]
    test_labels = np.array([x[1] for x in raw_data])

    predictions = []
    print(f"Running inference on {len(test_texts)} cases...")

    for text in test_texts:
        pred = classifier.predict(text)
        pred = np.round(pred)
        predictions.append(pred)

    predictions = np.array(predictions)
    accuracy = accuracy_score(test_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, predictions, average='binary', zero_division=0)
    cm = confusion_matrix(test_labels, predictions)

    # 5. Print Report
    print("\n" + "="*50)
    print("ADDRESSEE DETECTOR EVALUATION")
    print("="*50)
    print(f"Accuracy:           {accuracy*100:.2f}%")
    print(f"Precision:          {precision*100:.2f}%")
    print(f"Recall:             {recall*100:.2f}%")
    print(f"F1-Score:           {f1*100:.2f}%")
    print("-" * 50)
    print("Confusion Matrix:")
    print(f"Predicted [0] | Actual [0]: {cm[0][0]} (TN)")
    print(f"Predicted [1] | Actual [0]: {cm[0][1]} (FP)")
    print(f"Predicted [0] | Actual [1]: {cm[1][0]} (FN)")
    print(f"Predicted [1] | Actual [1]: {cm[1][1]} (TP)")
    print("="*50)

    # 6. Show Failures
    print("\nFAILED TEST CASES:")
    has_failures = False
    for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, predictions)):
        if true_label != pred_label:
            has_failures = True
            t_name = "ADDRESSED" if true_label == 1 else "NOT_ADDRESSED"
            p_name = "ADDRESSED" if pred_label == 1 else "NOT_ADDRESSED"
            print(f"✖ '{text}'\n  Expected: {t_name} | Got: {p_name}\n")

    if not has_failures:
        print("All tests passed perfectly!")

    #=================================================================
    print("\n" + "="*70)
    print("THREE-TIER LOGIC ANALYSIS")
    print("="*70)
    print(f"HIGH_THRESHOLD (Tier2 YES): >= {classifier.HIGH_THRESHOLD}")
    print(f"LOW_THRESHOLD (Tier2 NO):   <= {classifier.LOW_THRESHOLD}")
    print(f"AMBIGUOUS ZONE (Tier3):     {classifier.LOW_THRESHOLD} < score < {classifier.HIGH_THRESHOLD}")
    print("="*70 + "\n")
    
    # Test phrases with their expected tier behavior
    test_phrases = [
        # (text, description)
        ("Lucy, hi!", "Wake word - should be TIER1"),
        ("hey computer", "Wake word - should be TIER1"),
        ("show logs", "Clear command - likely TIER2-YES or high TIER3"),
        ("restart server", "Clear command"),
        ("damn, this coffee sucks", "Self-talk - should be TIER2-NO"),
        ("fuck, I'm tired", "Self-talk - should be TIER2-NO"),
        ("she's so smart", "Talking ABOUT AI - should be TIER2-NO"),
        ("interesting...", "Ambiguous - should be TIER3"),
        ("yes", "Ambiguous - should be TIER3"),
        ("no, not that", "Ambiguous - should be TIER3"),
        ("hmm, I see", "Ambiguous - should be TIER3"),
        ("continue", "Ambiguous command - likely TIER3"),
        ("you understand?", "Ambiguous question - likely TIER3"),
        ("how are you?", "Conversational - likely TIER3"),
        ("what if we try?", "Follow-up - likely TIER3"),
    ]
    
    print(f"{'Text':<45} {'Score': >6} {'Tier':<12} {'Cold': >6} {'@1s':>6} {'@5s':>6}")
    print("-"*85)
    
    for text, desc in test_phrases: 
        score = classifier.predict(text)
        
        # Determine tier
        text_lower = text.lower()
        is_wake = any(w in text_lower for w in classifier.WAKE_WORDS)
        
        if is_wake:
            tier = "TIER1-WAKE"
        elif score >= classifier.HIGH_THRESHOLD:
            tier = "TIER2-YES"
        elif score <= classifier.LOW_THRESHOLD:
            tier = "TIER2-NO"
        else:
            tier = "TIER3-AMB"
        
        # Test at different time contexts
        cold_result = classifier.should_reply(text, 100.0)
        at_1s = classifier.should_reply(text, 1.0)
        at_5s = classifier.should_reply(text, 5.0)
        
        cold_str = "✅" if cold_result else "❌"
        at_1s_str = "✅" if at_1s else "❌"
        at_5s_str = "✅" if at_5s else "❌"
        
        print(f"{text:<45} {score:>6.2f} {tier:<12} {cold_str:>6} {at_1s_str:>6} {at_5s_str: >6}")
    
    print("\n" + "="*70)
    print("Legend:  Cold = no recent AI speech | @1s/@5s = seconds since AI spoke")
    print("="*70)


    
    print("\n" + "="*70)
    print("AMBIGUOUS PHRASES:  CONTEXT SENSITIVITY TEST")
    print("="*70)
    print("These phrases are INTENTIONALLY ambiguous. The model can't know.")
    print("Context (time since AI spoke) should be the deciding factor.")
    print("="*70 + "\n")
    
    ambiguous = [
        "interesting...",
        "yes",
        "no",
        "I see",
        "good",
        "okay",
        "continue",
        "next",
        "and then?",
        "so what?",
        "seriously?",
        "wow",
        "well let's",
        "try it",
    ]
    
    time_points = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0, 9.0, 15.0]
    
    print(f"{'Phrase':<20}", end="")
    for t in time_points:
        print(f"{t:>5}s", end=" ")
    print(f"{'Score':>7}")
    print("-" * 75)
    
    for phrase in ambiguous:
        score = classifier.predict(phrase)
        print(f"{phrase:<20}", end="")
        
        for t in time_points:
            result = classifier.should_reply(phrase, t)
            symbol = "✅" if result else "·"  # Using dot for cleaner view
            print(f"{symbol: >5}", end=" ")
        
        print(f"{score:>7.2f}")
    
    print("\n" + "-"*75)
    print("✅ = will reply | · = will ignore")
    print(f"Conversation window: {classifier.CONVERSATION_WINDOW}s")


    
    print("\n" + "="*70)
    print("REALISTIC CONVERSATION SIMULATION")
    print("="*70 + "\n")
    
    last_ai_spoke = 0
    
    conversation = [
        # (user_says, delay_seconds, expected_response, scenario_note)
        ("Lucy, hi!", 0, True, "Wake word starts convo"),
        ("how are you?", 1.5, True, "Quick follow-up"),
        ("what can you do?", 2.0, True, "Continuing conversation"),
        ("interesting", 1.0, True, "Short reaction - AMBIGUOUS but recent"),
        ("show example", 1.5, True, "Command in active convo"),
        ("yeah, got it", 2.0, True, "Confirmation"),
        # User turns away to talk to friend
        ("Vasya, look what she can do", 5.0, False, "Talking to friend Vasya"),
        ("yeah, cool thing", 3.0, False, "Still talking to friend"),
        ("well alright, let's go", 4.0, False, "Leaving with friend"),
        # Long pause, comes back
        ("Lucy, another question", 20.0, True, "Wake word after long pause"),
        ("thanks, all clear", 2.0, True, "Closing in active convo"),
    ]
    
    for user_text, delay, expected, note in conversation:
        if delay > 0:
            print(f"    ... {delay}s pause ...")
            time.sleep(delay)
        
        if last_ai_spoke == 0:
            time_since_ai = 999
        else: 
            time_since_ai = time.time() - last_ai_spoke
        
        result = classifier.should_reply(user_text, time_since_ai)
        
        status = "✅" if result == expected else "❌"
        print(f"{status} User: \"{user_text}\"")
        print(f"   ⏱ {time_since_ai:.1f}s since AI | Expected: {expected} | Got: {result}")
        print(f"   📝 {note}")
        
        if result: 
            print(f"   🤖 AI responds...")
            last_ai_spoke = time.time()
        print()
