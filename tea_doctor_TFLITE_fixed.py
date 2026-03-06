"""
Tea Leaf Disease Detector - Streamlit App (TFLite v3.6)

ECA Tri-Branch Fusion CNN for Real-Time Tea Disease Detection.
Uses the v3.6 TFLite model with 8-channel colour + 11-channel texture
feature extraction and optional post-hoc refinement (temperature scaling
+ per-class thresholds).

Model : EfficientNetV2-B0 + 2x MobileNetV2 alpha=0.35 + ECA(k=5)
Params: 8.1 M total, 3.3 M trainable
TFLite: 9.05 MB - 27 ms - 37.6 FPS
"""

import streamlit as st
import cv2
import numpy as np
import json
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

# Optional: LBP from scikit-image (graceful fallback if missing)
try:
    from skimage.feature import local_binary_pattern
    HAS_LBP = True
except ImportError:
    HAS_LBP = False

# ============================================================================
# PATHS  -  point at the real v3.6 artefacts
# ============================================================================

MODEL_DIR = Path(r"C:\Users\Admin\Downloads\New tea")
TFLITE_PATH = MODEL_DIR / "fusion_model_baseline.tflite"
REFINE_CFG_PATH = MODEL_DIR / "refined_tflite_config.json"
RESULTS_PATH = MODEL_DIR / "results_final (1).json"

# Fallback: look for model in the same directory as this script
SCRIPT_DIR = Path(__file__).resolve().parent
TFLITE_PATH_LOCAL = SCRIPT_DIR / "fusion_model_baseline.tflite"
REFINE_CFG_LOCAL = SCRIPT_DIR / "refined_tflite_config.json"

# ============================================================================
# CONSTANTS
# ============================================================================

CLASS_NAMES = [
    "Brown_Blight", "Gray_Blight", "Healthy_leaf",
    "Helopeltis", "Red_Rust", "Red_spider", "Sunlight_Scorching",
]
IMG_SIZE = 224
COLOR_CHANNELS = 8
TEXTURE_CHANNELS = 11
LBP_RADIUS = 1
LBP_POINTS = 8

# ============================================================================
# TRANSLATIONS DICTIONARY
# ============================================================================

TRANSLATIONS = {
    "home_title": {
        "en": "Tea Doctor - AI Disease Detection",
        "hi": "चाय डॉक्टर - एआई रोग पहचान",
        "as": "চাৰ ডক্টর - এআই ৰোগ চিনাক্তকৰণ",
        "sa": "चाय डॉक्टर - एआई रोग पहचान",
    },
    "upload_image": {
        "en": "Upload a Tea Leaf Image",
        "hi": "चाय की पत्ती की तस्वीर अपलोड करें",
        "as": "চাৰ পাতৰ ছবি আপলোড কৰক",
        "sa": "चाय पत्ती की तस्वीर अपलोड करें",
    },
    "analyzing": {
        "en": "Analyzing image...",
        "hi": "तस्वीर का विश्लेषण जारी है...",
        "as": "ছবি বিশ্লেষণ চলছে...",
        "sa": "तस्वीर का विश्लेषण जारी है...",
    },
    "confidence": {
        "en": "Confidence",
        "hi": "आत्मविश्वास",
        "as": "আত্মবিশ্বাস",
        "sa": "आत्मविश्वास",
    },
    "error_not_leaf": {
        "en": "This doesn't look like a tea leaf. Please upload a tea leaf image.",
        "hi": "यह चाय की पत्ती नहीं लगती। कृपया चाय की पत्ती की तस्वीर अपलोड करें।",
        "as": "এটি চাৰ পাত যেন নলাগে। চাৰ পাতৰ ছবি আপলোড কৰক।",
        "sa": "यह चाय की पत्ती नहीं लगती। चाय की पत्ती की तस्वीर अपलोड करें।",
    },
    "error_blurry": {
        "en": "Image is too blurry. Please take a clearer photo.",
        "hi": "तस्वीर बहुत धुंधली है। कृपया स्पष्ट तस्वीर लें।",
        "as": "ছবি অতি ধোঁৱালি। স্পষ্ট ছবি লওক।",
        "sa": "तस्वीर बहुत धुंधली है। स्पष्ट तस्वीर लें।",
    },
    "show_heatmap": {
        "en": "Show Attention Map",
        "hi": "ध्यान नक्शा दिखाएं",
        "as": "মনোযোগ মানচিত্র দেখান",
        "sa": "ध्यान नक्शा दिखाएं",
    },
    "original": {
        "en": "Original",
        "hi": "मूल",
        "as": "মূল",
        "sa": "मूल",
    },
    "quality_check": {
        "en": "Quality Check",
        "hi": "गुणवत्ता जांच",
        "as": "গুণমান পরীক্ষা",
        "sa": "गुणवत्ता जांच",
    },
    "preprocessing": {
        "en": "Preprocessing...",
        "hi": "पूर्व-प्रक्रिया...",
        "as": "পূর্ব-প্রক্রিয়াকরণ...",
        "sa": "पूर्व-प्रक्रिया...",
    },
    "preprocessed": {
        "en": "Preprocessed",
        "hi": "प्रक्रिया किया गया",
        "as": "প্রক্রিয়াকৃত",
        "sa": "प्रक्रिया किया गया",
    },
    "analysis_complete": {
        "en": "Analysis complete.",
        "hi": "विश्लेषण पूर्ण।",
        "as": "বিশ্লেষণ সম্পূর্ণ।",
        "sa": "विश्लेषण पूर्ण।",
    },
    "healthy_leaf_msg": {
        "en": "This leaf appears healthy.",
        "hi": "यह पत्ती स्वस्थ दिखाई देती है।",
        "as": "এই পাতটি সুস্থ দেখাচ্ছে।",
        "sa": "यह पत्ती स्वस्थ दिखाई देती है।",
    },
    "high_confidence": {
        "en": "High confidence",
        "hi": "उच्च आत्मविश्वास",
        "as": "উচ্চ আত্মবিশ্বাস",
        "sa": "उच्च आत्मविश्वास",
    },
    "medium_confidence": {
        "en": "Medium confidence",
        "hi": "मध्यम आत्मविश्वास",
        "as": "মধ্যম আত্মবিশ্বাস",
        "sa": "मध्यम आत्मविश्वास",
    },
    "low_confidence": {
        "en": "Low confidence",
        "hi": "कम आत्मविश्वास",
        "as": "কম আত্মবিশ্বাস",
        "sa": "कम आत्मविश्वास",
    },
    "what_is_this": {
        "en": "What is this?",
        "hi": "यह क्या है?",
        "as": "এটি কি?",
        "sa": "यह क्या है?",
    },
    "how_spreads": {
        "en": "How it spreads",
        "hi": "यह कैसे फैलता है",
        "as": "এটি কীভাবে ছড়ায়",
        "sa": "यह कैसे फैलता है",
    },
    "causes_label": {
        "en": "Causes",
        "hi": "कारण",
        "as": "কারণ",
        "sa": "कारण",
    },
    "treatment_label": {
        "en": "Treatment",
        "hi": "उपचार",
        "as": "চিকিৎসা",
        "sa": "उपचार",
    },
    "prevention_label": {
        "en": "Prevention",
        "hi": "रोकथाम",
        "as": "প্রতিরোধ",
        "sa": "रोकथाम",
    },
    "all_probabilities": {
        "en": "All Probabilities",
        "hi": "सभी संभावनाएं",
        "as": "সমস্ত সম্ভাবনা",
        "sa": "सभी संभावनाएं",
    },
    "disease": {
        "en": "Disease",
        "hi": "रोग",
        "as": "ৰোগ",
        "sa": "रोग",
    },
    "attention_map": {
        "en": "Attention Map",
        "hi": "ध्यान मानचित्र",
        "as": "মনোযোগ মানচিত্র",
        "sa": "ध्यान मानचित्र",
    },
    "severity_label": {
        "en": "Severity",
        "hi": "गंभीरता",
        "as": "গুরুত্ব",
        "sa": "गंभीरता",
    },
    "about_title": {
        "en": "About Tea Doctor",
        "hi": "चाय डॉक्टर के बारे में",
        "as": "চাৰ ডক্টর সম্পর্কে",
        "sa": "चाय डॉक्टर के बारे में",
    },
    "ai_performance": {
        "en": "AI Performance",
        "hi": "एआई प्रदर्शन",
        "as": "এআই কর্মক্ষমতা",
        "sa": "एआई प्रदर्शन",
    },
    "emergency_helplines": {
        "en": "Emergency Helplines",
        "hi": "आपातकालीन हेल्पलाइन",
        "as": "জরুরি সহায়তা লাইন",
        "sa": "आपातकालीन हेल्पलाइन",
    },
    "banking_support": {
        "en": "Banking Support",
        "hi": "बैंकिंग समर्थन",
        "as": "ব্যাংকিং সহায়তা",
        "sa": "बैंकिंग समर्थन",
    },
    "service_label": {
        "en": "Service:",
        "hi": "सेवा:",
        "as": "সেবা:",
        "sa": "सेवा:",
    },
    "subsidies_smartcards": {
        "en": "Subsidies & Smart Cards",
        "hi": "सब्सिडी और स्मार्ट कार्ड",
        "as": "ভর্তুকি এবং স্মার্ট কার্ড",
        "sa": "सब्सिडी और स्मार्ट कार्ड",
    },
    "agri_advice": {
        "en": "24/7 Agri Advice",
        "hi": "24/7 कृषि सलाह",
        "as": "24/7 কৃষি পরামর্শ",
        "sa": "24/7 कृषि सलाह",
    },
    "disease_emergency": {
        "en": "Disease Emergency",
        "hi": "रोग आपातकाल",
        "as": "ৰোগ জরুরী",
        "sa": "रोग आपातकाल",
    },
    "plant_protection": {
        "en": "Plant Protection",
        "hi": "पौधा संरक्षण",
        "as": "উদ্ভিদ সুরক্ষা",
        "sa": "पौधा संरक्षण",
    },
    "local_support": {
        "en": "Local Support",
        "hi": "स्थानीय सहायता",
        "as": "স্থানীয় সহায়তা",
        "sa": "स्थानीय सहायता",
    },
    "model_information": {
        "en": "Model Information",
        "hi": "मॉडल जानकारी",
        "as": "মডেল তথ্য",
        "sa": "मॉडल जानकारी",
    },
    "input_tensors": {
        "en": "Input tensors",
        "hi": "इनपुट टेंसर्स",
        "as": "ইনপুট টেনসর",
        "sa": "इनपुट टेंसर्स",
    },
    "output_tensors": {
        "en": "Output tensors",
        "hi": "आउटपुट टेंसर्स",
        "as": "আউটপুট টেনসর",
        "sa": "आउटपुट टेंसर्स",
    },
    "refined_prediction": {
        "en": "Refined prediction (post-hoc calibration applied)",
        "hi": "परिष्कृत भविष्यवाणी (पोस्ट-हॉक कैलिब्रेशन लागू)",
        "as": "পৰিশোধিত পূৰ্বানুমান (পোষ্ট-হক কেলিব্রেশন প্ৰয়োগ)",
        "sa": "परिष्कृत भविष्यवाणी (पोस्ट-हॉक कैलिब्रेशन लागू)",
    },
}

DISEASE_NAMES = {
    "Brown_Blight": {
        "en": "Brown Blight", "hi": "भूरी झुलसा",
        "as": "ব্রাউন ব্লাইট", "sa": "भूरी झुलसा",
    },
    "Gray_Blight": {
        "en": "Gray Blight", "hi": "ग्रे झुलसा",
        "as": "গ্রে ব্লাইট", "sa": "ग्रे झुलसा",
    },
    "Healthy_leaf": {
        "en": "Healthy Leaf", "hi": "स्वस्थ पत्ती",
        "as": "সুস্থ পাত", "sa": "स्वस्थ पत्ती",
    },
    "Helopeltis": {
        "en": "Helopeltis (Mosquito Bug)", "hi": "हेलोपेल्टिस (मच्छर कीट)",
        "as": "হেলোপেল্টিস (মছুনি পোক)", "sa": "हेलोपेल्टिस (मच्छर कीट)",
    },
    "Red_Rust": {
        "en": "Red Rust", "hi": "लाल जंग",
        "as": "ৰঙা ৰং", "sa": "लाल जंग",
    },
    "Red_spider": {
        "en": "Red Spider Mite", "hi": "लाल मकड़ी का कण",
        "as": "ৰঙা মকৰা কণ", "sa": "लाल मकड़ी का कण",
    },
    "Sunlight_Scorching": {
        "en": "Sunlight Scorching", "hi": "धूप से झुलसा",
        "as": "ৰ'দত পোৰা", "sa": "धूप से झुलसा",
    },
}

DISEASE_INFO = {
    "Brown_Blight": {
        "severity": "medium",
        "what_is": {
            "en": "Brown blight is a fungal disease that causes brown circular lesions on leaves, often after physical damage or wet conditions.",
            "hi": "भूरी झुलसा एक कवक रोग है जो भौतिक क्षति या गीली स्थितियों के बाद पत्तियों पर भूरे गोलाकार घाव का कारण बनता है।",
            "as": "ব্রাউন ব্লাইট এক ভেঁকুৰ ৰোগ যি ভৌতিক ক্ষতি বা আর্দ্র অৱস্থাৰ পিছত পাতত ভূরে বৃত্তাকাৰ ক্ষত সৃষ্টি কৰে।",
            "sa": "भूरी झुलसा एक कवक रोग है जो भौतिक क्षति या गीली स्थितियों के बाद पत्तियों पर भूरे गोलाकार घाव का कारण बनता है।",
        },
        "spread": {
            "en": "Spreads via rain splash and high humidity; favored during monsoon conditions.",
            "hi": "बारिश के छींटे और उच्च आर्द्रता के माध्यम से फैलता है; मानसून के दौरान अनुकूल।",
            "as": "বৃষ্টি আৰু উচ্চ আৰ্দ্ৰতাৰ মাধ্যমে ফৈলে; বর্ষাকালে অনুকূল।",
            "sa": "बारिश के छींटे और उच्च आर्द्रता के माध्यम से फैलता है; मानसून के दौरान अनुकूल।",
        },
        "causes": {
            "en": "Rough plucking, hail or physical damage, prolonged leaf wetness, and poor drainage.",
            "hi": "कठोर तोड़ना, ओले या भौतिक क्षति, लंबे समय तक पत्ती की नमी, और खराब जल निकासी।",
            "as": "কঠোৰ তোড়া, শিলা বা ভৌতিক ক্ষতি, দীর্ঘস্থায়ী পাত সেতুপন, আৰু বেয়া জল निকাশী।",
            "sa": "कठोर तोड़ना, ओले या भौतिक क्षति, लंबे समय तक पत्ती की नमी, और खराब जल निकासी।",
        },
        "cure": {
            "en": "Apply Copper Oxychloride fungicide per label. Remove and destroy infected leaves. Improve drainage.",
            "hi": "निर्देशानुसार कॉपर ऑक्सीक्लोराइड कवकनाशी लागू करें। संक्रमित पत्तियों को हटाएं और नष्ट करें। जल निकासी में सुधार करें।",
            "as": "নির্দেশ অনুসাৰে কপাৰ অক্সিক্লোৰাইড কৰ্কটনাশী প্ৰয়োগ কৰক। সংক্রমিত পাত আঁতৰাওক আৰু ধ্বংস কৰক। জল নিৰ্কাশনে উন্নতি কৰক।",
            "sa": "निर्देशानुसार कॉपर ऑक्सीक्लोराइड कवकनाशी लागू करें। संक्रमित पत्तियों को हटाएं और नष्ट करें। जल निकासी में सुधार करें।",
        },
        "prevention": {
            "en": "Avoid wounds during plucking, improve field drainage, follow preventive spray schedules.",
            "hi": "तोड़ते समय घाव से बचें, खेत की जल निकासी में सुधार करें, रोकथाम स्प्रे कार्यक्रम का पालन करें।",
            "as": "তোড়াৰ সময় ক্ষত এড়াওক, খেতৰ জল নিকাশ উন্নত কৰক, প্রতিরোধমূলক স্প্রে সময়সূচী অনুসরণ কৰক।",
            "sa": "तोड़ते समय घाव से बचें, खेत की जल निकासी में सुधार करें, रोकथाम स्प्रे कार्यक्रम का पालन करें।",
        },
    },
    "Gray_Blight": {
        "severity": "high",
        "what_is": {
            "en": "Gray blight produces greyish lesions with dark centres that coalesce into larger patches.",
            "hi": "ग्रे झुलसा भूरे रंग के घाव पैदा करता है जिनके गहरे केंद्र हैं जो बड़े पैच में मिल जाते हैं।",
            "as": "গ্রে ব্লাইট ধূসর বর্ণের ক্ষত উৎপন্ন কৰে যাৰ অন্ধকাৰ কেন্দ্র আছে যি বৃহত্তর প্যাচত মিলিত হয়।",
            "sa": "ग्रे झुलसा भूरे रंग के घाव पैदा करता है जिनके गहरे केंद्र हैं जो बड़े पैच में मिल जाते हैं।",
        },
        "spread": {
            "en": "Disperses rapidly by wind and rain under humid conditions during monsoon periods.",
            "hi": "आर्द्र परिस्थितियों में हवा और वर्षा से तेजी से फैलता है मानसून के दौरान।",
            "as": "আর্দ্ৰ অৱস্থাত বায়ু আৰু বৃষ্টিৰ দ্বাৰা দ্রুত ছড়ায় বর্ষাকালত।",
            "sa": "आर्द्र परिस्थितियों में हवा और वर्षा से तेजी से फैलता है मानसून के दौरान।",
        },
        "causes": {
            "en": "Low soil potash, poor air circulation, dense canopy, and extended wetness.",
            "hi": "कम मिट्टी पोटाश, खराब वायु संचार, घने पत्तियों का आवरण, और विस्तारित नमी।",
            "as": "কম মাটি পোটাশ, খারাপ বায়ু সংবহন, ঘন ক্যানপি, আৰু বিস্তৃত আৰ্দ্ৰতা।",
            "sa": "कम मिट्टी पोटाश, खराब वायु संचार, घने पत्तियों का आवरण, और विस्तारित नमी।",
        },
        "cure": {
            "en": "Use Carbendazim fungicide where appropriate. Prune affected areas. Dispose of fallen debris.",
            "hi": "जहां उपयुक्त हो कार्बेंडाजिम कवकनाशी का उपयोग करें। प्रभावित क्षेत्रों को काटें। गिरी हुई मलबे का निपटान करें।",
            "as": "যেখানে উপযুক্ত কার্বেন্ডাজিম কৰ্কটনাশী ব্যবহাৰ কৰক। প্ৰভাৱিত এলাকা কাটক। পরিত্যক্ত ধ্বংসাৱশেষ নিষ্পত্তি কৰক।",
            "sa": "जहां उपयुक्त हो कार्बेंडाजिम कवकनाशी का उपयोग करें। प्रभावित क्षेत्रों को काटें। गिरी हुई मलबे का निपटान करें।",
        },
        "prevention": {
            "en": "Improve ventilation by pruning, maintain balanced fertilization with potash, keep field hygiene.",
            "hi": "कटाई-छंटाई से हवा का संचार बढ़ाएं, पोटाश के साथ संतुलित खाद बनाए रखें, खेत की स्वच्छता रखें।",
            "as": "কাটাই-কাটি দ্বাৰা বায়ু সংবহন উন্নত কৰক, পোটাশৰ সৈতে সুষম সার বজায় ৰাখক, খেতৰ স্বাস্থ্যবিধি বজায় ৰাখক।",
            "sa": "कटाई-छंटाई से हवा का संचार बढ़ाएं, पोटाश के साथ संतुलित खाद बनाए रखें, खेत की स्वच्छता रखें।",
        },
    },
    "Healthy_leaf": {
        "severity": "none",
        "what_is": {
            "en": "This leaf appears healthy with uniform green colour and no visible lesions.",
            "hi": "यह पत्ती समान हरे रंग के साथ स्वस्थ दिखाई देती है और कोई दृश्यमान घाव नहीं हैं।",
            "as": "এই পাতটি একীভূত সবুজ ৰঙ সহ স্বাস্থ্যকৰ দেখাচ্ছে আৰু কোনো দৃশ্যমান ক্ষত নেই।",
            "sa": "यह पत्ती समान हरे रंग के साथ स्वस्थ दिखाई देती है और कोई दृश्यमान घाव नहीं हैं।",
        },
        "spread": {
            "en": "Healthy leaves do not spread disease.",
            "hi": "स्वस्थ पत्तियां रोग नहीं फैलाती।",
            "as": "স্বাস্থ্যকৰ পাতে ৰোগ ছড়ায় নহয়।",
            "sa": "स्वस्थ पत्तियां रोग नहीं फैलाती।",
        },
        "causes": {
            "en": "Good nutrition, proper shade balance, and regular care.",
            "hi": "अच्छा पोषण, उचित छाया संतुलन, और नियमित देखभाल।",
            "as": "ভাল পুষ্টি, সঠিক ছাঁ সন্তুলন, আৰু নিয়মিত যত্ন।",
            "sa": "अच्छा पोषण, उचित छाया संतुलन, और नियमित देखभाल।",
        },
        "cure": {
            "en": "No treatment required; continue good agronomic practices and monitoring.",
            "hi": "कोई इलाज की आवश्यकता नहीं; अच्छी कृषि पद्धतियों और निगरानी जारी रखें।",
            "as": "কোনো চিকিৎসাৰ প্রয়োজন নেই; ভাল কৃষি অনুশীলন আৰু নিৰীক্ষণ অব্যাহত ৰাখক।",
            "sa": "कोई इलाज की आवश्यकता नहीं; अच्छी कृषि पद्धतियों और निगरानी जारी रखें।",
        },
        "prevention": {
            "en": "Balanced fertilization, proper drainage, and regular inspection.",
            "hi": "संतुलित खाद, उचित जल निकास, और नियमित निरीक्षण।",
            "as": "সুষম সার, সঠিক জল নিকাশন, আৰু নিয়মিত পরিদর্শন।",
            "sa": "संतुलित खाद, उचित जल निकास, और नियमित निरीक्षण।",
        },
    },
    "Helopeltis": {
        "severity": "critical",
        "what_is": {
            "en": "Helopeltis (mosquito bug) is an insect pest that pierces buds and young shoots, causing blackening.",
            "hi": "हेलोपेल्टिस (मच्छर कीट) एक कीट है जो कलियों और युवा अंकुरों को छेदता है, काला पड़ना पैदा करता है।",
            "as": "হেলোপেল্টিস (মছুনি পোক) এক পোক যি কলি আৰু যুবক চুপত ছিদ্র কৰে, কালো পৰা সৃষ্টি কৰে।",
            "sa": "हेलोपेल्टिस (मच्छर कीट) एक कीट है जो कलियों और युवा अंकुरों को छेदता है, काला पड़ना पैदा करता है।",
        },
        "spread": {
            "en": "Moves between bushes and increases under dense shade and weed growth.",
            "hi": "झाड़ियों के बीच चलता है और घने छाया और खरपतवार की वृद्धि के तहत बढ़ता है।",
            "as": "ঝাড়ীৰ মাজত চলে আৰু ঘন ছাঁ আৰু বন্য বৃদ্ধিৰ তলত বৃদ্ধি পায়।",
            "sa": "झाड़ियों के बीच चलता है और घने छाया और खरपतवार की वृद्धि के तहत बढ़ता है।",
        },
        "causes": {
            "en": "Excessive shade, heavy weed presence, dense planting, and lack of field sanitation.",
            "hi": "अत्यधिक छाया, भारी खरपतवार की उपस्थिति, घनी बुवाई, और खेत की स्वच्छता की कमी।",
            "as": "অতিরিক্ত ছাঁ, ভারী বন্য উপস্থিতি, ঘন রোপণ, আৰু খেতৰ স্বাস্থ্যবিধিৰ অভাৱ।",
            "sa": "अत्यधिक छाया, भारी खरपतवार की उपस्थिति, घनी बुवाई, और खेत की स्वच्छता की कमी।",
        },
        "cure": {
            "en": "Apply recommended insecticides per label. Remove damaged shoots. Repeat treatments as advised.",
            "hi": "निर्देश अनुसार अनुशंसित कीटनाशक लागू करें। क्षतिग्रस्त अंकुर हटाएं। सलाह के अनुसार उपचार दोहराएं।",
            "as": "নির্দেশ অনুসাৰে অনুমোদিত কীটনাশক প্ৰয়োগ কৰক। ক্ষতিগ্রস্ত অঙ্কুৰ আঁতৰাওক। পরামৰ্শ অনুসাৰে চিকিৎসা পুনৰাবৃত্তি কৰক।",
            "sa": "निर्देश अनुसार अनुशंसित कीटनाशक लागू करें। क्षतिग्रस्त अंकुर हटाएं। सलाह के अनुसार उपचार दोहराएं।",
        },
        "prevention": {
            "en": "Maintain weed control, reduce excessive shade, and monitor daily for early detection.",
            "hi": "खरपतवार नियंत्रण बनाए रखें, अत्यधिक छाया कम करें, और जल्दी पता लगाने के लिए दैनिक निगरानी करें।",
            "as": "বন্য নিয়ন্ত্রণ বজায় ৰাখক, অতিরিক্ত ছাঁ কম কৰক, আৰু প্রাথমিক সনাক্তকৰণৰ জন্য দৈনিক নিৰীক্ষণ কৰক।",
            "sa": "खरपतवार नियंत्रण बनाए रखें, अत्यधिक छाया कम करें, और जल्दी पता लगाने के लिए दैनिक निगरानी करें।",
        },
    },
    "Red_Rust": {
        "severity": "high",
        "what_is": {
            "en": "Red rust is a rust-like or algal growth that appears as orange, velvety patches on stems or leaves.",
            "hi": "लाल जंग एक जंग जैसी या शैवाल वृद्धि है जो तनों या पत्तियों पर नारंगी, मखमली पैच के रूप में दिखाई देती है।",
            "as": "ৰঙা ৰং এক জংৰ দৰে বা শৈবাল বৃদ্ধি যি কাণ্ড বা পাতত নাৰঙ্গী, মখমলী প্যাচ হিসাবে দেখা যায়।",
            "sa": "लाल जंग एक जंग जैसी या शैवाल वृद्धि है जो तनों या पत्तियों पर नारंगी, मखमली पैच के रूप में दिखाई देती है।",
        },
        "spread": {
            "en": "Can spread as dust or spores under dry, windy conditions affecting large areas quickly.",
            "hi": "सूखी, हवादार परिस्थितियों में धूल या बीजाणु के रूप में फैल सकता है।",
            "as": "শুকান, বায়ুৱহ অৱস্থাত ধূলি বা বীজাণু হিসাবে ছড়াই পাৰে।",
            "sa": "सूखी, हवादार परिस्थितियों में धूल या बीजाणु के रूप में फैल सकता है।",
        },
        "causes": {
            "en": "Waterlogged soils, weak or stressed plants, low potash levels, and poor air circulation.",
            "hi": "जलभराव वाली मिट्टी, कमजोर या तनावग्रस्त पौधे, कम पोटाश स्तर, और खराब वायु संचार।",
            "as": "জলভরা মাটি, দুর্বল বা চাপগ্রস্ত গছ, কম পোটাশ স্তৰ, আৰু খারাপ বায়ু সংবহন।",
            "sa": "जलभराव वाली मिट्टी, कमजोर या तनावग्रस्त पौधे, कम पोटाश स्तर, और खराब वायु संचार।",
        },
        "cure": {
            "en": "Use appropriate fungicide or algaecide as recommended. Remove affected material. Improve drainage.",
            "hi": "अनुशंसित कवकनाशी या शैवाल नाशी का उपयोग करें। प्रभावित सामग्री को हटाएं। जल निकास में सुधार करें।",
            "as": "অনুমোদিত কৰ্কটনাশী বা শৈবাল নাশী ব্যবহাৰ কৰক। প্ৰভাৱিত সামগ্রী আঁতৰাওক। জল নিৰ্কাশন উন্নত কৰক।",
            "sa": "अनुशंसित कवकनाशी या शैवाल नाशी का उपयोग करें। प्रभावित सामग्री को हटाएं। जल निकास में सुधार करें।",
        },
        "prevention": {
            "en": "Improve drainage, maintain balanced fertilization with potash, practice good field hygiene.",
            "hi": "जल निकास में सुधार करें, पोटाश के साथ संतुलित खाद बनाए रखें, अच्छी खेत स्वच्छता का अभ्यास करें।",
            "as": "জল নিৰ্কাশনে উন্নতি কৰক, পোটাশৰ সৈতে সুষম সার বজায় ৰাখক, ভাল খেত স্বাস্থ্যবিধি অনুশীলন কৰক।",
            "sa": "जल निकास में सुधार करें, पोटाश के साथ संतुलित खाद बनाए रखें, अच्छी खेत स्वच्छता का अभ्यास करें।",
        },
    },
    "Red_spider": {
        "severity": "high",
        "what_is": {
            "en": "Red spider mite causes stippling, discoloration, and leaf bronzing due to sap-sucking damage.",
            "hi": "लाल मकड़ी का कण धब्बे, विरंजन, और कांस्य रंग का कारण बनता है रस चूसने के कारण।",
            "as": "ৰঙা মকৰা কণ ধব্বা, বিবৰ্ণতা, আৰু পত্ৰ ব্রোঞ্জ কাৰণ কৰে ৰস চোষণৰ কাৰণে।",
            "sa": "लाल मकड़ी का कण धब्बे, विरंजन, और कांस्य रंग का कारण बनता है रस चूसने के कारण।",
        },
        "spread": {
            "en": "Favored by hot, dry, dusty conditions; spreads by wind or contact.",
            "hi": "गर्म, सूखी, धूलदार परिस्थितियों में पनपता है; हवा या संपर्क से फैलता है।",
            "as": "গৰম, শুকান, ধূলিময় অৱস্থাত পনপে; বায়ু বা সংস্পৰ্শত ছড়ায়।",
            "sa": "गर्म, सूखी, धूलदार परिस्थितियों में पनपता है; हवा या संपर्क से फैलता है।",
        },
        "causes": {
            "en": "Hot, dry weather, dusty surroundings, poor shade, and drought stress.",
            "hi": "गर्म, सूखा मौसम, धूलदार परिवेश, कम छाया, और सूखे से तनाव।",
            "as": "গৰম, শুকান বতৰ, ধূলিময় পৰিবেশ, কম ছাঁ, আৰু খৰাঙ্গৰ চাপ।",
            "sa": "गर्म, सूखा मौसम, धूलदार परिवेश, कम छाया, और सूखे से तनाव।",
        },
        "cure": {
            "en": "Apply miticides like sulfur or acaricides as recommended. Use high-pressure water spray to dislodge mites.",
            "hi": "सल्फर या एकारिसाइड जैसे कणनाशी लागू करें। मीठी पानी की तेज बौछार से कण हटाएं।",
            "as": "গন্ধক বা একারিছাইড জাতীয় কণনাশী প্রয়োগ কৰক। উচ্চ চাপৰ পানীৰ ছিটা দিয়ে কণ আঁতৰাওক।",
            "sa": "सल्फर या एकारिसाइड जैसे कणनाशी लागू करें। मीठी पानी की तेज बौछार से कण हटाएं।",
        },
        "prevention": {
            "en": "Maintain adequate irrigation during hot periods, reduce dust, ensure adequate shade and regular monitoring.",
            "hi": "गर्म अवधि में पर्याप्त सिंचाई बनाए रखें, धूल कम करें, पर्याप्त छाया और नियमित निगरानी सुनिश्चित करें।",
            "as": "গৰম সময়ত পৰ্যাপ্ত সিঞ্চন বজায় ৰাখক, ধূলি কম কৰক, পৰ্যাপ্ত ছাঁ আৰু নিয়মিত নিৰীক্ষণ নিশ্চিত কৰক।",
            "sa": "गर्म अवधि में पर्याप्त सिंचाई बनाए रखें, धूल कम करें, पर्याप्त छाया और नियमित निगरानी सुनिश्चित करें।",
        },
    },
    "Sunlight_Scorching": {
        "severity": "low",
        "what_is": {
            "en": "Sunlight scorching is non-biological leaf burn from intense sun and heat, not a disease.",
            "hi": "धूप का झुलसा तीव्र सूर्य और गर्मी से गैर-जैविक पत्ती की जलन है, रोग नहीं।",
            "as": "ৰ'দত পোৰা তীব্ৰ সূৰ্য আৰু গর্মত পৰা গৈৰ-জৈব পাত বিনাশ, ৰোগ নহয়।",
            "sa": "धूप का झुलसा तीव्र सूर्य और गर्मी से गैर-जैविक पत्ती की जलन है, रोग नहीं।",
        },
        "spread": {
            "en": "Not contagious; affects exposed plants during heatwaves when shade is insufficient.",
            "hi": "संक्रामक नहीं; जब छाया अपर्याप्त हो तो लू के दौरान उजागर पौधों को प्रभावित करता है।",
            "as": "সংক্ৰামক নহয়; যেতিয়া ছাঁ অপৰ্যাপ্ত হয় তেতিয়া তাপপ্ৰবাহৰ সময়ত উন্মুক্ত গছক প্ৰভাৱিত কৰে।",
            "sa": "संक्रामक नहीं; जब छाया अपर्याप्त हो तो लू के दौरान उजागर पौधों को प्रभावित करता है।",
        },
        "causes": {
            "en": "Lack of shade, extreme heat over 38 C, south-facing exposure, tender growth in harsh sun.",
            "hi": "छाया की कमी, 38 C से अधिक चरम गर्मी, दक्षिण-सामना करने वाला संपर्क, कठोर धूप में कोमल वृद्धि।",
            "as": "ছাঁৰ অভাৱ, 38 C তকৈ অধিক চৰম উষ্ণতা, দক্ষিণ-সামনা সংস্পৰ্শ, কঠোৰ ৰ'দত কোমল বৃদ্ধি।",
            "sa": "छाया की कमी, 38 C से अधिक चरम गर्मी, दक्षिण-सामना करने वाला संपर्क, कठोर धूप में कोमल वृद्धि।",
        },
        "cure": {
            "en": "Remove severely scorched leaves. Provide temporary shade. Monitor for secondary fungal infections.",
            "hi": "बहुत अधिक झुलसी पत्तियों को हटाएं। अस्थायी छाया प्रदान करें। माध्यमिक कवक संक्रमण के लिए निगरानी करें।",
            "as": "গুৰুতৰভাৱে পোৰা পাত আঁতৰাওক। অস্থায়ী ছাঁ প্ৰদান কৰক। মাধ্যমিক কবক সংক্ৰমণৰ বাবে নিৰীক্ষণ কৰক।",
            "sa": "बहुत अधिक झुलसी पत्तियों को हटाएं। अस्थायी छाया प्रदान करें। माध्यमिक कवक संक्रमण के लिए निगरानी करें।",
        },
        "prevention": {
            "en": "Plant two-tier shade (tall and short trees). Mulch soil with dry grass. Irrigate during extreme heat.",
            "hi": "दो-स्तरीय छाया लगाएं (ऊंचे और छोटे पेड़)। मिट्टी को सूखी घास से ढकें। चरम गर्मी में सिंचाई करें।",
            "as": "দুই-স্তৰীয় ছাঁ লগাওক (ওখ আৰু চুটি গছ)। মাটি শুকান ঘাঁহেৰে ঢাকক। চৰম উষ্ণতাত সিঞ্চন কৰক।",
            "sa": "दो-स्तरीय छाया लगाएं (ऊंचे और छोटे पेड़)। मिट्टी को सूखी घास से ढकें। चरम गर्मी में सिंचाई करें।",
        },
    },
}

SEVERITY_COLORS = {
    "none": "#22c55e", "low": "#84cc16",
    "medium": "#f59e0b", "high": "#ef4444", "critical": "#991b1b",
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text(key, lang="en"):
    """Get translated text, fallback to English."""
    entry = TRANSLATIONS.get(key, {})
    return entry.get(lang, entry.get("en", key))


def get_disease_name(disease, lang="en"):
    """Get translated disease name."""
    entry = DISEASE_NAMES.get(disease, {})
    return entry.get(lang, entry.get("en", disease))


# ============================================================================
# IMAGE QUALITY & LEAF CHECK
# ============================================================================

def assess_image_quality(img):
    """Return (score 0-100, issues list, acceptable bool)."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(gray))
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())

    issues = []
    if brightness < 15:
        issues.append("Very dark")
    elif brightness > 240:
        issues.append("Very bright")
    if lap_var < 10:
        issues.append("Extremely blurry")
    if contrast < 5:
        issues.append("No contrast")

    score = 100
    if lap_var < 10:
        score -= 30
    if brightness < 15 or brightness > 240:
        score -= 20
    if contrast < 5:
        score -= 20
    score = max(0, score)
    acceptable = score >= 35 and lap_var >= 8 and brightness > 10 and contrast > 3
    return score, issues, acceptable


def check_if_leaf(img):
    """Background-aware tea leaf detection (permissive)."""
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]
        total = h * w

        # Remove white background
        white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 60, 255))
        fg_mask = cv2.bitwise_not(white_mask)
        fg_px = max(int(np.sum(fg_mask > 0)), 1)
        if fg_px < total * 0.03:
            return False

        # Plant-colour masks
        green = cv2.inRange(hsv, (20, 25, 20), (95, 255, 255))
        yellow = cv2.inRange(hsv, (10, 25, 30), (45, 255, 255))
        red1 = cv2.inRange(hsv, (0, 20, 30), (15, 255, 255))
        red2 = cv2.inRange(hsv, (160, 20, 30), (180, 255, 255))
        plant = green | yellow | red1 | red2
        plant_ratio = np.sum(cv2.bitwise_and(plant, fg_mask) > 0) / fg_px * 100

        # Edges
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)
        edge_ratio = np.sum(cv2.bitwise_and(edges, fg_mask) > 0) / fg_px * 100
        lap_var = float(np.var(cv2.Laplacian(gray, cv2.CV_64F)[fg_mask > 0]))

        # Skin heuristic
        skin = cv2.inRange(hsv, (0, 10, 60), (25, 200, 255))
        skin_ratio = np.sum(cv2.bitwise_and(skin, fg_mask) > 0) / fg_px * 100

        color_ok = plant_ratio >= 12
        struct_ok = edge_ratio >= 2.0 and lap_var >= 10.0
        face_like = skin_ratio >= 25 and edge_ratio < 2 and lap_var < 10
        return (color_ok or struct_ok) and not face_like
    except Exception:
        return True  # let the model decide


# ============================================================================
# IMAGE PREPROCESSING (light denoise + CLAHE)
# ============================================================================

def preprocess_image(img):
    img = cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7,
                                           templateWindowSize=7, searchWindowSize=21)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# ============================================================================
# FEATURE EXTRACTORS  -  exact mirror of tea_train_v3_6 notebook
# ============================================================================

def extract_color_features(img_float):
    """
    8-channel colour feature map (v3.6).
    Input : float32 [H,W,3] in [0,1], RGB.
    Output: float32 [H,W,8].
    Channels: H, S, CLAHE-L*, a*, b*, ExG, a*/b*, R/G
    """
    img_u8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV).astype(np.float32)
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB).astype(np.float32)

    H = hsv[:, :, 0] / 180.0
    S = hsv[:, :, 1] / 255.0

    L_star = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    L_clahe = clahe.apply(L_star.astype(np.uint8)).astype(np.float32) / 255.0

    a_star = lab[:, :, 1] / 255.0
    b_star = lab[:, :, 2] / 255.0

    R, G, B = img_float[:, :, 0], img_float[:, :, 1], img_float[:, :, 2]
    ExG = np.clip(2 * G - R - B, -1, 1)

    ab_ratio = np.where(
        np.abs(b_star) > 0.01,
        np.clip(a_star / (b_star + 1e-8), -2, 2) / 4 + 0.5,
        0.5,
    )
    rg_ratio = np.where(
        G > 0.01,
        np.clip(R / (G + 1e-8), 0, 4) / 4,
        0.5,
    )

    return np.stack(
        [H, S, L_clahe, a_star, b_star, ExG, ab_ratio, rg_ratio], axis=-1
    ).astype(np.float32)


def extract_texture_features(img_float):
    """
    11-channel texture feature map (v3.6).
    Input : float32 [H,W,3] in [0,1], RGB.
    Output: float32 [H,W,11].
    Channels:
      0  Gray              5  MorphGrad          10 HueWeightedEdge
      1  Canny             6  CLAHE gray
      2  Gabor 0 deg       7  Lesion density
      3  Gabor 45 deg      8  LBP gray
      4  Local std-dev     9  LBP a*
    """
    img_u8 = np.clip(img_float * 255, 0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32) / 255.0

    # Canny
    canny = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0

    # Gabor (0 deg, 45 deg)
    gabor_out = []
    for theta in [0, np.pi / 4]:
        kern = cv2.getGaborKernel((21, 21), sigma=4.0, theta=theta,
                                   lambd=10.0, gamma=0.5, psi=0)
        resp = cv2.filter2D(gray, cv2.CV_32F, kern)
        resp = np.clip(np.abs(resp) / (np.abs(resp).max() + 1e-8), 0, 1)
        gabor_out.append(resp)

    # Local std-dev
    mu = cv2.blur(gray_f, (7, 7))
    sq = cv2.blur(gray_f ** 2, (7, 7))
    local_std = np.sqrt(np.clip(sq - mu ** 2, 0, None))
    local_std = local_std / (local_std.max() + 1e-8)

    # Morphological gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morph_grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel).astype(np.float32) / 255.0

    # CLAHE gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray).astype(np.float32) / 255.0

    # Lesion density (brown-ish mask blurred)
    hsv = cv2.cvtColor(img_u8, cv2.COLOR_RGB2HSV)
    brown_mask = cv2.inRange(hsv, (8, 60, 40), (30, 255, 200))
    lesion = cv2.blur(brown_mask.astype(np.float32) / 255.0, (15, 15))

    # LBP on grayscale
    if HAS_LBP:
        lbp_gray = local_binary_pattern(gray, P=LBP_POINTS, R=LBP_RADIUS,
                                         method="uniform").astype(np.float32) / (LBP_POINTS + 2)
    else:
        # Fallback: approximate LBP with local variance
        lbp_gray = local_std.copy()

    # LBP on a* channel
    lab = cv2.cvtColor(img_u8, cv2.COLOR_RGB2LAB)
    a_star = lab[:, :, 1]
    if HAS_LBP:
        lbp_a = local_binary_pattern(a_star, P=LBP_POINTS, R=LBP_RADIUS,
                                      method="uniform").astype(np.float32) / (LBP_POINTS + 2)
    else:
        a_f = a_star.astype(np.float32) / 255.0
        mu_a = cv2.blur(a_f, (7, 7))
        sq_a = cv2.blur(a_f ** 2, (7, 7))
        lbp_a = np.sqrt(np.clip(sq_a - mu_a ** 2, 0, None))
        lbp_a = lbp_a / (lbp_a.max() + 1e-8)

    # Hue-weighted edge magnitude (saturation x Sobel)
    sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge_mag = np.sqrt(sx ** 2 + sy ** 2)
    edge_mag = edge_mag / (edge_mag.max() + 1e-8)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    hue_edge = edge_mag * sat
    hue_edge = hue_edge / (hue_edge.max() + 1e-8)

    return np.stack([
        gray_f, canny, gabor_out[0], gabor_out[1],
        local_std, morph_grad, clahe_gray, lesion,
        lbp_gray, lbp_a, hue_edge,
    ], axis=-1).astype(np.float32)


# ============================================================================
# POST-HOC REFINEMENT  (temperature scaling + per-class thresholds)
# ============================================================================

@st.cache_data
def load_refinement_config():
    """Load refined_tflite_config.json -> (temperature, thresholds_array)."""
    for p in [REFINE_CFG_PATH, REFINE_CFG_LOCAL]:
        if p.exists():
            cfg = json.loads(p.read_text())
            temp = float(cfg.get("temperature", 1.0))
            thresh = cfg.get("thresholds_array", [0.5] * 7)
            return temp, np.array(thresh, dtype=np.float32)
    return 1.0, np.ones(7, dtype=np.float32) * 0.5


def apply_refinement(raw_probs, temperature, thresholds):
    """
    Post-hoc refinement matching training notebook Cell 26.
    1) logits = log(probs)   2) scale = logits / T
    3) softmax               4) divide by thresholds
    5) re-normalise
    """
    probs = np.clip(raw_probs, 1e-8, 1.0)
    logits = np.log(probs)
    scaled = logits / temperature
    # Numerically stable softmax
    shifted = scaled - scaled.max()
    exp_s = np.exp(shifted)
    calibrated = exp_s / (exp_s.sum() + 1e-8)
    # Threshold gating
    adjusted = calibrated / (thresholds + 1e-8)
    adjusted = adjusted / (adjusted.sum() + 1e-8)
    return adjusted


# ============================================================================
# TFLITE MODEL LOADING & PREDICTION
# ============================================================================

@st.cache_resource
def load_tflite_model():
    """Load the v3.6 TFLite model (tri-branch fusion)."""
    try:
        import tensorflow as tf
    except ImportError:
        try:
            from tflite_runtime.interpreter import Interpreter as TFInterpreter
            for p in [TFLITE_PATH, TFLITE_PATH_LOCAL]:
                if p.exists():
                    interp = TFInterpreter(str(p))
                    interp.allocate_tensors()
                    return interp, "tflite"
            return None, "model_missing"
        except ImportError:
            return None, "tf_missing"

    for p in [TFLITE_PATH, TFLITE_PATH_LOCAL]:
        if p.exists():
            try:
                interp = tf.lite.Interpreter(model_path=str(p))
                interp.allocate_tensors()
                return interp, "tflite"
            except Exception as e:
                st.error(f"Failed to load model from {p}: {e}")
                return None, "load_error"

    return None, "model_missing"


def predict_disease(img, interpreter):
    """
    Run the tri-branch TFLite model.
    Returns (class_name, confidence_pct, probs_array).
    """
    img_224 = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_f = img_224.astype(np.float32) / 255.0

    # Prepare three inputs
    rgb_input = np.expand_dims(img_224, 0).astype(np.uint8)             # [1,224,224,3]
    color_input = np.expand_dims(extract_color_features(img_f), 0)      # [1,224,224,8]
    texture_input = np.expand_dims(extract_texture_features(img_f), 0)  # [1,224,224,11]

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Map inputs by tensor name
    for det in input_details:
        name = det["name"].lower()
        if "texture" in name:
            interpreter.set_tensor(det["index"], texture_input.astype(det["dtype"]))
        elif "rgb" in name:
            interpreter.set_tensor(det["index"], rgb_input.astype(det["dtype"]))
        elif "color" in name:
            interpreter.set_tensor(det["index"], color_input.astype(det["dtype"]))

    interpreter.invoke()

    probs = interpreter.get_tensor(output_details[0]["index"])[0]
    probs = np.clip(probs, 0, 1)
    if probs.sum() < 0.01:
        probs = np.ones(7) / 7  # safety fallback

    idx = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx] * 100), probs


# ============================================================================
# HEATMAP  (edge + texture proxy - no Grad-CAM without full model)
# ============================================================================

def generate_heatmap(img):
    """SmoothGrad-style edge/texture heatmap."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    lap = np.abs(cv2.Laplacian(gray, cv2.CV_64F))
    lap = lap / (lap.max() + 1e-6)
    heatmap = 0.6 * edges + 0.4 * lap.astype(np.float32)
    for _ in range(3):
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 2.0)
    mn, mx = heatmap.min(), heatmap.max()
    if mx > mn:
        heatmap = (heatmap - mn) / (mx - mn)
    else:
        heatmap = np.zeros_like(heatmap)
    return np.power(heatmap, 0.8)


def overlay_heatmap(img, heatmap, alpha=0.4):
    hm = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    coloured = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    coloured = cv2.cvtColor(coloured, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img, 1 - alpha, coloured, alpha, 0)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    st.set_page_config(
        page_title="Tea Doctor v3.6",
        page_icon="🍵",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # -- Sidebar --
    with st.sidebar:
        st.title("🍵 Tea Doctor")
        st.caption("ECA Tri-Branch Fusion CNN v3.6")

        lang = st.selectbox(
            "🌐 Language",
            ["en", "hi", "as", "sa"],
            format_func=lambda x: {
                "en": "English", "hi": "हिंदी", "as": "অসমীয়া", "sa": "सादरी"
            }[x],
        )
        st.session_state.lang = lang

        st.divider()
        page = st.radio("Navigate", ["🏠 Home", "ℹ️ About"], label_visibility="collapsed")

        st.divider()
        use_refinement = st.toggle("🔬 Post-hoc refinement", value=True,
                                    help="Temperature scaling + per-class thresholds (improves accuracy by ~1.3%)")
        st.session_state.use_refinement = use_refinement

        skip_checks = st.toggle("⚡ Skip leaf checks", value=False,
                                 help="Bypass colour / structure validation")
        st.session_state.skip_checks = skip_checks

    # -- Pages --
    if "Home" in page:
        show_home()
    else:
        show_about()


# -----------------------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------------------
def show_home():
    lang = st.session_state.get("lang", "en")

    st.title(get_text("home_title", lang))
    st.caption("Upload or photograph a tea leaf — the AI will identify the disease and recommend treatment.")
    st.divider()

    # Load model
    interpreter, model_type = load_tflite_model()

    if model_type == "tf_missing":
        st.error("TensorFlow / tflite-runtime is not installed. "
                 "Run `pip install tensorflow` or `pip install tflite-runtime`.")
        st.stop()
    if model_type == "model_missing":
        st.error(f"Model file not found.\n\nSearched:\n- `{TFLITE_PATH}`\n- `{TFLITE_PATH_LOCAL}`\n\n"
                 "Copy `fusion_model_baseline.tflite` to one of these locations.")
        st.stop()
    if interpreter is None:
        st.error("Model failed to load. Check the error above.")
        st.stop()

    # -- Image input --
    col_upload, col_camera = st.columns(2)
    with col_upload:
        uploaded = st.file_uploader(get_text("upload_image", lang),
                                     type=["jpg", "jpeg", "png", "webp"])
    with col_camera:
        cam_img = st.camera_input("📷 Take a photo")

    source = uploaded or cam_img
    if not source:
        st.info("👆 Upload or take a photo to begin.")
        return

    # -- Load & normalise --
    image = np.array(Image.open(source))
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

    # -- Display original vs preprocessed --
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(get_text("original", lang))
        st.image(image, use_container_width=True)

    with st.spinner(get_text("preprocessing", lang)):
        preprocessed = preprocess_image(image)

    with c2:
        st.subheader(get_text("preprocessed", lang))
        st.image(preprocessed, use_container_width=True)

    st.divider()

    # -- Quality / leaf gate --
    if not st.session_state.get("skip_checks", False):
        score, issues, acceptable = assess_image_quality(image)
        if not acceptable:
            st.error(f"❌ {get_text('error_blurry', lang)}  (quality {score}/100: {', '.join(issues)})")
            st.stop()
        if not check_if_leaf(image):
            st.warning(f"⚠️ {get_text('error_not_leaf', lang)}  Proceeding anyway — confidence threshold will judge.")

    # -- Predict --
    with st.spinner(get_text("analyzing", lang)):
        pred_class, confidence, raw_probs = predict_disease(preprocessed, interpreter)

    # -- Optional post-hoc refinement --
    temperature, thresholds = load_refinement_config()
    use_ref = st.session_state.get("use_refinement", True)
    if use_ref:
        refined_probs = apply_refinement(raw_probs, temperature, thresholds)
        pred_idx = int(np.argmax(refined_probs))
        pred_class = CLASS_NAMES[pred_idx]
        confidence = float(refined_probs[pred_idx] * 100)
        display_probs = refined_probs * 100
    else:
        display_probs = raw_probs * 100

    confidence = float(np.clip(confidence, 0, 100))

    # -- Confidence gate --
    if confidence < 30:
        st.error(f"❌ Confidence too low ({confidence:.1f}%).  "
                 "This may not be a tea leaf, or the image quality is poor.")
        st.info("💡 Try: better lighting, different angle, or a real tea leaf image.")
        st.stop()

    # -- Results --
    st.success(get_text("analysis_complete", lang))
    disease_label = get_disease_name(pred_class, lang)

    # Hero result card
    res_col1, res_col2 = st.columns([2, 1])
    with res_col1:
        if "Healthy" in pred_class:
            st.header(f"🌿 {disease_label}")
            st.success(get_text("healthy_leaf_msg", lang))
        else:
            severity = DISEASE_INFO.get(pred_class, {}).get("severity", "medium")
            sev_color = SEVERITY_COLORS.get(severity, "#f59e0b")
            st.header(f"🔬 {disease_label}")
            st.markdown(
                f"**{get_text('severity_label', lang)}:** "
                f"<span style='color:{sev_color}; font-weight:bold'>{severity.upper()}</span>",
                unsafe_allow_html=True,
            )
    with res_col2:
        st.metric(get_text("confidence", lang), f"{confidence:.1f}%")
        if confidence >= 80:
            st.success(get_text("high_confidence", lang))
        elif confidence >= 60:
            st.info(get_text("medium_confidence", lang))
        else:
            st.warning(get_text("low_confidence", lang))
        if use_ref:
            st.caption(f"🔬 {get_text('refined_prediction', lang)}")

    st.progress(confidence / 100.0)

    # -- Disease info tabs --
    if pred_class in DISEASE_INFO:
        info = DISEASE_INFO[pred_class]
        st.divider()

        tab_what, tab_spread, tab_cause, tab_cure, tab_prev = st.tabs([
            f"❓ {get_text('what_is_this', lang)}",
            f"🌀 {get_text('how_spreads', lang)}",
            f"🔎 {get_text('causes_label', lang)}",
            f"💊 {get_text('treatment_label', lang)}",
            f"🛡️ {get_text('prevention_label', lang)}",
        ])
        with tab_what:
            st.write(info["what_is"].get(lang, info["what_is"]["en"]))
        with tab_spread:
            st.write(info["spread"].get(lang, info["spread"]["en"]))
        with tab_cause:
            st.write(info["causes"].get(lang, info["causes"]["en"]))
        with tab_cure:
            st.write(info["cure"].get(lang, info["cure"]["en"]))
        with tab_prev:
            st.write(info["prevention"].get(lang, info["prevention"]["en"]))

    # -- Attention heatmap --
    st.divider()
    if st.checkbox(get_text("show_heatmap", lang)):
        with st.spinner("Generating attention map..."):
            hm = generate_heatmap(preprocessed)
            overlay = overlay_heatmap(preprocessed, hm)
        st.subheader(get_text("attention_map", lang))
        st.image(overlay, use_container_width=True)
        st.caption("Red/Yellow = high attention  |  Blue = low attention")

    # -- All-class probability chart --
    with st.expander(get_text("all_probabilities", lang)):
        fig, ax = plt.subplots(figsize=(10, 5))
        labels = [get_disease_name(c, "en") for c in CLASS_NAMES]
        colours = ["#22c55e" if c == pred_class else "#94a3b8" for c in CLASS_NAMES]
        y = np.arange(len(CLASS_NAMES))
        ax.barh(y, display_probs, color=colours, height=0.55)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Confidence (%)")
        ax.set_xlim(0, max(float(display_probs.max()) + 15, 100))
        for i, v in enumerate(display_probs):
            ax.text(float(v) + 1, i, f"{v:.1f}%", va="center", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Translated legend for non-English
        if lang != "en":
            st.markdown("**🔤 Disease Names:**")
            for c in CLASS_NAMES:
                st.caption(f"• {get_disease_name(c, lang)}  ({get_disease_name(c, 'en')})")


# -----------------------------------------------------------------------
# ABOUT PAGE
# -----------------------------------------------------------------------
def show_about():
    lang = st.session_state.get("lang", "en")

    st.title(get_text("about_title", lang))

    # -- Model metrics --
    st.header(get_text("ai_performance", lang))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test Accuracy", "93.7%", help="With TTA (5 crops x 2 flips)")
    m2.metric("Refined Accuracy", "94.2%", help="Post-hoc: T-scaling + thresholds + TTA")
    m3.metric("Model Size (TFLite)", "9.05 MB")
    m4.metric("Latency", "27 ms", help="Single image, Tesla T4 GPU")

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Macro F1", "93.4%")
    m6.metric("FPS", "37.6")
    m7.metric("Parameters", "8.1M", help="3.3M trainable")
    m8.metric("Test Images", "1,599")

    st.divider()

    st.subheader("Architecture")
    st.markdown("""
| Component | Detail |
|---|---|
| **Spatial backbone** | EfficientNetV2-B0 (5.9M params) |
| **Colour backbone** | MobileNetV2 alpha=0.35 (0.74M) - 8-ch input |
| **Texture backbone** | MobileNetV2 alpha=0.35 (0.74M) - 11-ch input |
| **Attention** | ECA (k=5) on each branch after GAP |
| **Fusion** | Concat (128x3=384-d) - BN - Dense(256) - Dropout - Dense(7, softmax) |
| **Training** | 2-phase: 25 frozen + 55 fine-tune, Focal Loss (gamma=2), AdamW, CosineAnnealing |
| **Augmentation** | MixUp (alpha=0.3) + CutMix (alpha=1.0), triplet-safe |
| **Colour features** | H, S, CLAHE-L*, a*, b*, ExG, a*/b*, R/G |
| **Texture features** | Gray, Canny, Gabor x2, LocalStd, MorphGrad, CLAHEgray, LesionDensity, LBP x2, HueEdge |
    """)

    # -- Confusion matrix --
    cm_path = MODEL_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.divider()
        st.subheader("Confusion Matrix")
        st.image(str(cm_path), use_container_width=True)

    # -- Emergency helplines --
    st.divider()
    st.header(get_text("emergency_helplines", lang))

    helplines = [
        ("Tea Board of India", "1800-345-3644", "subsidies_smartcards"),
        ("Kisan Call Center", "1551", "agri_advice"),
        ("Tocklai Tea Research", "0376-2360974", "disease_emergency"),
        ("Assam Agri University", "0376-2340001", "plant_protection"),
        ("Small Tea Growers", "+91-94350-32115", "local_support"),
    ]
    cols = st.columns(len(helplines))
    for col, (name, number, svc) in zip(cols, helplines):
        with col:
            st.markdown(f"**{name}**")
            st.caption(get_text(svc, lang))
            st.code(number, language=None)

    # -- Banking --
    st.divider()
    st.header(get_text("banking_support", lang))
    st.markdown("""
| Organisation | Focus |
|---|---|
| **NABARD** | Agricultural Loans |
| **SBI** | Kisan Credit Card |
| **ACARDB** | Small Farmer Focus |
    """)

    # -- Model tensor info --
    st.divider()
    st.header(get_text("model_information", lang))
    try:
        interp, mtype = load_tflite_model()
        if interp:
            with st.expander(get_text("input_tensors", lang)):
                for d in interp.get_input_details():
                    st.code(f"{d['name']}  shape={d['shape']}  dtype={d['dtype']}", language=None)
            with st.expander(get_text("output_tensors", lang)):
                for d in interp.get_output_details():
                    st.code(f"{d['name']}  shape={d['shape']}  dtype={d['dtype']}", language=None)
        else:
            st.warning("Model not loaded - tensor info unavailable.")
    except Exception as e:
        st.error(f"Model inspection failed: {e}")


# -----------------------------------------------------------------------
if __name__ == "__main__":
    main()
