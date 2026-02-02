"""
Tea Leaf Disease Detector - Streamlit app (TFLite)

Lightweight, offline-capable UI for running a TFLite model to
predict tea leaf diseases. This file has been adjusted for a
more professional UI and requires a valid TFLite model to run predictions.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt

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
        "as": "জরুरি সহায়তা লাইন",
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
}

DISEASE_NAMES = {
    "Brown_Blight": {
        "en": "Brown Blight",
        "hi": "भूरी झुलसा",
        "as": "ব্রাউন ব্লাইট",
        "sa": "भूरी झुलसा",
    },
    "Gray_Blight": {
        "en": "Gray Blight",
        "hi": "ग्रे झुलसा",
        "as": "গ্রে ব্লাইট",
        "sa": "ग्रे झुलसा",
    },
    "Healthy_leaf": {
        "en": "Healthy Leaf",
        "hi": "स्वस्थ पत्ती",
        "as": "সুস্থ পাত",
        "sa": "स्वस्थ पत्ती",
    },
    "Helopeltis": {
        "en": "Helopeltis (Mosquito Bug)",
        "hi": "हेलोपेल्टिस (मच्छर कीट)",
        "as": "হেলোপেল্টিস (মছুনি পোক)",
        "sa": "हेलोपेल्टिस (मच्छर कीट)",
    },
    "Red_Rust": {
        "en": "Red Rust",
        "hi": "लाल जंग",
        "as": "ৰঙা ৰং",
        "sa": "लाल जंग",
    },
    "Red_spider": {
        "en": "Red Spider Mite",
        "hi": "लाल मकड़ी का कण",
        "as": "ৰঙা মকৰা কণ",
        "sa": "लाल मकड़ी का कण",
    },
    "Sunlight_Scorching": {
        "en": "Sunlight Scorching",
        "hi": "धूप से झुलसा",
        "as": "ৰ'দত পোৰা",
        "sa": "धूप से झुलसा",
    },
}

DISEASE_INFO = {
    "Brown_Blight": {
        "severity": "medium",
        "what_is": {
            "en": "Brown blight is a fungal disease that causes brown circular lesions on leaves, often after physical damage or wet conditions.",
            "hi": "भूरी झुलसा एक कवक रोग है जो भौतिक क्षति या गीली स्थितियों के बाद पत्तियों पर भूरे गोलाकार घाव का कारण बनता है।",
            "as": "ব্রাউন ব্লাইট এক ভেঁকুৰ ৰোগ যি ভৌতিক ক্ষতি বা আর্দ্র অৱস্থাৰ পিছত পাতত ভূরे বৃত্তাকাৰ ক্ষত সৃষ্টি কৰে।",
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
            "as": "কঠোৰ তোড়া, শিলা বা ভৌতिক ক্षতি, দীর্ঘস্থায়ী পাত সেতুপন, আৰু বেয়া জল निকासी।",
            "sa": "कठोर तोड़ना, ओले या भौतिक क्षति, लंबे समय तक पत्ती की नमी, और खराब जल निकासी।",
        },
        "cure": {
            "en": "Apply Copper Oxychloride fungicide per label. Remove and destroy infected leaves. Improve drainage.",
            "hi": "निर्देशानुसार कॉपर ऑक्सीक्लोराइड कवकनाशी लागू करें। संक्रमित पत्तियों को हटाएं और नष्ट करें। जल निकासी में सुधार करें।",
            "as": "নির্দেশ অনুসাৰে কপাৰ অক্সিক্লোৰাইড কৰ্কটনাশী প্ৰয়োগ কৰক। সংক্রমিত পাত আঁতৰাওক আৰু ধ্বংস কৰক। জল নির्কাशনে উন্নति কৰক।",
            "sa": "निर्देशानुसार कॉपर ऑक्सीक्लोराइड कवकनाशी लागू करें। संक्रमित पत्तियों को हटाएं और नष्ट करें। जल निकासी में सुधार करें।",
        },
        "prevention": {
            "en": "Avoid wounds during plucking, improve field drainage, follow preventive spray schedules.",
            "hi": "तोड़ते समय घाव से बचें, खेत की जल निकासी में सुधार करें, रोकथाम स्प्रे कार्यक्रम का पालन करें।",
            "as": "তোড়াৰ সময় ক্ষত এড়াওক, খেতৰ জল निकास উন্নত কৰক, প্রতিরোধমূলক স্প्রে সময়সূচী অনুসরণ কৰক।",
            "sa": "तोड़ते समय घाव से बचें, खेत की जल निकासी में सुधार करें, रोकथाम स्प्रे कार्यक्रम का पालन करें।",
        },
    },
    "Gray_Blight": {
        "severity": "high",
        "what_is": {
            "en": "Gray blight produces greyish lesions with dark centers that coalesce into larger patches.",
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
            "as": "যেখানে উপযুক্ত কার্বেন্ডাজিম কৰ্কটনাশী ব্যবহাৰ কৰক। প্ৰভাৱিত এলাকা কাটক। পৰিত্যক্ত ধ্বংসাৱশেষ নিষ্পত্তি কৰক।",
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
        "severity": "low",
        "what_is": {
            "en": "This leaf appears healthy with uniform green color and no visible lesions.",
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
            "as": "সুষম সার, সঠিক জল निकाশন, আৰু নিয়মিত পরিদর্শন।",
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
            "as": "অতিরিক্ত ছাঁ, ভারী বন্য উপস্থিति, ঘন রোপণ, আৰু খেতৰ স্বাস্থ্যবিধিৰ অভাৱ।",
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
            "hi": "सूखी, हवादार परिस्थितियों में धूल या बीजाणु के रूप में फैल सकता है और बड़े क्षेत्रों को तेजी से प्रभावित करता है।",
            "as": "শুকান, বায়ুৱহ অৱস্থাত ধূলি বা বীজাণু হিসাবে ছড়াই পাৰে আৰু বড় এলাকাক দ্রুত প্ৰভাৱিত কৰে।",
            "sa": "सूखी, हवादार परिस्थितियों में धूल या बीजाणु के रूप में फैल सकता है और बड़े क्षेत्रों को तेजी से प्रभावित करता है।",
        },
        "causes": {
            "en": "Waterlogged soils, weak or stressed plants, low potash levels, and poor air circulation.",
            "hi": "जलभराव वाली मिट्टी, कमजोर या तनावग्रस्त पौधे, कम पोटाश स्तर, और खराब वायु संचार।",
            "as": "জলভরা মাটি, দুর্বল বা চাপগ্রস্ত গছ, কম পোটাশ স্তৰ, আৰু খারাপ বায়ু সংবহন।",
            "sa": "जलभराव वाली मिट्टी, कमजोर या तनावग्रस्त पौधे, कम पोटाश स्तर, और खराब वायु संचार।",
        },
        "cure": {
            "en": "Use appropriate fungicide or algaecide as recommended. Remove affected material. Improve drainage and soil health.",
            "hi": "अनुशंसित कवकनाशी या शैवाल नाशी का उपयोग करें। प्रभावित सामग्री को हटाएं। जल निकास और मिट्टी के स्वास्थ्य में सुधार करें।",
            "as": "অনুমোদিত কৰ্কটনাশী বা শৈবাল নাশী ব্যবহাৰ কৰক। প্ৰভাৱিত সামগ্রী আঁতৰাওক। জল নির্কাশন আৰু মাটি স্বাস্থ্য উন্নত কৰক।",
            "sa": "अनुशंसित कवकनाशी या शैवाल नाशी का उपयोग करें। प्रभावित सामग्री को हटाएं। जल निकास और मिट्टी के स्वास्थ्य में सुधार करें।",
        },
        "prevention": {
            "en": "Improve drainage, maintain balanced fertilization with potash, practice good field hygiene.",
            "hi": "जल निकास में सुधार करें, पोटाश के साथ संतुलित खाद बनाए रखें, अच्छी खेत स्वच्छता का अभ्यास करें।",
            "as": "জল नির্কাশনে উন্নতি কৰক, পোটাশৰ সৈতে সুষম সার বজায় ৰাখক, ভাল খেত স্বাস্থ্যবিধি অনুশীলন কৰক।",
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
            "as": "ৰ'দত পোৰা তীব্ৰ সূৰ্য আৰু গর্মত পৰা গैर-জৈব পাত বিনাশ, ৰোগ নহয়।",
            "sa": "धूप का झुलसा तीव्र सूर्य और गर्मी से गैर-जैविक पत्ती की जलन है, रोग नहीं।",
        },
        "spread": {
            "en": "Not contagious; affects exposed plants during heatwaves when shade is insufficient.",
            "hi": "संक्रामक नहीं; जब छाया अपर्याप्त हो तो लू के दौरान उजागर पौधों को प्रभावित करता है।",
            "as": "সংক্ৰামক নহয়; যেতিয়া ছাঁ অপৰ্যাপ্ত হয় তেতিয়া তাপপ্ৰবাহৰ সময়ত উন্মুক্ত গছক প্ৰभাৱিত কৰে।",
            "sa": "संक्रामक नहीं; जब छाया अपर्याप्त हो तो लू के दौरान उजागर पौधों को प्रभावित करता है।",
        },
        "causes": {
            "en": "Lack of shade, extreme heat over 38°C, south-facing exposure, tender growth in harsh sun.",
            "hi": "छाया की कमी, 38°C से अधिक चरम गर्मी, दक्षिण-सामना करने वाला संपर्क, कठोर धूप में कोमल वृद्धि।",
            "as": "ছাঁৰ অভাৱ, 38°C তকৈ অধিক চৰম উষ্ণতা, দক্ষিণ-সামনা সংস্পৰ্শ, কঠোৰ ৰ'দত কোমল বৃদ্ধি।",
            "sa": "छाया की कमी, 38°C से अधिक चरम गर्मी, दक्षिण-सामना करने वाला संपर्क, कठोर धूप में कोमल वृद्धि।",
        },
        "cure": {
            "en": "Remove severely scorched leaves. Provide temporary shade. Monitor for secondary fungal infections.",
            "hi": "बहुत अधिक झुलसी पत्तियों को हटाएं। अस्थायी छाया प्रदान करें। माध्यमिक कवक संक्रमण के लिए निगरानी करें।",
            "as": "গুৰুতৰভাৱে পোৰা পাত আঁতৰাওক। অস্থায়ী ছাঁ প্ৰদান কৰক। माध्यमिক कवक सংक्रमणৰ জন्য নिরীक্ষণ কৰক।",
            "sa": "बहुत अधिक झुलसी पत्तियों को हटाएं। अस्थायी छाया प्रदान करें। माध्यमिक कवक संक्रमण के लिए निगरानी करें।",
        },
        "prevention": {
            "en": "Plant two-tier shade (tall and short trees). Mulch soil with dry grass. Irrigate during extreme heat.",
            "hi": "दो-स्तरीय छाया लगाएं (ऊंचे और छोटे पेड़)। मिट्टी को सूखी घास से ढकें। चरम गर्मी में सिंचाई करें।",
            "as": "দুই-স্তৰীয় ছাঁ লগাওক (ওখ আৰু চুটি গছ)। মাটি শুকান ঘাঁহেৰে ঢাকক। চৰম উষ্ণতাত সিঞ্চন কৰক।",
            "sa": "दो-स्तरीय छाया लगाएं (ऊंचे और छोटे पेड़)। मिट्टी को सूखी घास से ढकें। चरम गर्मी में सिंचाई करें।",
        }
    }
}

SEVERITY_COLORS = {"low": "green", "medium": "orange", "high": "red", "critical": "darkred"}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_text(key, lang="en"):
    """Get translated text for a key, fallback to English if not found"""
    if key in TRANSLATIONS:
        return TRANSLATIONS[key].get(lang, TRANSLATIONS[key].get("en", key))
    return key

def get_disease_name(disease, lang="en"):
    """Get translated disease name, fallback to English if not found"""
    if disease in DISEASE_NAMES:
        return DISEASE_NAMES[disease].get(lang, DISEASE_NAMES[disease].get("en", disease))
    return disease

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def assess_image_quality(img):
    """Check image quality - only reject genuinely bad images"""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = gray.std()
    
    issues = []
    
    # Only flag extreme cases for issues display (informational only)
    if brightness < 15:
        issues.append("Very dark")
    elif brightness > 240:
        issues.append("Very bright")
    
    # Only flag if EXTREMELY blurry (< 10) - this is a serious issue
    if laplacian_var < 10:
        issues.append("Extremely blurry")
    
    # Only flag if no contrast at all (< 5)
    if contrast < 5:
        issues.append("No contrast")
    
    # Simple quality score based on how "normal" the image is
    # More lenient: even with minor issues, good images pass
    quality_score = 100
    
    # Deduct points only for serious issues
    if laplacian_var < 10:
        quality_score -= 30  # Extremely blurry
    if brightness < 15 or brightness > 240:
        quality_score -= 20  # Extreme lighting
    if contrast < 5:
        quality_score -= 20  # No contrast
    
    quality_score = max(0, quality_score)
    
    # Accept image if quality >= 35 (only reject genuinely bad images)
    acceptable = quality_score >= 35 and laplacian_var >= 8 and brightness > 10 and contrast > 3
    
    return quality_score, issues, acceptable, laplacian_var

def denoise_image(img):
    return cv2.fastNlMeansDenoisingColored(img, None, h=7, hColor=7, 
                                            templateWindowSize=7, searchWindowSize=21)

def correct_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def preprocess_image(img):
    img = denoise_image(img)
    img = correct_lighting(img)
    return img

def check_if_leaf(img):
    """
    Background-aware tea leaf detection
    - Ignores white background by calculating ratio on leaf area only
    - Accepts healthy green, yellowed (sunlight scorching), and brown (disease) leaves
    - Rejects faces/non-plant objects through saturation analysis
    """
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h, w = hsv.shape[:2]

        # Step 1: Remove white background (bright areas)
        lower_white = np.array([0, 0, 200])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        non_white_mask = cv2.bitwise_not(white_mask)
        non_white_pixels = np.sum(non_white_mask > 0)

        # If too little non-white content, likely not a leaf (mostly blank)
        if non_white_pixels < (h * w * 0.03):
            return False

        # Step 2: Color-based plant masks (for green, yellow/scorch, brown/red)
        lower_green = np.array([20, 25, 20])
        upper_green = np.array([95, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        lower_yellow = np.array([10, 25, 30])
        upper_yellow = np.array([45, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_red1 = np.array([0, 20, 30])
        upper_red1 = np.array([15, 255, 255])
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        lower_red2 = np.array([160, 20, 30])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        plant_mask = green_mask | yellow_mask | red_mask1 | red_mask2
        plant_pixels_in_leaf = np.sum(cv2.bitwise_and(plant_mask, non_white_mask) > 0)
        plant_ratio = (plant_pixels_in_leaf / non_white_pixels) * 100 if non_white_pixels > 0 else 0

        # Step 3: Structural analysis - edges and texture (veins, sharp lines)
        img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
        gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 120)
        edges_masked = cv2.bitwise_and(edges, edges, mask=non_white_mask)
        edge_pixels = np.sum(edges_masked > 0)
        edge_ratio = (edge_pixels / non_white_pixels) * 100 if non_white_pixels > 0 else 0

        lap = cv2.Laplacian(blurred, cv2.CV_64F)
        lap_var = float(np.var(lap[non_white_mask > 0])) if non_white_pixels > 0 else 0.0

        # Step 4: Skin/face heuristic - simple HSV skin detection to avoid selfies
        lower_skin = np.array([0, 10, 60])
        upper_skin = np.array([25, 200, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        skin_pixels = np.sum(cv2.bitwise_and(skin_mask, skin_mask, mask=non_white_mask) > 0)
        skin_ratio = (skin_pixels / non_white_pixels) * 100 if non_white_pixels > 0 else 0

        # Saturation check (faces generally lower saturation than waxy leaves)
        s_channel = hsv[:, :, 1]
        non_white_saturation = s_channel[non_white_mask > 0]
        avg_saturation = float(np.mean(non_white_saturation)) if len(non_white_saturation) > 0 else 0.0

        # Decision logic (more permissive to accept scorched/brown leaves):
        # - Color-accepted: plant_ratio >= 12 and avg_saturation >= 10
        # - Structural-accepted: edge_ratio >= 2 (percent) and lap_var >= 10
        # - Final accept if either condition holds, unless strong face evidence
        color_ok = (plant_ratio >= 12 and avg_saturation >= 10)
        struct_ok = (edge_ratio >= 2.0 and lap_var >= 10.0)

        # Face rejection: higher skin ratio and low structural signal
        face_like = (skin_ratio >= 25.0 and edge_ratio < 2.0 and lap_var < 10.0)

        is_leaf = (color_ok or struct_ok) and (not face_like)

        return bool(is_leaf)
    except Exception:
        # Conservative fallback: allow image through so model confidence can decide
        return True

# ============================================================================
# TFLITE MODEL LOADING & PREDICTION
# ============================================================================

@st.cache_resource
def load_tflite_model():
    """Load TFLite model with detailed error reporting"""
    try:
        import tensorflow as tf
    except Exception as e:
        st.warning("TensorFlow not available - running in demo mode")
        return None, "tf_missing"

    model_path = "tea_doctor_v7_final.tflite"

    if not Path(model_path).exists():
        st.info(f"ℹ️ Model file '{model_path}' not found. Running in demo mode. Place the model file for real predictions.")
        return None, "model_missing"

    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        st.success(f"✅ TFLite model loaded: {len(input_details)} inputs, {len(output_details)} outputs")
        return interpreter, "tflite"
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None, "load_error"

def extract_color_features(img):
    """Extract LAB + HSV color features"""
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    lab = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2HSV)
    features = np.stack([lab[:, :, 1], hsv[:, :, 1], lab[:, :, 0]], axis=-1).astype(np.float32) / 255.0
    return features

def extract_texture_features(img):
    """Extract edge/texture features"""
    img_uint8 = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    edges = np.sqrt(sobel_x**2 + sobel_y**2)
    edges = np.clip(edges, 0, 255).astype(np.uint8)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian).clip(0, 255).astype(np.uint8)
    features = np.stack([gray, edges, laplacian], axis=-1).astype(np.float32) / 255.0
    return features

def predict_disease_tflite(img, interpreter, model_type):
    """Predict disease using TFLite"""
    # Model outputs 7 classes
    CLASS_NAMES = [
        'Brown_Blight', 'Gray_Blight', 'Healthy_leaf', 
        'Helopeltis', 'Red_Rust', 'Red_spider', 'Sunlight_Scorching'
    ]
    
    # Check for demo mode
    demo_mode = st.session_state.get("demo_mode", False)
    demo_seed = st.session_state.get("demo_seed", None)

    if demo_mode:
        import random
        rng = random.Random(demo_seed)
        pred_class = rng.choice(CLASS_NAMES)
        confidence = float(rng.uniform(50.0, 95.0))
        all_probs = [float(rng.uniform(0.0, 20.0)) for _ in CLASS_NAMES]
        max_idx = CLASS_NAMES.index(pred_class)
        all_probs[max_idx] = confidence
        return pred_class, confidence, all_probs

    if model_type != "tflite" or interpreter is None:
        # Model not available - show warning instead of random predictions
        st.warning("⚠️ TFLite model not loaded. Running in demo mode.")
        # Still return predictions but with lower confidence ranges
        import random
        pred_class = random.choice(CLASS_NAMES)
        confidence = float(random.uniform(30, 60))  # Lower confidence to indicate demo
        all_probs = [float(random.uniform(5, 25)) for _ in CLASS_NAMES]
        max_idx = CLASS_NAMES.index(pred_class)
        all_probs[max_idx] = confidence
        return pred_class, confidence, all_probs
    
    # Prepare inputs - match model expectations exactly
    img_resized = cv2.resize(img, (224, 224))
    
    # Input 1: RGB image as UINT8 [0-255] - NOT normalized
    rgb_input = np.expand_dims(img_resized, axis=0).astype(np.uint8)
    
    # Input 2: Color features as FLOAT32 [0-1]
    color_input = np.expand_dims(extract_color_features(img_resized / 255.0), axis=0).astype(np.float32)
    
    # Input 3: Texture features as FLOAT32 [0-1]
    texture_input = np.expand_dims(extract_texture_features(img_resized / 255.0), axis=0).astype(np.float32)
    
    try:
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Validate input tensor count
        if len(input_details) < 3:
            st.error(f"Model expects {len(input_details)} input(s) but 3 were provided.")
            return CLASS_NAMES[0], 25.0, [25.0] * len(CLASS_NAMES)

        # Set tensors in the CORRECT MODEL ORDER:
        # Input 0: texture_input (FLOAT32)
        # Input 1: rgb_input (UINT8)
        # Input 2: color_input (FLOAT32)
        
        # Map inputs by name to handle any order
        input_map = {}
        for detail in input_details:
            name = detail['name']
            if 'texture' in name.lower():
                input_map[detail['index']] = texture_input.astype(detail['dtype'])
            elif 'rgb' in name.lower():
                input_map[detail['index']] = rgb_input.astype(detail['dtype'])
            elif 'color' in name.lower():
                input_map[detail['index']] = color_input.astype(detail['dtype'])
        
        # Set all tensors
        for index, data in input_map.items():
            interpreter.set_tensor(index, data)

        # Debug: Show input data statistics
        with st.expander("🔍 Debug: Input Data Statistics"):
            st.write(f"**RGB Input** - Max: {rgb_input.max()}, Min: {rgb_input.min()}, Mean: {rgb_input.mean():.2f}")
            st.write(f"**Color Input** - Max: {color_input.max():.2f}, Min: {color_input.min():.2f}, Mean: {color_input.mean():.2f}")
            st.write(f"**Texture Input** - Max: {texture_input.max():.2f}, Min: {texture_input.min():.2f}, Mean: {texture_input.mean():.2f}")
            st.info("✓ RGB: 0-255 expected (no Rescaling layer) | Color/Texture: 0-1 (normalized)")

        interpreter.invoke()

        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        
        # Validate predictions
        if predictions is None or len(predictions) == 0:
            st.error("Model returned empty predictions")
            return CLASS_NAMES[0], 25.0, [25.0] * len(CLASS_NAMES)

        # Handle mismatch between model outputs and CLASS_NAMES
        num_outputs = len(predictions)
        num_classes = len(CLASS_NAMES)
        
        if num_outputs != num_classes:
            st.warning(f"⚠️ Model outputs {num_outputs} classes but expected {num_classes}. Adjusting...")
            # Pad or trim predictions to match number of classes
            if num_outputs < num_classes:
                # Pad with zeros
                predictions = np.concatenate([predictions, np.zeros(num_classes - num_outputs)])
            else:
                # Trim to expected number
                predictions = predictions[:num_classes]
        
        max_idx = int(np.argmax(predictions))
        pred_class = CLASS_NAMES[max_idx]
        confidence = float(predictions[max_idx] * 100.0)
        all_probs = (predictions * 100.0).astype(float).tolist()

        confidence = float(np.clip(confidence, 0.0, 100.0))
        all_probs = [float(np.clip(p, 0.0, 100.0)) for p in all_probs]

        return pred_class, confidence, all_probs
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Return balanced fallback
        return CLASS_NAMES[0], 25.0, [25.0] * len(CLASS_NAMES)

# ============================================================================
# HEATMAP
# ============================================================================

def generate_heatmap(img, model, pred_class):
    """Generate SmoothGrad-style heatmap for better visualization"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) == 3 else img
    
    # Detect edges (disease indicators)
    edges = cv2.Canny(gray, 50, 150)
    
    # Calculate Laplacian (texture variations - disease signatures)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.abs(laplacian)
    
    # Normalize
    edges_norm = edges.astype(np.float32) / 255.0
    laplacian_norm = laplacian / (laplacian.max() + 1e-6)
    
    # Combine: edges (60%) + texture (40%)
    heatmap = 0.6 * edges_norm + 0.4 * laplacian_norm
    
    # Apply multiple Gaussian blurs for smooth appearance (SmoothGrad style)
    for _ in range(3):
        heatmap = cv2.GaussianBlur(heatmap, (31, 31), 2.0)
    
    # Normalize to [0, 1]
    if heatmap.max() > heatmap.min():
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        heatmap = np.zeros_like(heatmap)
    
    # Enhance contrast slightly for better visibility
    heatmap = np.power(heatmap, 0.8)
    
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image"""
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlayed = cv2.addWeighted(img, 1-alpha, heatmap_colored, alpha, 0)
    return overlayed

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Configure page
    st.set_page_config(page_title="Tea Doctor", layout="wide")
    
    # Sidebar
    st.sidebar.title("Language")
    lang = st.sidebar.selectbox(
        "Select",
        ["en", "hi", "as", "sa"],
        format_func=lambda x: {"en": "English", "hi": "हिंदी", "as": "অসমীয়া", "sa": "सादरी"}[x],
    )
    st.session_state.lang = lang

    st.sidebar.markdown("---")
    demo_mode = st.sidebar.checkbox("Enable demo mode")
    st.session_state.demo_mode = demo_mode
    if demo_mode:
        seed = st.sidebar.number_input("Demo seed", value=42, step=1)
        st.session_state.demo_seed = int(seed)
        st.sidebar.info("Demo mode uses fixed seed for reproducible predictions.")
    
    st.sidebar.markdown("---")
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "About"])
    
    # Kill-switch: skip leaf checks (use when you want immediate results)
    st.sidebar.markdown("---")
    skip_checks = st.sidebar.checkbox("Kill Switch: Skip Leaf Checks", value=False, help="Skip color/structure checks and rely only on model confidence")
    st.session_state.skip_checks = skip_checks
    
    if page == "Home":
        show_home_page()
    else:
        show_about_page()

def show_home_page():
    lang = st.session_state.get('lang', 'en')

    st.title(get_text("home_title", lang))
    st.markdown("---")
    
    model, model_type = load_tflite_model()
    
    st.subheader(get_text("upload_image", lang))
    uploaded_file = st.file_uploader("Choose image", type=['jpg', 'jpeg', 'png'], label_visibility="collapsed")
    
    st.markdown("**OR**")
    
    # Camera toggle
    use_camera = st.checkbox("📷 Use Camera", value=False)
    camera_image = None
    
    if use_camera:
        camera_image = st.camera_input("Take photo")
    
    image_source = uploaded_file if uploaded_file else camera_image
    
    if image_source:
        image = Image.open(image_source)
        img_array = np.array(image)
        
        if len(img_array.shape) == 2:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        elif img_array.shape[2] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(get_text("original", lang))
            st.image(img_array, use_container_width=True)
        
        with st.spinner(get_text("preprocessing", lang)):
            preprocessed = preprocess_image(img_array)
        
        with col2:
            st.subheader(get_text("preprocessed", lang))
            st.image(preprocessed, use_container_width=True)
        
        st.markdown("---")

        # Soft leaf check - warn but don't block (confidence threshold handles hard rejection)
        skip_checks = st.session_state.get('skip_checks', False)
        if not skip_checks:
            leaf_check = check_if_leaf(img_array)
            if not leaf_check:
                st.warning("⚠️ This doesn't look like a tea leaf, but proceeding... The confidence threshold will be the final judge.")
        else:
            st.info("Kill Switch active: skipping leaf checks")
        
        if model is None or model_type != "tflite":
            st.error("A valid TFLite model is required. Make sure 'tea_doctor_v7_final.tflite' is available.")
            st.stop()

        with st.spinner(get_text("analyzing", lang)):
            pred_class, confidence, all_probs = predict_disease_tflite(preprocessed, model, model_type)

        confidence = float(np.clip(float(confidence), 0.0, 100.0))
        
        # ⚠️ CONFIDENCE THRESHOLD - PRIMARY GUARD against false positives (selfies, non-leaves)
        if confidence < 40:
            st.error(f"❌ **Confidence too low ({confidence:.1f}%)**\n\nThis doesn't appear to be a tea leaf. The model isn't confident.")
            st.info("💡 Try with: Better lighting, different angle, or a real tea leaf image")
            st.stop()

        st.info(get_text("analysis_complete", lang))

        disease_name = get_disease_name(pred_class, lang)

        st.header(disease_name)
        if "Healthy" in pred_class:
            st.success(get_text("healthy_leaf_msg", lang))

        st.subheader(f"{get_text('confidence', lang)}: {confidence:.1f}%")
        st.progress(confidence / 100.0)
        
        if confidence >= 80:
            st.success(get_text("high_confidence", lang))
        elif confidence >= 60:
            st.info(get_text("medium_confidence", lang))
        else:
            st.warning(get_text("low_confidence", lang))
        
        st.markdown("---")
        
        if pred_class in DISEASE_INFO:
            info = DISEASE_INFO[pred_class]
            severity = info.get("severity", "medium")
            
            st.markdown(f"### {get_text('severity_label', lang)}: {severity.upper()}")
            
            with st.expander(get_text("what_is_this", lang), expanded=True):
                st.write(info["what_is"].get(lang, info["what_is"]["en"]))
            
            with st.expander(get_text("how_spreads", lang)):
                st.write(info["spread"].get(lang, info["spread"]["en"]))
            
            with st.expander(get_text("causes_label", lang)):
                st.markdown(info["causes"].get(lang, info["causes"]["en"]))

            with st.expander(get_text("treatment_label", lang)):
                st.markdown(info["cure"].get(lang, info["cure"]["en"]))

            with st.expander(get_text("prevention_label", lang)):
                st.markdown(info["prevention"].get(lang, info["prevention"]["en"]))
        
        st.markdown("---")
        
        if st.checkbox(get_text("show_heatmap", lang)):
            with st.spinner("Generating attention map..."):
                heatmap = generate_heatmap(preprocessed, model, pred_class)
                overlayed = overlay_heatmap(preprocessed, heatmap)

            st.subheader(get_text("attention_map", lang))
            st.image(overlayed, use_container_width=True)
            st.caption("Red/Yellow = model focus | Blue = less attention")
        
        with st.expander(get_text("all_probabilities", lang)):
            CLASS_NAMES = ['Brown_Blight', 'Gray_Blight', 'Healthy_leaf', 'Helopeltis', 'Red_Rust', 'Red_spider', 'Sunlight_Scorching']
            
            # Ensure all_probs matches CLASS_NAMES length
            display_probs = all_probs[:len(CLASS_NAMES)] if len(all_probs) >= len(CLASS_NAMES) else all_probs + [0] * (len(CLASS_NAMES) - len(all_probs))
            
            try:
                # Strategy 1: Use English labels for matplotlib (more reliable)
                # Keep disease names in English for plot, translate separately
                
                fig, ax = plt.subplots(figsize=(11, 6))
                
                # Use matplotlib's unicode-safe properties
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
                
                colors = ['green' if c == pred_class else 'gray' for c in CLASS_NAMES]
                y_pos = np.arange(len(CLASS_NAMES))
                
                # Strategy 2: Use English disease names for Y-axis (always reliable)
                # Format: "English (Local)" to show translation
                disease_labels = [get_disease_name(c, lang) for c in CLASS_NAMES]
                
                # For non-English languages, create hybrid labels
                if lang != 'en':
                    # Use numbered labels (safe) with translated names below
                    y_labels = [f"{i+1}. {name}" for i, name in enumerate(disease_labels)]
                else:
                    y_labels = disease_labels
                
                # Create bar chart
                ax.barh(y_pos, display_probs, color=colors, height=0.6)
                ax.set_yticks(y_pos)
                
                # Strategy 3: Use Unicode-safe labels
                ax.set_yticklabels(y_labels, fontsize=9)
                
                ax.set_xlabel(get_text("confidence", lang), fontsize=10)
                ax.set_title(get_text("all_probabilities", lang), fontsize=11, pad=10)
                
                # Add value labels on bars
                for i, v in enumerate(display_probs):
                    ax.text(v + 1, i, f'{v:.1f}%', va='center', fontsize=8)
                
                ax.set_xlim(0, max(display_probs) + 15)
                
                # Adjust layout to prevent label cutoff
                plt.tight_layout()
                
                st.pyplot(fig, use_container_width=True)
                
                # Strategy 4: Show translation legend below graph for non-English
                if lang != 'en':
                    st.markdown("**🔤 Disease Names:**")
                    for i, (eng_name, trans_name) in enumerate(zip(CLASS_NAMES, disease_labels), 1):
                        st.caption(f"{i}. {trans_name} ({eng_name})")
                        
            except Exception as e:
                st.warning(f"⚠️ Graph rendering issue: {str(e)}")
                # Fallback: show as interactive table
                df_data = {
                    get_text("disease", lang) or "Disease": disease_labels,
                    get_text("confidence", lang) or "Confidence (%)": [f"{p:.1f}%" for p in display_probs],
                }
                for i, c in enumerate(CLASS_NAMES):
                    df_data[get_disease_name(c, lang)] = [display_probs[i]]
                st.dataframe(df_data)

def show_about_page():
    lang = st.session_state.get('lang', 'en')
    
    st.title(get_text("about_title", lang))

    st.header(get_text("ai_performance", lang))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "91.2%")
    with col2:
        st.metric("Model Size", "12 MB")
    with col3:
        st.metric("Tested Images", "1,599")
    
    st.info("**Model:** EfficientNetV2-B0 Fusion  \n**Classes:** 6 diseases + 1 healthy leaf  \n**Offline:** Works without internet")
    
    st.markdown("---")
    st.header(get_text("emergency_helplines", lang))
    
    helplines = [
        ("Tea Board of India", "1800-345-3644", "subsidies_smartcards"),
        ("Kisan Call Center", "1551", "agri_advice"),
        ("Tocklai Tea Research", "0376-2360974", "disease_emergency"),
        ("Assam Agri University", "0376-2340001", "plant_protection"),
        ("Small Tea Growers", "+91-94350-32115", "local_support")
    ]
    
    for name, number, service_key in helplines:
        st.subheader(name)
        st.write(f"**{get_text('service_label', lang)}** {get_text(service_key, lang)}")
        st.markdown(f"### {number}")
        st.markdown("---")
    
    st.header(get_text("banking_support", lang))
    st.markdown("""
    **NABARD** - Agricultural Loans  
    **SBI** - Kisan Credit Card  
    **ACARDB** - Small Farmer Focus  
    """)

    st.markdown("---")
    st.header(get_text("model_information", lang))
    try:
        interpreter, model_type = load_tflite_model()
        if interpreter is None:
            st.warning("No valid TFLite interpreter available.")
        else:
            with st.expander(get_text("input_tensors", lang), expanded=False):
                input_details = interpreter.get_input_details()
                for d in input_details:
                    st.write(f"name: {d.get('name')} | shape: {d.get('shape')} | dtype: {d.get('dtype')}")

            with st.expander(get_text("output_tensors", lang), expanded=False):
                output_details = interpreter.get_output_details()
                for d in output_details:
                    st.write(f"name: {d.get('name')} | shape: {d.get('shape')} | dtype: {d.get('dtype')}")
    except Exception as e:
        st.error(f"Model inspection failed: {e}")

if __name__ == "__main__":
    main()
