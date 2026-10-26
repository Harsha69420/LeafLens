"""
disease_info.py - Disease information, symptoms, and agronomic management guidelines.
LeafLens: Plant Disease Screening Using CNN and Transfer Learning
"""

DISEASE_DATABASE = {
    "Tomato___healthy": {
        "display_name": "Tomato — Healthy Leaf",
        "pathogen_type": "None",
        "description": "The leaf shows vigorous green pigmentation, normal vascular turgor, and no visible necrotic or chlorotic lesions.",
        "symptoms": [
            "Uniform leaf coloration without yellowing or brown spots",
            "Firm structural integrity and normal venation",
            "No mycelial growth, bacterial ooze, or viral mosaics"
        ],
        "preventive_actions": [
            "Maintain balanced N-P-K fertilization and adequate irrigation",
            "Ensure good crop spacing for adequate canopy ventilation",
            "Regularly scout crops for early insect or fungal signs",
            "Use mulch to prevent soil splashing onto bottom leaves"
        ],
        "severity": "None (Healthy)",
        "urgency": "Routine maintenance"
    },
    "Tomato___Early_blight": {
        "display_name": "Tomato — Early Blight (Alternaria solani)",
        "pathogen_type": "Fungal pathogen",
        "description": "Early blight is a common fungal disease caused by Alternaria solani. It primarily affects older foliage first and thrives in warm, humid weather.",
        "symptoms": [
            "Concentric rings ('target-board' or 'bullseye' pattern) on mature leaves",
            "Chlorotic yellow halos surrounding dark brown necrotic lesions",
            "Lower leaves drying, turning brown, and prematurely defoliating"
        ],
        "preventive_actions": [
            "Prune and safely dispose of infected lower leaves; do not compost them",
            "Water at the base using drip irrigation to avoid wet foliage",
            "Apply approved copper-based or chlorothalonil fungicides according to local agricultural extension guidelines",
            "Practice 2-3 year crop rotation away from solanaceous plants"
        ],
        "severity": "Moderate to High",
        "urgency": "Action within 24-48 hours recommended"
    },
    "Tomato___Late_blight": {
        "display_name": "Tomato — Late Blight (Phytophthora infestans)",
        "pathogen_type": "Oomycete pathogen",
        "description": "Late blight is a destructive, water-borne oomycete disease capable of destroying entire crops within days under cool, wet conditions.",
        "symptoms": [
            "Large, irregularly shaped, water-soaked dark greenish-brown lesions",
            "White velvety fungal/oomycete growth on the leaf undersides in high humidity",
            "Rapid browning and collapse of stems, petioles, and fruit rot"
        ],
        "preventive_actions": [
            "Immediately isolate and destroy infected plants to curb airborne spore spread",
            "Avoid overhead irrigation and ensure high airflow between rows",
            "Apply preventive systemic or contact fungicides (e.g., mancozeb, copper) early in cool damp weather",
            "Consult local agronomic extension immediately upon confirmation"
        ],
        "severity": "Critical",
        "urgency": "Immediate intervention required"
    },
    "Tomato___Bacterial_spot": {
        "display_name": "Tomato — Bacterial Spot (Xanthomonas spp.)",
        "pathogen_type": "Bacterial pathogen",
        "description": "Bacterial spot is caused by Xanthomonas species. It is seed-borne and rain-splashed, causing significant defoliation and fruit blemishes in warm, rainy climates.",
        "symptoms": [
            "Small (1-3 mm), dark, angular water-soaked lesions on foliage",
            "Lesions may turn blackish-brown and develop thin yellow halos",
            "Severe infections cause leaves to look scorched and drop prematurely"
        ],
        "preventive_actions": [
            "Use certified disease-free seeds and disease-resistant cultivars",
            "Avoid working in crop fields when foliage is wet to prevent mechanical transmission",
            "Apply copper bactericides combined with mancozeb for increased efficacy",
            "Sanitize garden tools and stakes with 10% bleach solution"
        ],
        "severity": "Moderate to High",
        "urgency": "Action within 48 hours recommended"
    },
    "Tomato___Leaf_Mold": {
        "display_name": "Tomato — Leaf Mold (Passalora fulva)",
        "pathogen_type": "Fungal pathogen",
        "description": "Leaf mold predominantly affects tomatoes grown in high-humidity greenhouse environments or dense outdoor canopies with poor circulation.",
        "symptoms": [
            "Pale greenish-yellow chlorotic spots on upper leaf surfaces with indistinct margins",
            "Dense, olive-green to grayish-brown velvety mold on the underside of corresponding spots",
            "Infected leaves eventually curl, wither, and drop prematurely"
        ],
        "preventive_actions": [
            "Significantly increase greenhouse ventilation and reduce relative humidity below 85%",
            "Increase plant spacing and prune lower sucker branches to boost airflow",
            "Apply protective fungicides (copper or sulfur) if symptoms progress",
            "Select resistant tomato varieties for future plantings"
        ],
        "severity": "Moderate",
        "urgency": "Prompt management within a few days"
    },
    "Tomato___Septoria_leaf_spot": {
        "display_name": "Tomato — Septoria Leaf Spot (Septoria lycopersici)",
        "pathogen_type": "Fungal pathogen",
        "description": "A widespread fungal foliar disease causing numerous small spots on lower leaves.",
        "symptoms": [
            "Numerous circular spots with dark brown margins and gray or tan centers",
            "Tiny black specks (pycnidia fruiting bodies) visible in lesion centers",
            "Progressive defoliation from bottom of the plant upwards"
        ],
        "preventive_actions": [
            "Remove infected bottom foliage and avoid wetting leaves when watering",
            "Apply protective organic or synthetic fungicides",
            "Mulch soil around plants to prevent soilborne splash"
        ],
        "severity": "Moderate",
        "urgency": "Manage within a few days"
    },
    "Tomato___Spider_mites_Two_spotted_spider_mite": {
        "display_name": "Tomato — Two-Spotted Spider Mites (Tetranychus urticae)",
        "pathogen_type": "Arachnid Pest",
        "description": "Microscopic sap-feeding mites that thrive in hot, dry conditions.",
        "symptoms": [
            "Fine yellow stippling or bronzing on leaf surfaces",
            "Delicate silken webbing visible on leaf undersides and branch forks",
            "Leaves turn yellow, dry out, and drop"
        ],
        "preventive_actions": [
            "Spray foliage with insecticidal soap, neem oil, or horticultural oils",
            "Introduce predatory mites (e.g., Phytoseiulus persimilis)",
            "Maintain adequate humidity to discourage mite reproduction"
        ],
        "severity": "Moderate",
        "urgency": "Treat promptly to prevent population explosion"
    },
    "Tomato___Target_Spot": {
        "display_name": "Tomato — Target Spot (Corynespora cassiicola)",
        "pathogen_type": "Fungal pathogen",
        "description": "Fungal disease causing target-like lesions on foliage, stems, and fruit.",
        "symptoms": [
            "Brown spots with subtle concentric rings and light brown centers",
            "Defoliation under humid, warm conditions"
        ],
        "preventive_actions": [
            "Improve air circulation and avoid overhead irrigation",
            "Apply recommended fungicides preventively"
        ],
        "severity": "Moderate",
        "urgency": "Action within a few days"
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display_name": "Tomato — Yellow Leaf Curl Virus (TYLCV)",
        "pathogen_type": "Viral pathogen (Whitefly-transmitted)",
        "description": "A devastating viral disease transmitted by the silverleaf whitefly (Bemisia tabaci).",
        "symptoms": [
            "Severe upward curling and cupping of leaflets",
            "Pronounced interveinal chlorosis (yellowing) of young leaves",
            "Severe plant stunting and drastic reduction in fruit yield"
        ],
        "preventive_actions": [
            "Control whitefly vectors using yellow sticky traps and approved insecticides",
            "Immediately rogue out and destroy symptomatic plants",
            "Use insect netting (fine mesh) in nursery and greenhouse settings"
        ],
        "severity": "Critical",
        "urgency": "Immediate removal of infected plants"
    },
    "Tomato___Tomato_mosaic_virus": {
        "display_name": "Tomato — Tomato Mosaic Virus (ToMV)",
        "pathogen_type": "Viral pathogen (Mechanically transmitted)",
        "description": "A highly stable, contagious viral pathogen transmitted through mechanical contact and seeds.",
        "symptoms": [
            "Mottling with alternating light and dark green areas on leaves",
            "Fern-like distortion or blistering of foliage",
            "Stunted growth and internal brown necrosis of fruits"
        ],
        "preventive_actions": [
            "Disinfect all tools and hands before handling plants (milk dip or trisodium phosphate)",
            "Remove and destroy infected plants immediately",
            "Avoid using tobacco products near tomato crops"
        ],
        "severity": "Critical",
        "urgency": "Immediate removal of infected plants"
    }
}

DISCLAIMER = (
    "DISCLAIMER: LeafLens is an educational and research screening prototype powered by computer vision "
    "and transfer learning. It is NOT a substitute for professional agricultural diagnosis, laboratory pathogen "
    "testing, or registered agronomist consultation. Always verify suspected crop pathology with local agricultural "
    "extension agents before applying chemical or pesticide treatments."
)

def get_disease_info(class_name: str) -> dict:
    """Retrieve detailed disease information with a graceful fallback."""
    if class_name in DISEASE_DATABASE:
        return DISEASE_DATABASE[class_name]
    
    # Generic fallback
    clean_name = class_name.replace("___", " - ").replace("_", " ")
    return {
        "display_name": clean_name,
        "pathogen_type": "Botanical / Foliar anomaly",
        "description": f"Condition classified as {clean_name}. Detailed clinical profile is pending specialized agronomic entry.",
        "symptoms": [
            "Foliar discoloration or morphological alteration consistent with classifier pattern.",
            "Visual inspection suggested for confirmation."
        ],
        "preventive_actions": [
            "Isolate suspected leaves and monitor nearby foliage.",
            "Consult local agricultural extension service for certified identification.",
            "Maintain sanitation and optimal watering practices."
        ],
        "severity": "Evaluation Recommended",
        "urgency": "Consult agronomic specialist"
    }
