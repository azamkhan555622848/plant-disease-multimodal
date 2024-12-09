import os
from pathlib import Path
import json

# Dictionary of plant diseases and their descriptions
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab",
        "host_plant": "Apple",
        "description": "Apple scab is a fungal disease caused by Venturia inaequalis. Symptoms include dark, scaly lesions on leaves and fruit. The disease thrives in cool, wet conditions and can significantly reduce fruit quality and yield.",
        "symptoms": "Dark olive-green to brown spots on leaves, rough or corky patches on fruit surface.",
        "conditions": "Humid conditions, temperatures between 55-75°F, frequent rainfall."
    },
    "Apple___Black_rot": {
        "disease_name": "Black Rot",
        "host_plant": "Apple",
        "description": "Black rot is caused by the fungus Botryosphaeria obtusa. It affects leaves, fruit, and bark. The disease can cause significant fruit rot and branch dieback.",
        "symptoms": "Purple spots on leaves, rotting fruit with concentric rings, cankers on branches.",
        "conditions": "Warm, humid weather, poor sanitation, tree stress."
    },
    "Corn___Common_rust": {
        "disease_name": "Common Rust",
        "host_plant": "Corn",
        "description": "Common rust is caused by the fungus Puccinia sorghi. It produces small, circular to elongate brown pustules on both leaf surfaces. The disease can reduce yield in susceptible varieties.",
        "symptoms": "Circular to elongated brown pustules, chlorotic areas around pustules.",
        "conditions": "Cool temperatures (60-70°F), high humidity, heavy dew."
    },
    "Potato___Early_blight": {
        "disease_name": "Early Blight",
        "host_plant": "Potato",
        "description": "Early blight is caused by Alternaria solani. It affects leaves, stems, and tubers. The disease can cause significant defoliation and yield reduction.",
        "symptoms": "Dark brown lesions with concentric rings, yellowing of surrounding tissue.",
        "conditions": "Warm temperatures, high humidity, extended leaf wetness."
    },
    "Tomato___Leaf_Mold": {
        "disease_name": "Leaf Mold",
        "host_plant": "Tomato",
        "description": "Leaf mold is caused by the fungus Passalora fulva. It primarily affects leaves and can cause significant defoliation in greenhouse tomatoes.",
        "symptoms": "Pale green to yellow spots on upper leaf surface, olive-green to brown fuzzy growth on lower surface.",
        "conditions": "High humidity (85% or higher), moderate temperatures."
    }
}

def create_description(disease_info):
    """Create a formatted description from disease information"""
    desc = [
        f"Plant Disease: {disease_info['disease_name']}",
        f"Host Plant: {disease_info['host_plant']}",
        f"Description: {disease_info['description']}",
        f"Symptoms: {disease_info['symptoms']}",
        f"Favorable Conditions: {disease_info['conditions']}",
        "---"
    ]
    return "\n".join(desc)

def main():
    # Create directories
    base_dir = Path("data")
    agrold_dir = base_dir / "agrold"
    os.makedirs(agrold_dir, exist_ok=True)
    
    # Create descriptions
    descriptions = [create_description(info) for info in DISEASE_INFO.values()]
    
    # Save as text file
    with open(agrold_dir / "descriptions.txt", 'w', encoding='utf-8') as f:
        f.write("\n".join(descriptions))
    
    # Save raw data as JSON
    with open(agrold_dir / "disease_info.json", 'w', encoding='utf-8') as f:
        json.dump(DISEASE_INFO, f, indent=2)
    
    print(f"Created {len(DISEASE_INFO)} disease descriptions")
    print("Files saved:")
    print(f"- Text descriptions: {agrold_dir}/descriptions.txt")
    print(f"- Raw data: {agrold_dir}/disease_info.json")

if __name__ == "__main__":
    main() 