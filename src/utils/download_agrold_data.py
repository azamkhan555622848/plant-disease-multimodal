import os
from pathlib import Path
import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from tqdm import tqdm

def query_agrold():
    """Query AgroLD for plant disease information"""
    endpoint = "http://agrold.southgreen.fr/sparql"
    sparql = SPARQLWrapper(endpoint)
    
    # Query to get plant disease information
    query = """
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX obo: <http://purl.obolibrary.org/obo/>
    PREFIX dc: <http://purl.org/dc/elements/1.1/>
    
    SELECT DISTINCT ?disease ?diseaseName ?plant ?plantName ?description
    WHERE {
        ?disease a obo:OGMS_0000031 ;  # Disease
                rdfs:label ?diseaseName .
        OPTIONAL {
            ?disease dc:description ?description .
        }
        OPTIONAL {
            ?disease obo:RO_0002558 ?plant .  # Has host
            ?plant rdfs:label ?plantName .
        }
    }
    LIMIT 1000
    """
    
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    
    return results["results"]["bindings"]

def process_results(results):
    """Process SPARQL results into a structured format"""
    processed_data = []
    
    for result in results:
        entry = {
            'disease_name': result.get('diseaseName', {}).get('value', ''),
            'plant_name': result.get('plantName', {}).get('value', ''),
            'description': result.get('description', {}).get('value', ''),
            'disease_uri': result.get('disease', {}).get('value', ''),
            'plant_uri': result.get('plant', {}).get('value', '')
        }
        processed_data.append(entry)
    
    return pd.DataFrame(processed_data)

def create_text_descriptions(df):
    """Create detailed text descriptions from the data"""
    descriptions = []
    
    for _, row in df.iterrows():
        desc = f"Plant Disease: {row['disease_name']}\n"
        if row['plant_name']:
            desc += f"Host Plant: {row['plant_name']}\n"
        if row['description']:
            desc += f"Description: {row['description']}\n"
        desc += "---\n"
        descriptions.append(desc)
    
    return descriptions

def main():
    # Create directories
    base_dir = Path("data")
    agrold_dir = base_dir / "agrold"
    os.makedirs(agrold_dir, exist_ok=True)
    
    print("Querying AgroLD database...")
    try:
        # Get data from AgroLD
        results = query_agrold()
        
        # Process into DataFrame
        df = process_results(results)
        
        # Save raw data
        df.to_csv(agrold_dir / "agrold_raw.csv", index=False)
        print(f"Saved raw data to {agrold_dir}/agrold_raw.csv")
        
        # Create and save text descriptions
        descriptions = create_text_descriptions(df)
        
        with open(agrold_dir / "descriptions.txt", 'w', encoding='utf-8') as f:
            f.write("\n".join(descriptions))
        
        print(f"Saved text descriptions to {agrold_dir}/descriptions.txt")
        print(f"\nFound {len(df)} disease entries")
        
    except Exception as e:
        print(f"Error downloading AgroLD data: {e}")
        print("\nAlternative data sources:")
        print("1. Visit https://www.plantvillage.org/en/diseases")
        print("2. Use the Plant Disease Dataset descriptions")
        print("3. Create a descriptions.txt file in data/agrold/ with format:")
        print("   Plant Disease: [disease name]")
        print("   Host Plant: [plant name]")
        print("   Description: [description]")
        print("   ---")

if __name__ == "__main__":
    main() 