"""
Upload tile index, shapefiles, and image tiles to Supabase
Run: python upload_to_supabase.py
"""

import os
import json
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION - Supabase credentials
# =============================================================================
SUPABASE_URL = "https://zuibgmmcyynfiaylkjns.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1aWJnbW1jeXluZmlheWxram5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY5MjAxMDMsImV4cCI6MjA4MjQ5NjEwM30.i_xsQciDxGA7BhUJkJmKs8vc7x0bYG3ktORVcAuSnwU"

# Local data paths
TILE_INDEX_PATH = Path(r"C:\fyp\tile_index.geojson")
SHAPEFILE_PATH = Path(r"C:\Users\junyi\Downloads\SabahShapefiles")

# Supabase table names (no image tiles - using live fetch)
TILE_INDEX_TABLE = "tile_index"
DISTRICT_TABLE = "districts"

# =============================================================================
# Install required packages first:
# pip install supabase geopandas tqdm
# =============================================================================

try:
    from supabase import create_client, Client
except ImportError:
    print("Installing supabase package...")
    os.system("pip install supabase")
    from supabase import create_client, Client

def check_credentials():
    if "YOUR_" in SUPABASE_URL or "YOUR_" in SUPABASE_KEY:
        print("="*60)
        print("ERROR: Please update SUPABASE_URL and SUPABASE_KEY")
        print("="*60)
        print("\n1. Go to https://supabase.com and create a project")
        print("2. Go to Settings > API")
        print("3. Copy 'Project URL' and 'anon public' key")
        print("4. Paste them in this file at lines 12-13")
        return False
    return True

def create_tables(supabase: Client):
    """Show table structure (already created by user)"""
    print("\nTables already created in Supabase SQL Editor")
    print("  - tile_index")
    print("  - districts")

def upload_tile_index(supabase: Client):
    """Upload tile index GeoJSON to Supabase"""
    print("\nUploading tile index...")
    
    if not TILE_INDEX_PATH.exists():
        print(f"  Tile index not found: {TILE_INDEX_PATH}")
        return
    
    gdf = gpd.read_file(str(TILE_INDEX_PATH))
    gdf = gdf.to_crs(epsg=4326)
    
    records = []
    for _, row in tqdm(gdf.iterrows(), total=len(gdf), desc="  Preparing"):
        record = {
            'png_path': row.get('png_path', ''),
            'mask_path': row.get('mask_path', ''),
            'tif_path': row.get('tif_path', ''),
            'geometry': row.geometry.wkt
        }
        records.append(record)
    
    # Upload in batches
    batch_size = 100
    for i in tqdm(range(0, len(records), batch_size), desc="  Uploading"):
        batch = records[i:i+batch_size]
        try:
            supabase.table(TILE_INDEX_TABLE).insert(batch).execute()
        except Exception as e:
            print(f"  Error uploading batch {i}: {e}")
    
    print(f"  Uploaded {len(records)} tile records")

def upload_shapefiles(supabase: Client):
    """Upload shapefiles to Supabase"""
    print("\nUploading shapefiles...")
    
    if not SHAPEFILE_PATH.exists():
        print(f"  Shapefile path not found: {SHAPEFILE_PATH}")
        return
    
    # Find .shp files
    shp_files = list(SHAPEFILE_PATH.glob("*.shp"))
    if not shp_files:
        print("  No .shp files found")
        return
    
    for shp_file in shp_files:
        print(f"  Processing: {shp_file.name}")
        gdf = gpd.read_file(str(shp_file))
        gdf = gdf.to_crs(epsg=4326)
        
        records = []
        for _, row in gdf.iterrows():
            # Use correct column names from shapefile
            record = {
                'name': row.get('ADM2_NAME', shp_file.stem.replace('_District_Boundary', '')),
                'type': row.get('ADM1_NAME', 'Sabah'),
                'geometry': row.geometry.wkt
            }
            records.append(record)
        
        # Upload
        try:
            supabase.table(DISTRICT_TABLE).insert(records).execute()
            print(f"    Uploaded {len(records)} features")
        except Exception as e:
            print(f"    Error: {e}")

def main():
    print("="*60)
    print("SUPABASE DATA UPLOAD")
    print("="*60)
    
    if not check_credentials():
        return
    
    print(f"\nConnecting to: {SUPABASE_URL}")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Connected!")
    
    # Show SQL to create tables
    create_tables(supabase)
    
    # Upload data (no tiles - using live fetch from Sentinel)
    upload_tile_index(supabase)
    upload_shapefiles(supabase)
    
    print("\n" + "="*60)
    print("UPLOAD COMPLETE!")
    print("="*60)
    print(f"View your data at: {SUPABASE_URL}")

if __name__ == "__main__":
    main()

