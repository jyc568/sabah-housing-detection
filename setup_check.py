"""
Demo Setup Script for Sabah Informal Housing Detection System
Run this after cloning to verify everything is ready.
"""

import sys
import os
from pathlib import Path

def check_python_version():
    print("Checking Python version...")
    if sys.version_info < (3, 9):
        print(f"  ❌ Python 3.9+ required, found {sys.version}")
        return False
    print(f"  ✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    print("\nChecking dependencies...")
    required = [
        ("flask", "Flask"),
        ("flask_cors", "Flask-CORS"),
        ("geopandas", "GeoPandas"),
        ("shapely", "Shapely"),
        ("PIL", "Pillow"),
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("timm", "timm"),
        ("supabase", "Supabase"),
    ]
    
    missing = []
    for module, name in required:
        try:
            __import__(module)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} - not installed")
            missing.append(name)
    
    if missing:
        print(f"\n  Install missing: pip install {' '.join(missing).lower()}")
        return False
    return True

def check_model_files():
    print("\nChecking model checkpoints...")
    
    # Check for model files
    original = Path("checkpoints/MOE_epoch_2_v2.pth")
    extended = Path("checkpoints/best_gram_extended.pth")
    
    found = True
    if original.exists():
        size_mb = original.stat().st_size / (1024*1024)
        print(f"  ✅ Original GRAM: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Original GRAM not found at {original}")
        print("     Download from GitHub Releases")
        found = False
    
    if extended.exists():
        size_mb = extended.stat().st_size / (1024*1024)
        print(f"  ✅ Extended GRAM: {size_mb:.1f} MB")
    else:
        print(f"  ❌ Extended GRAM not found at {extended}")
        print("     Download from GitHub Releases")
        found = False
    
    return found

def check_gram_model():
    print("\nChecking GRAM model architecture...")
    gram_model = Path("GRAM-main/model.py")
    if gram_model.exists():
        print(f"  ✅ GRAM model.py found")
        return True
    else:
        print(f"  ❌ GRAM model.py not found")
        return False

def check_static_files():
    print("\nChecking static files...")
    dashboard = Path("static/dashboard.html")
    if dashboard.exists():
        size_kb = dashboard.stat().st_size / 1024
        print(f"  ✅ Dashboard HTML: {size_kb:.1f} KB")
        return True
    else:
        print(f"  ❌ Dashboard not found at {dashboard}")
        return False

def check_supabase_config():
    print("\nChecking Supabase configuration...")
    with open("snapshot_server.py", "r") as f:
        content = f.read()
    
    if "zuibgmmcyynfiaylkjns.supabase.co" in content:
        print("  ✅ Supabase URL configured")
        return True
    elif "YOUR_SUPABASE" in content:
        print("  ⚠️ Supabase URL not configured (using local fallback)")
        return True
    else:
        print("  ✅ Supabase appears configured")
        return True

def main():
    print("=" * 60)
    print("SABAH INFORMAL HOUSING DETECTION - SETUP CHECK")
    print("=" * 60)
    
    checks = [
        check_python_version(),
        check_dependencies(),
        check_model_files(),
        check_gram_model(),
        check_static_files(),
        check_supabase_config(),
    ]
    
    print("\n" + "=" * 60)
    if all(checks):
        print("✅ ALL CHECKS PASSED!")
        print("\nRun the server with:")
        print("  python snapshot_server.py")
        print("\nThen open: http://127.0.0.1:5000")
    else:
        print("❌ SOME CHECKS FAILED")
        print("\nFix the issues above before running.")
    print("=" * 60)

if __name__ == "__main__":
    main()
