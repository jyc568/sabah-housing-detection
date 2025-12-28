# snapshot_server.py
# Minimal snapshot API. Requirements: flask, rasterio, shapely, geopandas, numpy, pillow, torch
import json
from flask import Flask, request, send_file, jsonify
import geopandas as gpd
from shapely.geometry import box, mapping
from shapely import wkt
from PIL import Image
import numpy as np
import io
from pathlib import Path
import torch
import torchvision.transforms as T
from gram_loader import get_gram_model, gram_predict
import cv2
from flask_cors import CORS
import math
import requests
from supabase import create_client

# -------- CONFIG ----------
# Supabase Cloud Database
SUPABASE_URL = "https://zuibgmmcyynfiaylkjns.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp1aWJnbW1jeXluZmlheWxram5zIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjY5MjAxMDMsImV4cCI6MjA4MjQ5NjEwM30.i_xsQciDxGA7BhUJkJmKs8vc7x0bYG3ktORVcAuSnwU"
USE_CLOUD_DB = True  # Set to False to use local files

# Local fallback paths (used if USE_CLOUD_DB is False)
TILE_INDEX = Path(r'tile_index.geojson')  # Original tile index
GADM_SHP = Path(r'shapefiles')  # for district lookup

# Ensemble: Original GRAM + Extended GRAM (newly trained)
ORIGINAL_CKPT = Path(r'checkpoints/MOE_epoch_2_v2.pth')
EXTENDED_CKPT = Path(r'checkpoints/best_gram_extended.pth')  # NEW: Extended model
# Weights for 2-model ensemble (should sum to 1.0)
WEIGHT_ORIGINAL = 0.2   # Original GRAM (high recall)
WEIGHT_EXTENDED = 0.8   # Extended GRAM (better precision, lower FPR)
MAX_TILES = 9   # max tiles to stitch (legacy, kept for compatibility)
MODEL_INPUT_SIZE = 512
THRESH = 0.1    # default threshold (overrideable per request)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LIVE_ZOOM = 14  # Live inference zoom level (approx 10m/px)
# -------------------------

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (xtile, ytile)

def num2deg(xtile, ytile, zoom):
    n = 2.0 ** zoom
    lon_deg = xtile / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
    lat_deg = math.degrees(lat_rad)
    return (lat_deg, lon_deg)

def fetch_live_tiles(bbox):
    # bbox: shapely box
    min_lon, min_lat, max_lon, max_lat = bbox.bounds
    
    # Get tile range
    xtile_min, ytile_max = deg2num(min_lat, min_lon, LIVE_ZOOM) # y is inverted? lat increases up, ytile decreases down?
    # deg2num: ytile 0 is at +85 lat. So high lat = low ytile.
    # min_lat is bottom, so it should be larger ytile. max_lat is top, smaller ytile.
    xtile_max, ytile_min = deg2num(max_lat, max_lon, LIVE_ZOOM)
    
    # Swap if needed (deg2num behavior check: lat increases -> ytile decreases)
    if ytile_min > ytile_max: ytile_min, ytile_max = ytile_max, ytile_min
    if xtile_min > xtile_max: xtile_min, xtile_max = xtile_max, xtile_min
    
    # Check count (limit to 3x3 or 4x4 to prevent abuse)
    width_tiles = xtile_max - xtile_min + 1
    height_tiles = ytile_max - ytile_min + 1
    if width_tiles * height_tiles > 16:
        print(f"Requested too many live tiles: {width_tiles}x{height_tiles}")
        return None, None, None, None
        
    imgs = []
    tile_coords = [] # (x, y)
    
    # Base URL (Sentinel-2 Cloudless - using EOX for consistency with frontend)
    base_url = "https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2020_3857/default/g/{z}/{y}/{x}.jpg"
    
    print(f"Fetching live tiles: x[{xtile_min}-{xtile_max}] y[{ytile_min}-{ytile_max}] z{LIVE_ZOOM}")
    
    full_canvas_h = height_tiles * 256
    full_canvas_w = width_tiles * 256
    canvas = np.zeros((full_canvas_h, full_canvas_w, 3), dtype=np.uint8)
    
    for x in range(xtile_min, xtile_max + 1):
        for y in range(ytile_min, ytile_max + 1):
            url = base_url.format(x=x, y=y, z=LIVE_ZOOM)
            try:
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                if r.status_code == 200:
                    arr = np.frombuffer(r.content, np.uint8)
                    im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) # 256x256
                    
                    # Paste into canvas
                    # Grid position relative to min
                    gx = x - xtile_min
                    gy = y - ytile_min
                    px = gx * 256
                    py = gy * 256
                    canvas[py:py+256, px:px+256] = im
                else:
                    print(f"Failed {url}: {r.status_code}")
            except Exception as e:
                print(f"Error fetching {url}: {e}")

    # BBox of the fetched canvas (outer edges of min/max tiles)
    lat_n, lon_w = num2deg(xtile_min, ytile_min, LIVE_ZOOM)
    lat_s, lon_e = num2deg(xtile_max + 1, ytile_max + 1, LIVE_ZOOM)
    
    # In num2deg: xtile is left edge, ytile is top edge.
    # xtile_min -> left of left tile (West)
    # ytile_min -> top of top tile (North)
    # xtile_max+1 -> right of right tile (East)
    # ytile_max+1 -> bottom of bottom tile (South)
    
    # Note: num2deg returns (lat, lon) for NW corner of tile (xtile, ytile).
    
    # Compute ppd
    # 256 pixels / (lon_e - lon_w) for full width? No, per tile.
    # tile_width_deg ≈ 360 / 2^zoom
    tile_w_deg = 360.0 / (2**LIVE_ZOOM)
    ppd_x = 256.0 / tile_w_deg
    
    # Lat varies, but approx ppd_y
    # We can rely on the computed canvas corners:
    out_bbox_coords = (lon_w, lat_s, lon_e, lat_n) # minx, miny, maxx, maxy
    
    # Canvas is built. No GT for live tiles.
    gt_canvas = np.zeros((full_canvas_h, full_canvas_w), dtype=np.uint8)
    
    # Compute PPD based on actual bounds coverage (approx for standard Mercator)
    if (lon_e - lon_w) == 0: return None, None, None, None
    ppd_x = full_canvas_w / (lon_e - lon_w)
    ppd_y = full_canvas_h / (lat_n - lat_s)
    
    return canvas, gt_canvas, out_bbox_coords, (ppd_x, ppd_y)

app = Flask(__name__, static_url_path='/static', static_folder='static')
CORS(app) # Enable CORS for all routes

# Load data from Supabase or local files
def load_tile_index_from_supabase():
    """Load tile index from Supabase cloud database"""
    print("Loading tile index from Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table('tile_index').select('*').execute()
    
    if not result.data:
        print("  No tile index data found in Supabase!")
        return gpd.GeoDataFrame()
    
    # Convert to GeoDataFrame
    records = []
    geometries = []
    for row in result.data:
        try:
            geom_data = row.get('geometry')
            if geom_data is None:
                continue
            # Handle both WKT string and GeoJSON dict
            if isinstance(geom_data, str):
                geom = wkt.loads(geom_data)
            elif isinstance(geom_data, dict):
                from shapely.geometry import shape
                geom = shape(geom_data)
            else:
                print(f"  Unknown geometry type: {type(geom_data)}")
                continue
            
            records.append({
                'png_path': row.get('png_path', ''),
                'mask_path': row.get('mask_path', ''),
            })
            geometries.append(geom)
        except Exception as e:
            print(f"  Error parsing geometry: {e}")
    
    if not records:
        print("  No valid geometries found!")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    print(f"  Loaded {len(gdf)} tiles from cloud")
    return gdf

def load_districts_from_supabase():
    """Load districts from Supabase cloud database"""
    print("Loading districts from Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    result = supabase.table('districts').select('*').execute()
    
    if not result.data:
        print("  No district data found in Supabase!")
        return gpd.GeoDataFrame()
    
    # Convert to GeoDataFrame
    records = []
    geometries = []
    for row in result.data:
        try:
            geom_data = row.get('geometry')
            if geom_data is None:
                continue
            # Handle both WKT string and GeoJSON dict
            if isinstance(geom_data, str):
                geom = wkt.loads(geom_data)
            elif isinstance(geom_data, dict):
                from shapely.geometry import shape
                geom = shape(geom_data)
            else:
                print(f"  Unknown geometry type: {type(geom_data)}")
                continue
            
            records.append({
                'name': row.get('name', ''),
                'type': row.get('type', ''),
            })
            geometries.append(geom)
        except Exception as e:
            print(f"  Error parsing geometry: {e}")
    
    if not records:
        print("  No valid geometries found!")
        return gpd.GeoDataFrame()
    
    gdf = gpd.GeoDataFrame(records, geometry=geometries, crs="EPSG:4326")
    print(f"  Loaded {len(gdf)} districts from cloud")
    return gdf

# Load tile index and districts
if USE_CLOUD_DB:
    print("\n=== USING SUPABASE CLOUD DATABASE ===")
    tile_index = load_tile_index_from_supabase()
    gadm = load_districts_from_supabase()
else:
    print("\n=== USING LOCAL FILES ===")
    tile_index = gpd.read_file(str(TILE_INDEX)).to_crs(epsg=4326)
    gadm = gpd.read_file(str(GADM_SHP)).to_crs(epsg=4326)

# load models (ensemble)
print("Loading ORIGINAL GRAM model...")
model_original = get_gram_model(str(ORIGINAL_CKPT))
print("Loading EXTENDED GRAM model...")
model_extended = get_gram_model(str(EXTENDED_CKPT))

transform = T.Compose([T.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)), T.ToTensor(),
                       T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])

def find_tiles_for_bbox(min_lon, min_lat, max_lon, max_lat, max_tiles=MAX_TILES):
    query_geom = box(min_lon, min_lat, max_lon, max_lat)
    matches = tile_index[tile_index.geometry.intersects(query_geom)].copy()
    matches['int_area'] = matches.geometry.intersection(query_geom).area
    matches = matches.sort_values('int_area', ascending=False)
    return matches.head(max_tiles)

def stitch_tiles(matches, bbox):
    # matches: GeoDataFrame with png_path, geometry in EPSG:4326
    # bbox: shapely box in same CRS
    # load PNGs and GT masks, place them into a canvas
    imgs = []
    masks = [] # GT masks
    bboxes = []
    for _, r in matches.iterrows():
        png = r['png_path']
        if not png or not Path(png).exists():
            continue
        
        # Load Image
        im = np.array(Image.open(png).convert('RGB'))
        imgs.append(im)
        
        # Load Mask if exists
        # Assumption: structure is .../images/filename.png -> .../masks/filename.png
        # We try to replace "images" folder with "masks"
        mask_path = png.replace('images', 'masks') 
        # Handle case sensitivity or slight path diffs if strictly needed, but try direct first
        if Path(mask_path).exists():
            # Load as grayscale
            m = np.array(Image.open(mask_path).convert('L'))
            # binarize 0 or 255
            m = (m > 127).astype(np.uint8) * 255
        else:
            # Empty mask
            m = np.zeros(im.shape[:2], dtype=np.uint8)
            
        masks.append(m)
        bboxes.append(r.geometry.bounds)  # (minx,miny,maxx,maxy)
        
    if not imgs:
        return None, None, None, None
        
    # For simplicity: place images in grid via their geocentroids:
    centroids = [ ( (b[0]+b[2])/2.0, (b[1]+b[3])/2.0 ) for b in bboxes ]
    
    # compute pixel scale
    tile_width_deg = np.mean([b[2]-b[0] for b in bboxes])
    tile_height_deg = np.mean([b[3]-b[1] for b in bboxes])
    
    out_minx, out_miny, out_maxx, out_maxy = bbox.bounds
    
    # choose pixels per deg
    ppd_x = imgs[0].shape[1] / tile_width_deg
    ppd_y = imgs[0].shape[0] / tile_height_deg
    
    # canvas size
    W = int(np.round((out_maxx - out_minx) * ppd_x))
    H = int(np.round((out_maxy - out_miny) * ppd_y))
    W = max(W, MODEL_INPUT_SIZE); H = max(H, MODEL_INPUT_SIZE)
    
    canvas = np.zeros((H, W, 3), dtype=np.uint8)
    gt_canvas = np.zeros((H, W), dtype=np.uint8) # Single channel for GT
    
    # paste each tile
    for im, m, b in zip(imgs, masks, bboxes):
        # compute pixel position for tile's top-left
        tile_minx, tile_miny, tile_maxx, tile_maxy = b
        px = int(np.round((tile_minx - out_minx) * ppd_x))
        py = int(np.round((out_maxy - tile_maxy) * ppd_y))  # origin top-left
        
        h,w = im.shape[:2]
        # clip and paste
        x0 = max(0, px); y0 = max(0, py)
        x1 = min(W, px + w); y1 = min(H, py + h)
        
        sx0 = max(0, -px); sy0 = max(0, -py)
        sx1 = sx0 + (x1 - x0); sy1 = sy0 + (y1 - y0)
        
        if x1 > x0 and y1 > y0:
            canvas[y0:y1, x0:x1] = im[sy0:sy1, sx0:sx1]
            gt_canvas[y0:y1, x0:x1] = m[sy0:sy1, sx0:sx1]
            
    return canvas, gt_canvas, (out_minx, out_miny, out_maxx, out_maxy), (ppd_x, ppd_y)

def predict_patch(model, img_patch):
    # img_patch: PIL Image or numpy (H,W,3)
    # returns: (H,W) prob map
    if isinstance(img_patch, np.ndarray):
        img_patch = Image.fromarray(img_patch).convert('RGB')
        
    # Transform expects PIL, resizes to 512 (which is what we want for 512 patches)
    # But wait, our transform Resize(512) might distort if patch is not 512?
    # We will ensure patch is 512x512.
    inp = transform(img_patch).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
         # GRAM predict
        prob_t = gram_predict(model, inp) # (1, 1, 512, 512) or similar
        prob = prob_t.squeeze().cpu().numpy()
        
    return prob

def model_predict_on_canvas(canvas):
    # canvas: HxWx3 uint8
    # Strategy: 
    # 1. Upscale canvas by 2x (to match training flow: 256 tile -> 512 input)
    # 2. Pad to multiple of 512
    # 3. Sliding window (512x512)
    # 4. Crop and Resize back
    
    H_orig, W_orig = canvas.shape[:2]
    
    # 1. Upscale 2x
    # We use cv2 for speed
    upscale_h = H_orig * 2
    upscale_w = W_orig * 2
    canvas_up = cv2.resize(canvas, (upscale_w, upscale_h), interpolation=cv2.INTER_LINEAR)
    
    # 2. Pad to multiple of 512
    # patch size
    P = 512
    stride = 512 # non-overlapping for speed/simplicity. Overlapping better for seams but 2x scale is robust.
    
    pad_h = (P - (upscale_h % P)) % P
    pad_w = (P - (upscale_w % P)) % P
    
    canvas_padded = np.pad(canvas_up, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    H_pad, W_pad = canvas_padded.shape[:2]
    
    prob_map_pad = np.zeros((H_pad, W_pad), dtype=np.float32)
    counts_map = np.zeros((H_pad, W_pad), dtype=np.float32)
    
    # 3. Sliding loop
    for y in range(0, H_pad, stride):
        for x in range(0, W_pad, stride):
            # Extract patch
            patch = canvas_padded[y:y+P, x:x+P, :]
            if patch.shape[0] != P or patch.shape[1] != P:
                # Should not happen if padded correctly
                continue
                
            # Predict
            # patch is 512x512. 'transform' will do ToTensor and Normalize. 
            # Resize(512) in transform is redundancy but safe.
            prob_p = predict_patch(model, patch) # returns 512x512
            
            prob_map_pad[y:y+P, x:x+P] = prob_p
            counts_map[y:y+P, x:x+P] += 1
            
    # Normalize (if overlapping)
    # prob_map_pad /= np.maximum(counts_map, 1)
    
    # 4. Crop and Resize back
    prob_map_up = prob_map_pad[:upscale_h, :upscale_w]
    
    # Downscale to original resolution
    prob_resized = cv2.resize(prob_map_up, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
    
    return prob_resized

def postprocess_and_vectorize(prob_map, thr=THRESH):
    mask = (prob_map >= thr).astype('uint8') * 255
    # morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < 50: continue
        # compute bounding box and mean_prob
        mask_comp = np.zeros_like(mask)
        cv2.drawContours(mask_comp, [c], -1, 255, -1)
        meanprob = float((prob_map * (mask_comp>0)).sum() / (mask_comp>0).sum())
        # approx polygon coords in pixel space
        coords = c.squeeze().tolist()
        polys.append({'coords_px': coords, 'area_px': area, 'mean_prob': meanprob})
    return polys, mask

@app.route('/snapshot', methods=['POST'])
def snapshot():
    js = request.json
    min_lon = float(js.get('min_lon')); min_lat = float(js.get('min_lat'))
    max_lon = float(js.get('max_lon')); max_lat = float(js.get('max_lat'))
    thr = float(js.get('thr', THRESH))
    # Logic Change: Enforce 512x512 at Zoom 14 (approx 10m/px)
    # 1. Calculate center
    c_lon = (min_lon + max_lon) / 2.0
    c_lat = (min_lat + max_lat) / 2.0
    
    # 2. Determine degrees per pixel at Z14
    # At Z14, 360 degrees = 2^14 * 256 pixels
    # deg_per_px = 360 / (2**14 * 256)
    Z14_RES = 360.0 / (2**14 * 256.0) # approx 8.5e-5 deg/px
    
    # We want exactly 512x512 pixels
    # But wait, aspect ratio in degrees changes with latitude. 
    # lon_deg_per_px = Z14_RES
    # lat_deg_per_px = Z14_RES * cos(lat) ? 
    # Mercator is simpler: 
    # Use num2deg to find consistent box
    
    # Find center tile coordinates
    xtile, ytile = deg2num(c_lat, c_lon, LIVE_ZOOM)
    # We want a 512x512 pixel window around this point. 
    # LIVE_ZOOM tiles are 256x256. So we need a 2x2 tile area roughly? 
    # Or just crop from the tiles we get.
    
    # Let's define the fixed bbox in DEGREES that corresponds to 512x512 pixels at this zoom
    # Actually, simpler: just grab the tiles covering 512x512 pixels around center
    
    # For now, let's just use the logic to get a box of "radius" ~250 pixels
    tgt_w_deg = 512 * Z14_RES
    tgt_h_deg = 512 * Z14_RES * math.cos(math.radians(c_lat)) # approx corrections
    
    # Re-define query bbox to be this standardized window
    req_min_lon = c_lon - tgt_w_deg/2
    req_max_lon = c_lon + tgt_w_deg/2
    req_min_lat = c_lat - tgt_h_deg/2 
    req_max_lat = c_lat + tgt_h_deg/2
    
    # Override the user's bbox with this fixed one
    # Note: Frontend might be confused if overlay doesn't match drawn box?
    # User said "Program only allow for one bbox... size 512x512"
    # We should return the image and let frontend display it.
    # But if frontend draws overlay at `bbox`, we must update `bbox`?
    # Actually, we can return the NEW bbox in the headers or just return the image
    # and relying on frontend using the same bbox? 
    # Usually `snapshot` assumes request bbox == response bbox.
    # We will proceed with fetching tiles for this NEW bbox.
    
    matches = find_tiles_for_bbox(req_min_lon, req_min_lat, req_max_lon, req_max_lat, max_tiles=9)
    canvas = None
    gt_canvas = None
    
    bbox_obj = box(req_min_lon, req_min_lat, req_max_lon, req_max_lat)
    
    if not matches.empty:
        canvas, gt_canvas, bbox_coords, ppd = stitch_tiles(matches, bbox_obj)
        
    if canvas is None:
        print("Local tiles not found. Fallback to Live Z14...")
        canvas, gt_canvas, bbox_coords, ppd = fetch_live_tiles(bbox_obj)
        
    if canvas is None:
        return jsonify({'error':'no tiles found'}), 404
        
    # Resize strictly to 512x512 (Standardize)
    # The stitching might have produced something slightly off due to tile alignment
    canvas = cv2.resize(canvas, (512, 512), interpolation=cv2.INTER_LINEAR)
    if gt_canvas is not None:
        gt_canvas = cv2.resize(gt_canvas, (512, 512), interpolation=cv2.INTER_NEAREST)
        
    # Recalculate bbox_coords to match exactly what we are returning?
    # We essentially returned the requested fixed window.
    # Force bbox_coords to match the standard request
    bbox_coords = (req_min_lon, req_min_lat, req_max_lon, req_max_lat)
        
    # ENSEMBLE Prediction: Run all 3 models and combine
    inp_tensor = transform(Image.fromarray(canvas)).unsqueeze(0).to(DEVICE)
    
    # Original GRAM prediction
    prob_orig_t = gram_predict(model_original, inp_tensor)
    prob_orig = prob_orig_t.squeeze().cpu().numpy()
    
    # Extended GRAM prediction
    prob_ext_t = gram_predict(model_extended, inp_tensor)
    prob_ext = prob_ext_t.squeeze().cpu().numpy()
    
    # Weighted 2-model ensemble
    prob = WEIGHT_ORIGINAL * prob_orig + WEIGHT_EXTENDED * prob_ext
    
    # Postprocess
    polys, mask = postprocess_and_vectorize(prob, thr=thr)
    
    # ... (Rest of function uses polys, mask, gt_canvas)
    # We need to ensure we don't duplicate code or break flow.
    # The original code continued...
    
    # Old block removed, flow continues to IoU Calculation
    # IoU Calculation (only meaningful when GT exists)
    pred_bool = mask > 0
    gt_bool = gt_canvas > 0 if gt_canvas is not None else np.zeros_like(pred_bool)
    
    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0
    
    # Model Confidence Score - average probability in detected regions
    # This is independent of ground truth and shows how confident the model is
    if pred_bool.sum() > 0:
        model_confidence = float(prob[pred_bool].mean())
    else:
        model_confidence = 0.0
    
    # Detection coverage - percentage of analyzed area with detections
    detection_coverage = float(pred_bool.sum()) / pred_bool.size * 100
    
    # compute stats & district join
    results = []
    for p in polys:
        # convert pixel coords to lon/lat using bbox coords and ppd
        out_minx, out_miny, out_maxx, out_maxy = bbox_coords
        ppd_x, ppd_y = ppd
        # convert coordinates: px (x) to lon = out_minx + px / ppd_x
        lonlat_coords = []
        for x,y in p['coords_px']:
            lon = out_minx + x / ppd_x
            lat = out_maxy - y / ppd_y
            lonlat_coords.append((lon, lat))
        poly_geom = box(min([c[0] for c in lonlat_coords]), min([c[1] for c in lonlat_coords]),
                        max([c[0] for c in lonlat_coords]), max([c[1] for c in lonlat_coords]))
        # find district
        df_j = gadm[gadm.geometry.intersects(box(poly_geom.bounds[0], poly_geom.bounds[1], poly_geom.bounds[2], poly_geom.bounds[3]))]
        district = df_j.iloc[0]['NAME_2'] if not df_j.empty and 'NAME_2' in df_j.columns else None
        results.append({'polygon_lonlat': lonlat_coords, 'area_px': p['area_px'], 'mean_prob': p['mean_prob'], 'district': district})
        
    # prepare overlay PNG (RGBA)
    H, W = canvas.shape[:2]
    overlay_rgba = np.zeros((H, W, 4), dtype=np.uint8) # Transparent
    
    # TP (Yellow) = Pred & GT
    tp = np.logical_and(pred_bool, gt_bool)
    overlay_rgba[tp] = [255, 255, 0, 160] # Yellow
    
    # FP (Red) = Pred & ~GT
    fp = np.logical_and(pred_bool, ~gt_bool)
    overlay_rgba[fp] = [255, 0, 0, 160] # Red
    
    # FN (Green) = ~Pred & GT
    fn = np.logical_and(~pred_bool, gt_bool)
    overlay_rgba[fn] = [0, 255, 0, 160] # Green
    
    # send overlay as png bytes
    pil = Image.fromarray(overlay_rgba)
    buf = io.BytesIO(); pil.save(buf, format='PNG'); buf.seek(0)
    
    resp = send_file(buf, mimetype='image/png', as_attachment=False,
                     download_name='snapshot_overlay.png')
    resp.headers['X-IoU-Score'] = f"{iou:.4f}"
    resp.headers['X-Model-Confidence'] = f"{model_confidence:.4f}"
    resp.headers['X-Detection-Coverage'] = f"{detection_coverage:.2f}"
    resp.headers['X-Match-Status'] = "High Confidence" if model_confidence > 0.3 else "Low Confidence"
    # Return the actual bbox used for the 512x512 window
    resp.headers['X-Actual-BBox'] = json.dumps([bbox_coords[0], bbox_coords[1], bbox_coords[2], bbox_coords[3]])
    return resp, 200

@app.route('/', methods=['GET'])
def index():
    return send_file('static/dashboard.html')

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'online', 'endpoints': ['/snapshot (POST)', '/analyze_district (POST)']}), 200

@app.route('/analyze_district', methods=['POST'])
def analyze_district():
    """Analyze all tiles for a specific district and return detection results"""
    data = request.json
    district = data.get('district', '')
    
    if not district:
        return jsonify({'error': 'District name required'}), 400
    
    # Filter tiles by district
    district_tiles = tile_index[tile_index['district'].str.contains(district, case=False, na=False)]
    
    if len(district_tiles) == 0:
        return jsonify({'error': f'No tiles found for district: {district}'}), 404
    
    tiles_info = []
    detections = []
    total_confidence = 0.0
    detection_count = 0
    
    # Process up to 20 tiles for performance (can be adjusted)
    max_tiles = min(20, len(district_tiles))
    sample_tiles = district_tiles.sample(n=max_tiles) if len(district_tiles) > max_tiles else district_tiles
    
    for idx, row in sample_tiles.iterrows():
        # Get tile bounds
        bounds = row.geometry.bounds  # (minx, miny, maxx, maxy)
        min_lon, min_lat, max_lon, max_lat = bounds
        
        tiles_info.append({
            'min_lon': min_lon, 'min_lat': min_lat,
            'max_lon': max_lon, 'max_lat': max_lat,
            'name': row.get('tile_name', f'tile_{idx}')
        })
        
        # Load tile image
        png_path = row.get('png_path', '')
        if png_path and Path(png_path).exists():
            try:
                img = Image.open(png_path).convert('RGB')
                img_resized = img.resize((512, 512))
                inp = transform(img_resized).unsqueeze(0).to(DEVICE)
                
                # Ensemble prediction
                with torch.no_grad():
                    prob_orig = gram_predict(model_original, inp).squeeze().cpu().numpy()
                    prob_ft = gram_predict(model_finetuned, inp).squeeze().cpu().numpy()
                    prob = WEIGHT_ORIGINAL * prob_orig + WEIGHT_FINETUNED * prob_ft
                
                # Check if there are detections
                mask = (prob >= THRESH).astype(np.uint8)
                if mask.sum() > 100:  # At least 100 positive pixels
                    # Create overlay image
                    overlay = np.zeros((512, 512, 4), dtype=np.uint8)
                    overlay[mask > 0] = [255, 0, 0, 180]  # Red with alpha
                    
                    # Encode as base64 data URL
                    overlay_img = Image.fromarray(overlay, mode='RGBA')
                    buf = io.BytesIO()
                    overlay_img.save(buf, format='PNG')
                    buf.seek(0)
                    import base64
                    b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
                    overlay_url = f"data:image/png;base64,{b64}"
                    
                    confidence = float(prob[mask > 0].mean()) if mask.sum() > 0 else 0.0
                    total_confidence += confidence
                    detection_count += 1
                    
                    detections.append({
                        'min_lon': min_lon, 'min_lat': min_lat,
                        'max_lon': max_lon, 'max_lat': max_lat,
                        'overlay_url': overlay_url,
                        'confidence': confidence
                    })
            except Exception as e:
                print(f"Error processing tile {png_path}: {e}")
                continue
    
    avg_confidence = total_confidence / max(1, detection_count)
    
    return jsonify({
        'district': district,
        'tiles': tiles_info,
        'detections': detections,
        'detection_count': detection_count,
        'avg_confidence': avg_confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

