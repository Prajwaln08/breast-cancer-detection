"""Breast Ultrasound AI: research portal.

Shows what the v3 multi-task U-Net ensemble predicts for one scan: normal / benign / malignant,
the lesion outlines, and each lesion's position, size and shape.

Run from the project folder:
    .env/bin/streamlit run ultrasonic_scans/app.py

The image steps are the same as in 03_model_unet.ipynb and the measurements are the same as in
04_measure.ipynb, so the app gives the same results as the notebooks.
"""
import base64
import hashlib
import html
import io
import json
import os
import tempfile
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import streamlit as st
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps, UnidentifiedImageError
from scipy import ndimage
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.distance import pdist, squareform
from skimage.measure import approximate_polygon, find_contours, label, perimeter, regionprops

BASE_DIR = Path(__file__).resolve().parent     # the ultrasonic_scans folder
MODEL_DIR = BASE_DIR / 'outputs' / 'unet_multitask'     # model files on a computer that trained them
ASSET_DIR = BASE_DIR / 'portal_assets'                  # settings and test results used by the portal
DATA_DIR = ASSET_DIR / 'examples'                       # test-set example scans and their masks
DOWNLOAD_DIR = Path(tempfile.gettempdir()) / 'breast_ultrasound_ai_models'
MODEL_RELEASE_URL = 'https://github.com/Prajwaln08/breast-cancer-detection/releases/download/v3-cross-validation'
MAX_IMAGE_SIDE = 4000                          # larger uploads are refused (scans here are about 500-1000 px)
MODEL_VERSION = 'v3'
DATASET_URL = 'https://doi.org/10.1016/j.dib.2019.104863'   # the published BUSI dataset paper
RENDER_SIDE = 1100                             # longest side of the annotated image, for sharp lines and labels

LESION_COLOUR = (225, 29, 92)                  # outline and label of predicted lesions
DIAMETER_COLOUR = (34, 211, 238)               # longest-diameter line
TRUE_COLOUR = (74, 222, 128)                   # hand-drawn outline (test images only)


# ------------------------------------------------------------------
# Page styling and graphics
# ------------------------------------------------------------------

PAGE_CSS = """
:root {
  --navy: #0B2F4E; --blue: #145A86; --teal: #0F766E; --teal-soft: #E3F4F1;
  --ink: #15293D; --muted: #52647A; --line: #D9E3EC; --card: #FFFFFF; --page: #F4F7FA;
  --red: #B42318; --red-soft: #FDECEA; --amber: #8A5A00; --amber-soft: #FFF4DB;
  --green: #17694D; --green-soft: #E5F5EE; --blue-soft: #E6F1F8; --track: #E4EBF2;
}
.stApp { background: var(--page); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], [data-testid="stDecoration"] { display: none; }
.block-container { max-width: 1320px; padding: 1rem 1.5rem 3rem; }
.stApp h1, .stApp h2, .stApp h3 { color: inherit; padding: 0; margin: 0; }

.hero { position: relative; overflow: hidden; display: flex; align-items: center; gap: 14px;
  padding: 15px 22px; border-radius: 14px; color: #fff;
  background: linear-gradient(120deg, #0B2F4E 0%, #115073 58%, #0F766E 100%); }
.hero-logo { flex: 0 0 auto; width: 46px; height: 46px; position: relative; z-index: 1; }
.hero-text { position: relative; z-index: 1; min-width: 0; }
.hero h1 { font-size: 1.5rem; line-height: 1.15; font-weight: 700; letter-spacing: 0; color: #fff; }
.hero p { margin: 2px 0 0; font-size: .92rem; color: #DCEBF3; }
.hero-tags { position: relative; z-index: 1; display: flex; flex-wrap: wrap; gap: 8px; margin-left: auto; }
.hero-tag { padding: 5px 11px; border: 1px solid rgba(255,255,255,.55); background: rgba(11,47,78,.72);
  border-radius: 999px; font-size: .8rem; font-weight: 600; color: #fff; }
.hero-wave { position: absolute; right: 0; top: -60px; width: 430px; height: 220px; opacity: .22; }

.notice { display: flex; gap: 11px; align-items: flex-start; padding: 10px 15px; margin: 10px 0 0;
  border-radius: 11px; border: 1px solid; }
.notice img { flex: 0 0 auto; width: 22px; height: 22px; margin-top: 1px; }
.notice strong { font-size: .94rem; margin-right: 6px; }
.notice span { font-size: .9rem; line-height: 1.4; }
.notice.disclaimer { background: var(--blue-soft); border-color: #BCD7E8; color: var(--navy); }
.notice.warning { background: var(--amber-soft); border-color: #EBCB85; color: #5F3F00; margin: 0 0 12px; padding: 13px 16px; }
.notice.warning strong { display: block; font-size: 1rem; }

.section-title { display: flex; align-items: center; gap: 9px; font-size: 1.14rem; font-weight: 700;
  margin: 18px 0 8px; color: var(--ink); }
.section-title b { display: inline-grid; place-items: center; width: 25px; height: 25px; border-radius: 50%;
  background: var(--navy); color: #fff; font-size: .82rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 13px; padding: 15px 17px; color: var(--ink); }
.panel-head { display: flex; align-items: center; gap: 9px; }
.panel-head img { width: 24px; height: 24px; flex: 0 0 auto; }
.panel-head h2, .panel-head h3 { font-size: 1.02rem; line-height: 1.25; font-weight: 700; color: var(--navy); }
.panel-head p { margin: 1px 0 0; font-size: .82rem; color: var(--muted); }
.active-tag { margin-left: auto; padding: 3px 9px; border-radius: 999px; background: var(--teal-soft);
  color: #0B5D57; font-size: .76rem; font-weight: 700; white-space: nowrap; }
.input-note { margin: 0; font-size: .8rem; line-height: 1.35; color: var(--muted); }
.input-note a { color: #0B5D57; font-weight: 600; text-decoration: underline; }
.st-key-upload_panel > :last-child, .st-key-example_panel > :last-child { margin-top: auto; }
.st-key-upload_panel, .st-key-example_panel { background: var(--card); border-radius: 13px; }
div[data-testid="stImage"] img { border-radius: 10px; }

.overview { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(0, 1fr); gap: 12px; margin-top: 12px; }
.mini-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 11px; }
.mini { padding: 9px 11px; border: 1px solid var(--line); border-radius: 10px; background: #FAFCFD; }
.mini .value { font-size: 1.28rem; line-height: 1.15; font-weight: 700; color: var(--navy); }
.mini .name { margin-top: 2px; font-size: .82rem; line-height: 1.2; font-weight: 600; }
.mini .range { margin-top: 2px; font-size: .73rem; color: var(--muted); }
.caveat { margin: 9px 0 0; font-size: .79rem; line-height: 1.4; color: var(--muted); }
.steps { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; margin-top: 11px; }
.step { display: flex; gap: 8px; padding: 9px 10px; border: 1px solid var(--line); border-radius: 10px; background: #FAFCFD; }
.step b { flex: 0 0 auto; display: inline-grid; place-items: center; width: 23px; height: 23px; border-radius: 50%;
  background: var(--teal); color: #fff; font-size: .76rem; }
.step strong { display: block; font-size: .86rem; color: var(--navy); }
.step span { display: block; margin-top: 1px; font-size: .77rem; line-height: 1.3; color: var(--muted); }

.case-bar { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin-bottom: 12px; padding: 11px 16px; }
.case-bar span { display: block; font-size: .74rem; font-weight: 700; color: var(--muted); }
.case-bar strong { display: block; margin-top: 1px; font-size: .93rem; overflow-wrap: anywhere; }

.status { display: flex; align-items: center; gap: 14px; margin-top: 12px; }
.ring { flex: 0 0 auto; width: 86px; height: 86px; }
.status-label { font-size: .78rem; font-weight: 700; color: var(--muted); }
.status-name { font-size: 1.32rem; line-height: 1.2; font-weight: 700; }
.tone-normal { color: var(--green); } .tone-benign { color: var(--blue); } .tone-malignant { color: var(--red); }
.explain { margin: 11px 0 0; padding: 9px 11px; border-radius: 9px; font-size: .86rem; line-height: 1.4; }
.explain.normal { background: var(--green-soft); color: #0F523B; }
.explain.benign { background: var(--blue-soft); color: #0E4264; }
.explain.malignant { background: var(--red-soft); color: #8A1C13; }
.prob { margin-top: 12px; }
.prob-head { display: flex; justify-content: space-between; font-size: .9rem; font-weight: 600; }
.bar { position: relative; height: 9px; margin-top: 5px; background: var(--track); border-radius: 999px; }
.bar i { display: block; height: 100%; border-radius: 999px; }
.bar b { position: absolute; top: -3px; width: 2px; height: 15px; background: var(--ink); }
.bar-note { margin-top: 4px; font-size: .76rem; color: var(--muted); }
.fact { display: flex; justify-content: space-between; gap: 12px; padding-top: 10px; margin-top: 11px;
  border-top: 1px solid var(--line); font-size: .9rem; }
.fact span { color: var(--muted); } .fact strong { text-align: right; }

.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; margin: 2px 0 0; font-size: .84rem; color: var(--muted); }
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend i { display: inline-block; width: 18px; height: 3px; border-radius: 3px; }

.lesion-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.metric-card h3 { display: flex; align-items: center; gap: 8px; font-size: .98rem; font-weight: 700; color: var(--navy); }
.metric-card h3 img { width: 20px; height: 20px; }
.metric-body { display: flex; align-items: center; gap: 15px; margin-top: 11px; }
.metric-graphic { flex: 0 0 104px; width: 104px; }
.metric-graphic img { display: block; width: 100%; height: auto; }
.big { font-size: 1.2rem; line-height: 1.2; font-weight: 700; }
.small { margin-top: 4px; font-size: .84rem; line-height: 1.4; color: var(--muted); }
.size-bar { margin-top: 14px; }

.kpi-title { display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin: 16px 0 7px; }
.kpi-title h3 { font-size: 1rem; font-weight: 700; color: var(--navy); }
.kpi-title span { font-size: .78rem; color: var(--muted); }
.kpi-grid { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 9px; }
.kpi { padding: 11px 13px; background: var(--card); border: 1px solid var(--line); border-radius: 11px; }
.kpi span { display: block; font-size: .78rem; line-height: 1.25; font-weight: 600; color: var(--muted); }
.kpi strong { display: block; margin-top: 3px; font-size: 1.14rem; line-height: 1.2; color: var(--ink); overflow-wrap: anywhere; }
.empty { text-align: center; padding: 30px 20px; margin-top: 12px; }
.empty img { width: 62px; height: 62px; }
.empty strong { display: block; margin-top: 7px; font-size: 1.05rem; }
.empty span { display: block; margin: 3px auto 0; max-width: 560px; color: var(--muted); font-size: .9rem; }

@media (max-width: 1000px) {
  .overview, .lesion-grid { grid-template-columns: minmax(0, 1fr); }
  .kpi-grid { grid-template-columns: repeat(3, minmax(0, 1fr)); }
  .case-bar { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
@media (max-width: 640px) {
  .block-container { padding: .8rem 1rem 2.5rem; }
  .hero { flex-wrap: wrap; padding: 15px 16px; }
  .hero-tags { margin-left: 0; }
  .mini-grid, .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .steps { grid-template-columns: minmax(0, 1fr); }
}
"""

LOGO_SVG = """
<svg class="hero-logo" viewBox="0 0 64 64" role="img" aria-label="Breast Ultrasound AI logo">
  <rect width="64" height="64" rx="16" fill="#FFFFFF"/>
  <path d="M23 10h18v7a4 4 0 0 1-4 4H27a4 4 0 0 1-4-4z" fill="#0B2F4E"/>
  <g fill="none" stroke="#0F766E" stroke-linecap="round" stroke-width="3.6">
    <path d="M21 30a15 15 0 0 1 22 0"/><path d="M15 39a24 24 0 0 1 34 0"/>
  </g>
  <circle cx="32" cy="48" r="6.5" fill="#E11D5C"/><circle cx="32" cy="48" r="2.4" fill="#FFFFFF"/>
</svg>
"""

WAVE_SVG = """
<svg class="hero-wave" viewBox="0 0 430 220" aria-hidden="true">
  <g fill="none" stroke="#FFFFFF" stroke-width="1.6">
    <path d="M165 45a120 120 0 0 1 190 0"/><path d="M135 78a160 160 0 0 1 250 0"/>
    <path d="M105 111a200 200 0 0 1 310 0"/><path d="M75 144a240 240 0 0 1 370 0"/>
    <path d="M45 177a280 280 0 0 1 430 0"/><path d="M15 210a320 320 0 0 1 490 0"/>
  </g>
</svg>
"""

SHIELD_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#0B2F4E" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 3l7 3v5c0 4.6-2.9 8.2-7 10-4.1-1.8-7-5.4-7-10V6z"/><path d="M9 12l2 2 4-4"/>
</svg>
"""

ALERT_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#8A5A00" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 4l9 16H3z"/><path d="M12 10v4"/><path d="M12 17h.01"/>
</svg>
"""

PIN_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#0F766E" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 21s7-6 7-11a7 7 0 0 0-14 0c0 5 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>
</svg>
"""

RULER_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#0F766E" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <path d="M3 16L16 3l5 5L8 21z"/><path d="M7 12l2 2"/><path d="M10 9l2 2"/><path d="M13 6l2 2"/>
</svg>
"""

SHAPE_ICON = """
<svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke="#0F766E" stroke-width="2"
     stroke-linecap="round" stroke-linejoin="round">
  <path d="M6 7c3-4 9-4 12-1s3 7 1 10-7 5-11 3-6-8-2-12z"/>
</svg>
"""

SCAN_ICON = """
<svg viewBox="0 0 64 64" aria-hidden="true" fill="none" stroke="#0F766E" stroke-width="3" stroke-linecap="round">
  <path d="M10 22V14a4 4 0 0 1 4-4h8"/><path d="M42 10h8a4 4 0 0 1 4 4v8"/>
  <path d="M54 42v8a4 4 0 0 1-4 4h-8"/><path d="M22 54h-8a4 4 0 0 1-4-4v-8"/>
  <path d="M22 31a14 14 0 0 1 20 0"/><path d="M27 38a7 7 0 0 1 10 0"/>
</svg>
"""


UPLOAD_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M12 16V5"/><path d="M7.5 10L12 5l4.5 5"/><path d="M4 16v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/>
</svg>
"""

LIBRARY_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 15a6 6 0 0 1 10 0"/><path d="M10 12a3 3 0 0 1 4 0"/>
</svg>
"""

CHART_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <path d="M4 20V4"/><path d="M4 20h16"/><path d="M8 16v-4"/><path d="M13 16V8"/><path d="M18 16v-6"/>
</svg>
"""

FLOW_ICON = """
<svg viewBox="0 0 24 24" fill="none" stroke="#0F766E" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
  <circle cx="5" cy="6" r="2"/><circle cx="19" cy="6" r="2"/><circle cx="12" cy="18" r="2"/>
  <path d="M7 6h10"/><path d="M6 8l5 8"/><path d="M18 8l-5 8"/>
</svg>
"""


def page_html(content):
    """Add custom HTML to the page."""
    st.html(content)


def svg_image(svg, css_class='', alt=''):
    """An SVG graphic as an <img>, because st.html removes inline <svg> elements."""
    svg = svg.strip().replace('<svg ', '<svg xmlns="http://www.w3.org/2000/svg" ', 1)
    encoded = base64.b64encode(svg.encode()).decode()
    return f'<img class="{css_class}" src="data:image/svg+xml;base64,{encoded}" alt="{html.escape(alt)}">'


def show_header(config):
    page_html(f'<style>{PAGE_CSS}</style>')
    page_html(f"""
    <header class="hero">
      {svg_image(WAVE_SVG, 'hero-wave')}{svg_image(LOGO_SVG, 'hero-logo', 'Breast Ultrasound AI logo')}
      <div class="hero-text">
        <h1>Breast Ultrasound AI</h1>
        <p>Lesion detection, position and size for image review</p>
      </div>
      <div class="hero-tags">
        <div class="hero-tag">Research prototype</div>
        <div class="hero-tag">Model {MODEL_VERSION}: {len(config['model_files'])}-model ensemble</div>
      </div>
    </header>
    <div class="notice disclaimer" role="note">
      {svg_image(SHIELD_ICON)}
      <div><strong>Research use only. Not a medical device.</strong>
      <span>Do not use results to diagnose, rule out disease or make treatment decisions.
      Findings require review by a qualified clinician.</span></div>
    </div>
    """)


def section_title(text, number=None):
    badge = f'<b>{number}</b>' if number else ''
    page_html(f'<h2 class="section-title">{badge}{html.escape(text)}</h2>')


def panel_heading(icon, title, detail, active=False):
    tag = '<span class="active-tag">Showing</span>' if active else ''
    page_html(f"""
    <div class="panel-head">{svg_image(icon)}
      <div><h3>{html.escape(title)}</h3><p>{html.escape(detail)}</p></div>{tag}
    </div>""")


# ------------------------------------------------------------------
# Load the saved models and settings (only once per app start)
# ------------------------------------------------------------------

@st.cache_resource
def load_settings():
    """Portal settings. Returns (config, shape rule, test metrics, model checksums)."""
    return tuple(json.loads((ASSET_DIR / name).read_text())
                 for name in ['config.json', 'shape_rule.json', 'metrics.json', 'model_checksums.json'])


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url, target, expected_sha256):
    """Download a file and keep it only if its checksum is correct."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + f'.{os.getpid()}.{threading.get_ident()}.part')
    try:
        with urllib.request.urlopen(url, timeout=120) as response, open(partial, 'wb') as file:
            while chunk := response.read(1024 * 1024):
                file.write(chunk)
        if file_sha256(partial) != expected_sha256:
            raise ValueError(f'Checksum mismatch for {target.name}')
        os.replace(partial, target)
    finally:
        partial.unlink(missing_ok=True)


@st.cache_resource(show_spinner='Preparing the 5 models. The first start can take a few minutes...')
def prepare_model_files():
    """Find the model files on this computer, or download and verify them. Returns their paths."""
    config, _, _, checksums = load_settings()
    paths = []
    for name in config['model_files']:
        local = MODEL_DIR / name
        if local.exists():
            paths.append(local)
            continue
        target = DOWNLOAD_DIR / name
        if not target.exists() or file_sha256(target) != checksums[name]:
            download_file(f'{MODEL_RELEASE_URL}/{name}', target, checksums[name])
        paths.append(target)
    return paths


@st.cache_resource
def prediction_lock():
    """Only one scan is analysed at a time, because the models are loaded in turn to save memory."""
    return threading.Lock()


def predict_with_models(x, config):
    """Run the 5 models in turn. Only one model is in memory at a time."""
    head = {'classes': len(config['classes']), 'dropout': config['classification_dropout']}
    lesion_maps, class_scores = [], []
    with prediction_lock():
        model = smp.Unet(encoder_name=config['encoder'], encoder_weights=None,
                         in_channels=3, classes=1, aux_params=head)
        model.eval()
        with torch.no_grad():
            for path in prepare_model_files():
                model.load_state_dict(torch.load(path, map_location='cpu', mmap=True, weights_only=True))
                mask_logits, class_logits = model(x)
                lesion_maps.append(torch.sigmoid(mask_logits)[0, 0].numpy())
                class_scores.append(torch.softmax(class_logits, dim=1)[0].numpy())
    return lesion_maps, class_scores


# ------------------------------------------------------------------
# From the model output to a clean mask (same as 03_model_unet)
# ------------------------------------------------------------------

def to_model_input(image, config):
    """Grayscale PIL image -> normalised tensor (1, 3, size, size), padded to a square with black."""
    size = config['image_size']
    padded = ImageOps.pad(image, (size, size), method=Image.BICUBIC, color=0)
    x = torch.from_numpy(np.array(padded)).float().div(255).unsqueeze(0).repeat(3, 1, 1)
    mean = torch.tensor(config['normalize_mean']).view(3, 1, 1)
    std = torch.tensor(config['normalize_std']).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0)


def unpad_probs(probs, width, height, size):
    """Cut the black borders off a probability map and enlarge it smoothly to the original size."""
    scale = size / max(width, height)
    new_width, new_height = round(width * scale), round(height * scale)
    left, top = round((size - new_width) / 2), round((size - new_height) / 2)
    cropped = probs[top:top + new_height, left:left + new_width].astype(np.float32)
    return np.array(Image.fromarray(cropped).resize((width, height), Image.BILINEAR))


def clean_mask(mask, config):
    """Fill holes inside lesions and drop regions that are too small."""
    filled = ndimage.binary_fill_holes(mask)
    labelled, n_regions = ndimage.label(filled, structure=np.ones((3, 3)))
    sizes = ndimage.sum(filled, labelled, range(1, n_regions + 1))
    min_size = max(config['min_lesion_pixels'], mask.size * config['min_region_pct'] / 100)
    keep = [number for number, size in enumerate(sizes, start=1) if size >= min_size]
    return np.isin(labelled, keep)


# ------------------------------------------------------------------
# Measure each lesion (same as 04_measure, with the predicted-outline shape rule)
# ------------------------------------------------------------------

def find_lesions(mask, min_pixels):
    """Split a mask into separate lesions, largest first: a list of skimage regions."""
    regions = [r for r in regionprops(label(mask, connectivity=2)) if r.area >= min_pixels]
    regions.sort(key=lambda r: r.area, reverse=True)
    return regions


def zone(fraction, names):
    """Which third of the image a 0..1 position falls in."""
    return names[min(int(fraction * 3), 2)]


def diameter_line(region):
    """End points ((x1, y1), (x2, y2)) of the longest straight line across a lesion, for drawing."""
    top, left, _, _ = region.bbox
    edge = region.image & ~ndimage.binary_erosion(region.image)
    points = np.argwhere(edge) + [top, left]
    try:
        points = points[ConvexHull(points).vertices]          # the farthest points are always on the hull
    except QhullError:
        pass                                                  # very thin lesion: use all edge points
    distances = squareform(pdist(points))
    first, second = np.unravel_index(distances.argmax(), distances.shape)
    (y1, x1), (y2, x2) = points[first], points[second]
    return (float(x1), float(y1)), (float(x2), float(y2))


def outline_points(region):
    """The lesion outline scaled to fit a 100 x 100 box, as an SVG points string."""
    padded = np.pad(region.image, 1).astype(float)
    outline = approximate_polygon(max(find_contours(padded, 0.5), key=len), tolerance=0.6)
    rows, cols = outline[:, 0], outline[:, 1]
    scale = 84 / max(np.ptp(rows), np.ptp(cols), 1)
    x = 50 + (cols - (cols.min() + cols.max()) / 2) * scale
    y = 50 + (rows - (rows.min() + rows.max()) / 2) * scale
    return ' '.join(f'{a:.1f},{b:.1f}' for a, b in zip(x, y))


def measure_lesion(number, region, width, height, rule):
    """Size, position and shape of one predicted lesion."""
    top, left, bottom, right = region.bbox
    centre_y, centre_x = region.centroid
    box_width, box_height = right - left, bottom - top

    convex_perimeter = perimeter(region.image_convex)
    roughness = region.perimeter / convex_perimeter if convex_perimeter else 1.0
    axis_ratio = region.axis_minor_length / region.axis_major_length if region.axis_major_length else 1.0

    if region.solidity < rule['pred_irregular_solidity'] or roughness > rule['pred_irregular_roughness']:
        shape = 'irregular'
    else:
        shape = 'round' if axis_ratio >= rule['round_axis_ratio'] else 'oval'

    return {
        'number': number,
        'area_px': int(region.area),
        'area_pct': round(region.area / (width * height) * 100, 2),
        'diameter_px': round(float(region.feret_diameter_max), 1),
        'centre_x': float(centre_x), 'centre_y': float(centre_y),
        'x_frac': float(centre_x / width), 'y_frac': float(centre_y / height),
        'side': zone(centre_x / width, ['left', 'centre', 'right']),
        'depth_zone': zone(centre_y / height, ['shallow', 'mid', 'deep']),
        'depth_px': int(top),
        'shape': shape,
        'orientation': 'taller than wide' if box_height > box_width else 'wider than tall',
        'box': (int(left), int(top), int(right), int(bottom)),
        'diameter_line': diameter_line(region),
        'outline': outline_points(region),
    }


# ------------------------------------------------------------------
# Analyse one scan (the result is remembered, so the page stays fast)
# ------------------------------------------------------------------

@st.cache_data(show_spinner='Analysing the scan...', max_entries=50)
def analyse(image_bytes):
    """Run all 5 models on one scan. Returns the class result, the final mask and the lesions."""
    config, rule, _, _ = load_settings()
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    width, height = image.size
    x = to_model_input(image, config)

    lesion_maps, class_scores = predict_with_models(x, config)

    # Average the 5 models
    lesion_probs = np.mean(lesion_maps, axis=0)
    probabilities = np.mean(class_scores, axis=0)

    # Mask at the original image size
    enlarged = unpad_probs(lesion_probs, width, height, config['image_size'])
    mask = clean_mask(enlarged > config['mask_threshold'], config)

    # Malignant if its probability reaches the alert level, otherwise the more likely of normal / benign
    if probabilities[2] >= config['malignant_threshold']:
        predicted = 2
    else:
        predicted = int(probabilities[:2].argmax())

    lesions = [measure_lesion(number, region, width, height, rule)
               for number, region in enumerate(find_lesions(mask, config['min_lesion_pixels']), start=1)]

    return {
        'predicted': config['classes'][predicted],
        'probabilities': dict(zip(config['classes'], probabilities.tolist())),
        'mask': mask,
        'lesions': lesions,
    }


# ------------------------------------------------------------------
# The annotated scan
# ------------------------------------------------------------------

def scale_mask(mask, size):
    """Enlarge a True/False mask smoothly to size (width, height)."""
    image = Image.fromarray(mask.astype(np.uint8) * 255).resize(size, Image.BILINEAR)
    return np.array(image) > 127


def outline_of(mask, width):
    """The band of pixels along the inside edge of a mask."""
    return mask & ~ndimage.binary_erosion(mask, iterations=width)


def draw_label(draw, text, x, y, canvas_size, font, pad):
    """A filled label above position (x, y), kept inside the image."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    box_width, box_height = right - left + 2 * pad, bottom - top + 2 * pad
    x = min(max(x, 0), canvas_size[0] - box_width)
    y = y - box_height - pad
    if y < 0:                                     # no room above the lesion: move the label inside the image
        y = pad
    draw.rounded_rectangle((x, y, x + box_width, y + box_height), radius=pad, fill=LESION_COLOUR)
    draw.text((x + pad - left, y + pad - top), text, fill='white', font=font)


@st.cache_data(show_spinner=False, max_entries=50)
def annotated_scan(image_bytes, true_mask_paths=None):
    """The scan with highlighted lesions, centre markers, diameter lines and labels."""
    result = analyse(image_bytes)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')
    scale = RENDER_SIDE / max(image.size)
    size = (round(image.width * scale), round(image.height * scale))
    canvas = np.array(image.resize(size, Image.LANCZOS).convert('RGB')).astype(np.float32)
    line_width = max(2, round(RENDER_SIDE / 360))

    if result['lesions']:
        predicted = scale_mask(result['mask'], size)
        canvas[predicted] = canvas[predicted] * 0.88 + np.array(LESION_COLOUR) * 0.12     # light fill
        canvas[outline_of(predicted, line_width)] = LESION_COLOUR

    if true_mask_paths is not None:                  # drawn last, so the lesion fill never hides it
        true_mask = load_true_mask(true_mask_paths)
        if true_mask.any():
            canvas[outline_of(scale_mask(true_mask, size), line_width)] = TRUE_COLOUR

    picture = Image.fromarray(canvas.astype(np.uint8))
    draw = ImageDraw.Draw(picture)
    font = ImageFont.load_default(size=round(RENDER_SIDE / 42))
    marker = round(RENDER_SIDE / 90)
    for lesion in result['lesions']:
        (x1, y1), (x2, y2) = lesion['diameter_line']
        draw.line([(x1 * scale, y1 * scale), (x2 * scale, y2 * scale)], fill=DIAMETER_COLOUR, width=line_width)
        for x, y in [(x1 * scale, y1 * scale), (x2 * scale, y2 * scale)]:
            draw.ellipse((x - marker / 2, y - marker / 2, x + marker / 2, y + marker / 2), fill=DIAMETER_COLOUR)

        x, y = lesion['centre_x'] * scale, lesion['centre_y'] * scale
        draw.line([(x - marker, y), (x + marker, y)], fill='white', width=line_width)
        draw.line([(x, y - marker), (x, y + marker)], fill='white', width=line_width)

        left, top, _, _ = lesion['box']
        draw_label(draw, f"Lesion {lesion['number']}  |  {lesion['diameter_px']:.0f} px",
                   left * scale, top * scale, size, font, round(RENDER_SIDE / 130))
    return np.array(picture)


# ------------------------------------------------------------------
# Choosing a scan
# ------------------------------------------------------------------

@st.cache_data
def load_test_scans():
    """The test images: never used for training the models."""
    return pd.read_csv(ASSET_DIR / 'test_manifest.csv')


@st.cache_data
def available_examples():
    """Test images whose files are present. Deployments can run without the dataset."""
    scans = load_test_scans()
    return scans[[(DATA_DIR / path).exists() for path in scans['image_path']]].reset_index(drop=True)


@st.cache_data(max_entries=50)
def load_true_mask(mask_paths):
    """Merge the hand-drawn masks of a test image into one True/False array."""
    masks = [np.array(Image.open(DATA_DIR / path).convert('L')) > 127 for path in mask_paths.split(';')]
    return np.any(masks, axis=0)


def use_upload():
    """The upload box changed: show the new file, or fall back to the example if it was removed."""
    if st.session_state.get('upload') is not None:
        st.session_state['active_source'] = 'upload'
    elif st.session_state.get('example_index') is not None:
        st.session_state['active_source'] = 'example'
    else:
        st.session_state['active_source'] = None


def use_example():
    """The example list changed: show that example."""
    if st.session_state.get('example_index') is not None:
        st.session_state['active_source'] = 'example'
    elif st.session_state.get('upload') is not None:
        st.session_state['active_source'] = 'upload'
    else:
        st.session_state['active_source'] = None


def choose_scan():
    """Two separate ways to pick a scan. Returns a dict describing the scan being shown, or None."""
    active = st.session_state.setdefault('active_source', None)
    scans = available_examples()
    if active:
        page_html(f'<style>.st-key-{active}_panel {{ border-color: #0F766E !important; '
                  'box-shadow: 0 0 0 1px #0F766E; }</style>')

    upload_column, example_column = st.columns(2, gap='small')
    with upload_column.container(border=True, key='upload_panel', height='stretch'):
        panel_heading(UPLOAD_ICON, 'Upload a scan', 'PNG or JPEG breast ultrasound image', active == 'upload')
        upload = st.file_uploader('Breast ultrasound image', type=['png', 'jpg', 'jpeg'], key='upload',
                                  on_change=use_upload, label_visibility='collapsed')
        page_html('<p class="input-note">Do not upload identifiable patient data. Remove identifiers first. Scans '
                  'are processed by the server running this app and are not saved by the portal.</p>')
    with example_column.container(border=True, key='example_panel', height='stretch'):
        panel_heading(LIBRARY_ICON, 'Try an example', 'Test-set image with its hand-drawn reference',
                      active == 'example')
        index = st.selectbox('Example from the test set', scans.index, index=None, key='example_index',
                             placeholder='Select a test image' if len(scans) else 'Examples are not included here',
                             disabled=scans.empty, on_change=use_example,
                             format_func=lambda i: Path(scans.loc[i, 'image_path']).stem,
                             label_visibility='collapsed')
        page_html(f'<p class="input-note"><strong>Source:</strong> <a href="{DATASET_URL}" target="_blank" '
                  'rel="noopener noreferrer">Dataset of breast ultrasound images</a> (Al-Dhabyani et al., '
                  '<em>Data in Brief</em>, 2020). These test images were not used to train the models.</p>')

    if active == 'upload' and upload is not None:
        return {'bytes': upload.getvalue(), 'row': None, 'name': upload.name, 'source': 'Uploaded scan'}
    if active == 'example' and index is not None:
        row = scans.loc[index]
        return {'bytes': (DATA_DIR / row['image_path']).read_bytes(), 'row': row,
                'name': Path(row['image_path']).stem, 'source': 'BUSI test set'}
    return None


def scan_is_usable(image_bytes):
    """Check the file, showing a message if it can't be used."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        st.error('This file could not be read as an image. Please use a PNG or JPEG.')
        return False
    if max(image.size) > MAX_IMAGE_SIDE:
        st.error(f'This image is too large. The longest side can be at most {MAX_IMAGE_SIDE} pixels.')
        return False
    return True


# ------------------------------------------------------------------
# Result graphics
# ------------------------------------------------------------------

def show_notice(title, text):
    page_html(f"""
    <div class="notice warning" role="alert">{svg_image(ALERT_ICON)}
      <div><strong>{html.escape(title)}</strong><span>{html.escape(text)}</span></div>
    </div>""")


def show_consistency_notice(result):
    """Warn when the class and the outline disagree: these results need the most care."""
    has_lesion = bool(result['lesions'])
    if result['predicted'] != 'normal' and not has_lesion:
        show_notice('Tumour pattern suspected, but the location could not be determined.',
                    'Do not read this scan as clear. No outline is drawn because the model could not place '
                    'the lesion reliably.')
    elif result['predicted'] == 'normal' and has_lesion:
        show_notice('The class and the outline disagree.',
                    'The model says normal, but it outlined a possible lesion. The outlined area should '
                    'still be looked at.')


def ring_svg(value, colour):
    """Circular gauge for a 0..1 value."""
    svg = f"""
    <svg viewBox="0 0 100 100">
      <circle cx="50" cy="50" r="41" fill="none" stroke="#E4EBF2" stroke-width="11"/>
      <circle cx="50" cy="50" r="41" fill="none" stroke="{colour}" stroke-width="11" stroke-linecap="round"
              pathLength="100" stroke-dasharray="{value * 100:.2f} 100" transform="rotate(-90 50 50)"/>
      <text x="50" y="57" text-anchor="middle" font-family="Arial, sans-serif" font-size="17"
            font-weight="700" fill="#15293D">{value:.1%}</text>
    </svg>"""
    return svg_image(svg, 'ring', f'{value:.1%}')


def result_card(result, config, test_row, true_mask):
    """The predicted class, the three probabilities and the number of lesions."""
    predicted = result['predicted']
    probabilities = result['probabilities']
    alert_level = config['malignant_threshold']
    colours = {'normal': '#17694D', 'benign': '#145A86', 'malignant': '#B42318'}
    names = {'normal': 'No tumour pattern', 'benign': 'Benign pattern', 'malignant': 'Malignant pattern'}

    if predicted == 'malignant':
        explanation = (f"Flagged for review because the malignant probability reached the {alert_level:.0%} alert "
                       'level. The level is deliberately low so fewer cancers are missed, so some flagged scans '
                       'are not cancer.')
    elif predicted == 'benign':
        explanation = 'The image pattern is closest to benign scans in the training data.'
    elif result['lesions']:
        explanation = 'The image pattern is closest to normal scans in the training data.'
    else:
        explanation = 'No lesion detected. This does not rule out disease.'

    bars = ''
    for name, probability in probabilities.items():
        tick = note = ''
        if name == 'malignant':
            tick = f'<b style="left:{alert_level:.1%}"></b>'
            note = f'<div class="bar-note">Black mark: {alert_level:.0%} alert level</div>'
        bars += f"""
        <div class="prob">
          <div class="prob-head"><span>{html.escape(name.capitalize())}</span><span>{probability:.1%}</span></div>
          <div class="bar"><i style="width:{probability:.1%};background:{colours[name]}"></i>{tick}</div>{note}
        </div>"""

    facts = f'<div class="fact"><span>Lesions outlined</span><strong>{len(result["lesions"])}</strong></div>'
    if test_row is not None:
        reference = html.escape(str(test_row['label']).capitalize())
        facts += f'<div class="fact"><span>Reference class</span><strong>{reference}</strong></div>'
        if test_row['label'] != 'normal':
            overlap = (result['mask'] & true_mask).sum()
            dice = 2 * overlap / max(result['mask'].sum() + true_mask.sum(), 1)
            facts += f'<div class="fact"><span>Overlap with reference (Dice)</span><strong>{dice:.2f}</strong></div>'

    return f"""
    <section class="card" aria-label="AI finding">
      <div class="panel-head"><h3>AI finding</h3></div>
      <div class="status">
        {ring_svg(probabilities[predicted], colours[predicted])}
        <div><div class="status-label">Predicted class probability</div>
        <div class="status-name tone-{predicted}">{names[predicted]}</div></div>
      </div>
      <div class="explain {predicted}">{html.escape(explanation)}</div>
      {bars}{facts}
    </section>"""


def position_svg(lesion):
    """Image-shaped map with thirds and a marker at the lesion centre."""
    x, y = 6 + lesion['x_frac'] * 108, 6 + lesion['y_frac'] * 78
    lines = ''.join(f'<line x1="{6 + 36 * i}" y1="6" x2="{6 + 36 * i}" y2="84"/>' for i in (1, 2))
    lines += ''.join(f'<line x1="6" y1="{6 + 26 * i}" x2="114" y2="{6 + 26 * i}"/>' for i in (1, 2))
    label_text = f"Lesion centre: {lesion['side']}, {lesion['depth_zone']}"
    svg = f"""
    <svg viewBox="0 0 120 90">
      <rect x="6" y="6" width="108" height="78" rx="6" fill="#E6F1F8" stroke="#9DBDD2"/>
      <g stroke="#B9D0E0" stroke-dasharray="3 3">{lines}</g>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="#E11D5C" opacity=".2"/>
      <circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#E11D5C" stroke="#FFFFFF" stroke-width="2"/>
    </svg>"""
    return svg_image(svg, alt=label_text)


def lesion_cards(lesion):
    """Position, size and shape cards for one lesion."""
    side = lesion['side'].capitalize()
    size_width = min(lesion['area_pct'] / 25, 1)
    outline_svg = f"""
    <svg viewBox="0 0 100 100">
      <polygon points="{lesion['outline']}" fill="#FCE4EC" stroke="#E11D5C" stroke-width="2.5" stroke-linejoin="round"/>
    </svg>"""
    return f"""
    <div class="lesion-grid">
      <section class="card metric-card">
        <h3>{svg_image(PIN_ICON)}Position</h3>
        <div class="metric-body">
          <div class="metric-graphic">{position_svg(lesion)}</div>
          <div><div class="big">{side}, {lesion['depth_zone']}</div>
          <div class="small">{lesion['x_frac']:.0%} across, {lesion['y_frac']:.0%} down the image.
          Starts {lesion['depth_px']} px from the top.</div></div>
        </div>
      </section>
      <section class="card metric-card">
        <h3>{svg_image(RULER_ICON)}Size</h3>
        <div class="big" style="margin-top:12px">{lesion['diameter_px']:.0f} px longest diameter</div>
        <div class="small">Area {lesion['area_px']:,} px, {lesion['area_pct']:.2f}% of the image</div>
        <div class="size-bar"><div class="bar"><i style="width:{size_width:.1%};background:#0F766E"></i></div>
        <div class="bar-note">Share of image, bar full at 25%</div></div>
      </section>
      <section class="card metric-card">
        <h3>{svg_image(SHAPE_ICON)}Shape</h3>
        <div class="metric-body">
          <div class="metric-graphic">{svg_image(outline_svg, alt='Predicted lesion outline')}</div>
          <div><div class="big">{lesion['shape'].capitalize()}</div>
          <div class="small">{lesion['orientation'].capitalize()}. Based on the predicted outline, so approximate.</div></div>
        </div>
      </section>
    </div>"""


def lesion_kpis(lesion):
    """The full measurements of one lesion as KPI boxes."""
    boxes = [
        ('Longest diameter', f"{lesion['diameter_px']:.1f} px"),
        ('Area', f"{lesion['area_px']:,} px"),
        ('Area of image', f"{lesion['area_pct']:.2f}%"),
        ('Centre X', f"{lesion['centre_x']:.0f} px"),
        ('Centre Y', f"{lesion['centre_y']:.0f} px"),
        ('Side of image', lesion['side'].capitalize()),
        ('Depth zone', lesion['depth_zone'].capitalize()),
        ('Depth from top', f"{lesion['depth_px']} px"),
        ('Shape', lesion['shape'].capitalize()),
        ('Orientation', lesion['orientation'].capitalize()),
    ]
    cards = ''.join(f'<div class="kpi"><span>{name}</span><strong>{value}</strong></div>' for name, value in boxes)
    return f"""
    <div class="kpi-title"><h3>Lesion {lesion['number']} measurements</h3>
    <span>Pixels are not calibrated to millimetres</span></div>
    <div class="kpi-grid">{cards}</div>"""


def overview_html(metrics, config):
    """Model performance and how the model works, always visible at the top of the page."""
    scans = load_test_scans()['label'].value_counts()
    ranges = metrics['confidence_95']
    tiles = [
        ('malignant_recall', 'Malignancy sensitivity', True),
        ('malignant_precision', 'Malignancy PPV', True),
        ('accuracy', '3-class accuracy', True),
        ('dice_tumour_images', 'Outline overlap (Dice)', False),
        ('normal_images_with_false_lesion', 'False lesion on normal scans', True),
        ('tumour_images_with_no_lesion_found', 'Tumour scans with no outline', True),
    ]
    cards = ''
    for key, name, is_percent in tiles:
        low, high = ranges[key]
        if is_percent:
            value, span = f'{metrics[key]:.1%}', f'{low:.1%} to {high:.1%}'
        else:
            value, span = f'{metrics[key]:.3f}', f'{low:.3f} to {high:.3f}'
        cards += (f'<div class="mini"><div class="value">{value}</div><div class="name">{name}</div>'
                  f'<div class="range">95% range: {span}</div></div>')

    steps = [
        ('Prepare', f"Grayscale, padded to {config['image_size']} x {config['image_size']} px"),
        ('Predict', f"{len(config['model_files'])} U-Net models estimate lesion area and class"),
        ('Combine', 'Outputs averaged, specks removed, holes filled'),
        ('Measure', 'Position, size and shape from the final outline'),
    ]
    step_cards = ''.join(f'<div class="step"><b>{number}</b><div><strong>{title}</strong><span>{text}</span></div></div>'
                         for number, (title, text) in enumerate(steps, start=1))
    return f"""
    <div class="overview">
      <section class="card" aria-label="Model performance">
        <div class="panel-head">{svg_image(CHART_ICON)}
          <div><h2>Model performance</h2><p>Held-out test set, {metrics['test_images']} images</p></div></div>
        <div class="mini-grid">{cards}</div>
        <p class="caveat">{scans['malignant']} malignant, {scans['benign']} benign and {scans['normal']} normal scans
        from one public dataset (BUSI). Not externally validated. PPV depends on how common cancer is in the
        population being scanned.</p>
      </section>
      <section class="card" aria-label="How it works">
        <div class="panel-head">{svg_image(FLOW_ICON)}
          <div><h2>How it works</h2><p>From image to AI finding</p></div></div>
        <div class="steps">{step_cards}</div>
        <p class="caveat">Trained on breast ultrasound only. Sizes are in pixels and positions are within the image,
        not the breast. Shape labels are approximate and are not BI-RADS categories. Some tumours cannot be located:
        the page then says so and draws no outline.</p>
      </section>
    </div>"""


# ------------------------------------------------------------------
# The page
# ------------------------------------------------------------------

def show_case_bar(scan, config):
    image = Image.open(io.BytesIO(scan['bytes']))
    items = [('Scan', scan['name']), ('Source', scan['source']),
             ('Image size', f'{image.width} x {image.height} px'),
             ('Model', f"{MODEL_VERSION}, trained {config['trained_on'][:10]}")]
    cells = ''.join(f'<div><span>{name}</span><strong>{html.escape(str(value))}</strong></div>' for name, value in items)
    page_html(f'<div class="card case-bar">{cells}</div>')


def show_results(scan, config):
    image_bytes, test_row = scan['bytes'], scan['row']
    result = analyse(image_bytes)
    true_mask = load_true_mask(test_row['mask_paths']) if test_row is not None else None
    lesions = result['lesions']

    section_title('Review the AI result', 2)
    show_case_bar(scan, config)
    show_consistency_notice(result)

    scan_column, result_column = st.columns([1.55, 1], gap='small')
    with scan_column:
        view = st.radio('Scan view', ['Annotated', 'Original'], horizontal=True, label_visibility='collapsed')
        if view == 'Annotated':
            mask_paths = test_row['mask_paths'] if test_row is not None else None
            st.image(annotated_scan(image_bytes, mask_paths), width='stretch', output_format='PNG')
            reference = '<span><i style="background:#22C55E"></i>Hand-drawn reference</span>' if test_row is not None else ''
            page_html(f"""
            <div class="legend">
              <span><i style="background:#E11D5C"></i>Predicted lesion</span>
              <span><i style="background:#0891B2"></i>Longest diameter</span>
              <span><strong>+</strong>Lesion centre</span>{reference}
            </div>""")
        else:
            st.image(Image.open(io.BytesIO(image_bytes)).convert('L'), width='stretch')
    with result_column:
        page_html(result_card(result, config, test_row, true_mask))

    section_title('Lesion details', 3)
    if not lesions:
        page_html('<div class="card">No lesion was outlined, so there are no measurements for this scan.</div>')
        return

    chosen = lesions[0]
    if len(lesions) > 1:
        number = st.radio('Lesion', [lesion['number'] for lesion in lesions], horizontal=True,
                          format_func=lambda n: f'Lesion {n}', label_visibility='collapsed')
        chosen = lesions[number - 1]
    page_html(lesion_cards(chosen))
    page_html(''.join(lesion_kpis(lesion) for lesion in lesions))


def main():
    st.set_page_config(page_title='Breast Ultrasound AI', page_icon=':material/monitor_heart:',
                       layout='wide', initial_sidebar_state='collapsed')
    config, _, metrics, _ = load_settings()

    show_header(config)
    page_html(overview_html(metrics, config))

    section_title('Select a scan', 1)
    scan = choose_scan()
    if scan is None:
        page_html(f"""
        <div class="card empty">{svg_image(SCAN_ICON)}
          <strong>No scan selected</strong>
          <span>Upload a breast ultrasound image, or choose a test-set example when examples are available.</span>
        </div>""")
        st.stop()

    if not scan_is_usable(scan['bytes']):
        st.stop()
    try:
        prepare_model_files()
    except (OSError, ValueError, urllib.error.URLError):
        st.error('The model files could not be loaded. Please try again later.')
        st.stop()
    show_results(scan, config)


main()
