import unicodedata
from typing import Dict, List, Tuple, Optional

def norm(text: str) -> str:
    """Text normalization: lowercase + remove whitespace/punctuation + keep alphanumerics only."""
    if not text:
        return ""
    text = unicodedata.normalize('NFKC', text)
    text = text.lower()
    text = ''.join(c for c in text if c.isalnum())
    return text

def poly_to_bbox(poly: List[List[float]]) -> Tuple[float, float, float, float]:
    """Convert polygon to bbox."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return (min(xs), min(ys), max(xs), max(ys))

def bbox_iou(box1: Tuple, box2: Tuple) -> float:
    """Compute IoU."""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
    xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    union = (x2_1 - x1_1) * (y2_1 - y1_1) + (x2_2 - x1_2) * (y2_2 - y1_2) - inter
    return inter / union if union > 0 else 0.0

def point_in_bbox(point: Tuple[float, float], bbox: Tuple) -> bool:
    """Check whether a point is inside a bbox."""
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2

def bbox_center(bbox: Tuple) -> Tuple[float, float]:
    """Return the bbox center point."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def union_bboxes(bboxes: List[Tuple]) -> Tuple:
    """Merge multiple bboxes."""
    if not bboxes:
        return (0, 0, 0, 0)
    x1 = min(b[0] for b in bboxes)
    y1 = min(b[1] for b in bboxes)
    x2 = max(b[2] for b in bboxes)
    y2 = max(b[3] for b in bboxes)
    return (x1, y1, x2, y2)

def group_into_lines(items: List[Dict], y_threshold: float = 10.0) -> List[List[Dict]]:
    """Cluster OCR items into lines by y-axis distance."""
    if not items:
        return []
    sorted_items = sorted(items, key=lambda it: (it['bbox'][1] + it['bbox'][3]) / 2)
    lines = []
    current_line = [sorted_items[0]]
    
    for item in sorted_items[1:]:
        y_cur = (item['bbox'][1] + item['bbox'][3]) / 2
        y_prev = (current_line[-1]['bbox'][1] + current_line[-1]['bbox'][3]) / 2
        if abs(y_cur - y_prev) < y_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    lines.append(current_line)
    
    # Sort each line by x coordinate
    for line in lines:
        line.sort(key=lambda it: it['bbox'][0])
    return lines