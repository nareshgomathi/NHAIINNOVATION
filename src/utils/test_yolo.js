/**
 * faceDetectionAndRecognition.ts
 *
 * Combined YOLO face detection  +  MobileFaceNet face recognition
 * for React Native (react-native-fast-tflite).
 *
 * Pipeline:
 *   1. loadYOLOModel()      — load YOLOv8 face-detector
 *   2. loadFaceModel()      — load MobileFaceNet embedder
 *   3. loadGallery()        — parse gallery.json from ASSETS_DIR
 *   4. prepareYOLOImage()   — resize raw photo to 640×640
 *   5. detectFace()         — run YOLO, returns bbox in 640-px space
 *   6. cropFace()           — JPEG-crop the detected region
 *   7. recognizeFace()      — embed + cosine-match against gallery
 */

import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';

// ─── Shared constants ─────────────────────────────────────────────────────────

const ASSETS_DIR = `${RNFS.DocumentDirectoryPath}/FaceRecognition`;

// ─── YOLO constants ───────────────────────────────────────────────────────────

const YOLO_INPUT_SIZE  = 640;
const CONF_THRESHOLD   = 0.25;
const IOU_THRESHOLD    = 0.45;
const NUM_ANCHORS      = 8400;

// ─── MobileFaceNet constants ──────────────────────────────────────────────────

const FACE_INPUT_SIZE  = 112;
// const EMBEDDING_SIZE = 512;   // informational only — not used at runtime

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface BBox {
  x:     number;   // top-left x  (normalised 0–1 inside YOLO, then converted to px)
  y:     number;
  w:     number;
  h:     number;
  score: number;
}

/** Bounding-box in 640-pixel space, returned by detectFace() */
export interface DetectionResult {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Best gallery match returned by recognizeFace() */
export interface RecognitionResult {
  name:  string;
  score: number;   // cosine similarity in [-1, 1]; same person ≈ 0.6–0.9
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module-level singletons
// ═══════════════════════════════════════════════════════════════════════════════

let yoloModel: TensorflowModel | null = null;
let faceModel: TensorflowModel | null = null;
let gallery:   Record<string, number[]> | null = null;

// ═══════════════════════════════════════════════════════════════════════════════
// 1.  Model loading
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Load the YOLOv8 face-detection TFLite model.
 * Safe to call multiple times — skips if already loaded.
 */
export async function loadYOLOModel(): Promise<void> {
  if (yoloModel) {
    console.log('[YOLO] Model already loaded');
    return;
  }
  try {
    await ensureDir(ASSETS_DIR);
    console.log('[YOLO] Loading TFLite model…');
    yoloModel = await loadTensorflowModel(
      require('../../assets/model_float16.tflite'),
      [],
    );
    console.log('[YOLO] Model loaded successfully');
  } catch (err) {
    console.error('[YOLO] Model load error:', err);
    throw err;
  }
}

/**
 * Load the MobileFaceNet embedding TFLite model.
 * Safe to call multiple times — skips if already loaded.
 */
export async function loadFaceModel(): Promise<void> {
  if (faceModel) {
    console.log('[Face] Model already loaded');
    return;
  }
  try {
    console.log('[Face] Loading MobileFaceNet TFLite…');
    faceModel = await loadTensorflowModel(
      require('../../assets/mobilefacenet_float32.tflite'),
      [],
    );
    console.log('[Face] Model loaded successfully');
  } catch (err) {
    console.error('[Face] Model load error:', err);
    throw err;
  }
}

/**
 * Load gallery.json from ASSETS_DIR.
 * Expects { "PersonName": [512-d float array], … }
 * (already L2-normalised — the Python script does this before saving).
 */
export async function loadGallery(): Promise<void> {
  if (gallery) return;

  gallery = require('../../assets/gallery.json');

  console.log(
    `[Face] Gallery loaded - ${Object.keys(gallery).length} persons`
  );
}
// ═══════════════════════════════════════════════════════════════════════════════
// 2.  YOLO — image preparation
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Resize a raw photo URI to 640×640 (cover crop) so that both
 * detectFace() and cropFace() use the same pixel grid.
 */
export async function prepareYOLOImage(imageUri: string): Promise<string> {
  const resized = await ImageResizer.createResizedImage(
    imageUri,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    'JPEG',
    90,
    0,
    undefined,
    false,
    { mode: 'cover', onlyScaleDown: false },
  );
  return resized.uri;
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.  YOLO — detection
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Run YOLOv8 face detection on a 640×640 image URI.
 * Returns the best bounding-box in 640-pixel space, or null if no face found.
 *
 * @param imageUri  Must already be resized to 640×640 (use prepareYOLOImage).
 */
export async function detectFace(imageUri: string): Promise<DetectionResult | null> {
  if (!yoloModel) throw new Error('[YOLO] Model not loaded — call loadYOLOModel() first');

  const localPath = imageUri.replace('file://', '');
  const exists    = await RNFS.exists(localPath).catch(() => false);
  if (!exists) {
    console.warn('[YOLO] Image file not found:', localPath);
    return null;
  }

  // Build float32 tensor [1, 3, 640, 640] — HWC interleaved / 255
  const inputTensor = await yoloUriToFloat32Tensor(imageUri);

  // Inference
  const outputs = await yoloModel.run([inputTensor.buffer]);

  // Decode raw output → bounding boxes
  const boxes = decodeYOLOBoxes(new Float32Array(outputs[0]));
  if (boxes.length === 0) {
    console.log('[YOLO] No detections above threshold');
    return null;
  }

  // Non-maximum suppression
  const kept = nms(boxes, IOU_THRESHOLD);
  if (kept.length === 0) return null;

  // Highest-confidence box, convert normalised → 640-px coords
  const best = kept[0];
  const x    = Math.max(0, Math.round(best.x * YOLO_INPUT_SIZE));
  const y    = Math.max(0, Math.round(best.y * YOLO_INPUT_SIZE));
  const w    = Math.min(YOLO_INPUT_SIZE - x, Math.round(best.w * YOLO_INPUT_SIZE));
  const h    = Math.min(YOLO_INPUT_SIZE - y, Math.round(best.h * YOLO_INPUT_SIZE));

  console.log(`[YOLO] Detection — x:${x} y:${y} w:${w} h:${h} score:${best.score.toFixed(3)}`);
  return { x, y, w, h };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4.  YOLO — face crop
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Crop the detected face region from the 640×640 image and save it as JPEG.
 *
 * @param imageUri   The 640×640 URI (output of prepareYOLOImage).
 * @param detection  Bounding-box from detectFace() — in 640-px space.
 * @param fileName   Output filename (default: 'cropped_face.jpg').
 * @returns          file:// URI of the saved crop, or null on failure.
 */
export async function cropFace(
  imageUri:  string,
  detection: DetectionResult,
  fileName:  string = 'cropped_face.jpg',
): Promise<string | null> {
  try {
    // Re-decode the same 640×640 pixel buffer used during detection
    const resized = await ImageResizer.createResizedImage(
      imageUri,
      YOLO_INPUT_SIZE,
      YOLO_INPUT_SIZE,
      'JPEG',
      95,
      0,
      undefined,
      false,
      { mode: 'cover', onlyScaleDown: false },
    );

    const b64     = await RNFS.readFile(resized.uri, 'base64');
    const decoded = jpeg.decode(Buffer.from(b64, 'base64'), { useTArray: true });

    const { x, y, w, h } = detection;   // already in 0–640 pixel space
    if (w <= 0 || h <= 0) return null;

    // Clamp to image bounds
    const x0 = Math.max(0, x);
    const y0 = Math.max(0, y);
    const x1 = Math.min(YOLO_INPUT_SIZE, x + w);
    const y1 = Math.min(YOLO_INPUT_SIZE, y + h);
    const cw  = x1 - x0;
    const ch  = y1 - y0;

    if (cw <= 0 || ch <= 0) return null;

    const srcData  = decoded.data as Uint8Array;
    const cropData = new Uint8Array(cw * ch * 4);

    for (let row = 0; row < ch; row++) {
      for (let col = 0; col < cw; col++) {
        const srcIdx = ((y0 + row) * YOLO_INPUT_SIZE + (x0 + col)) * 4;
        const dstIdx = (row * cw + col) * 4;
        cropData[dstIdx]     = srcData[srcIdx];
        cropData[dstIdx + 1] = srcData[srcIdx + 1];
        cropData[dstIdx + 2] = srcData[srcIdx + 2];
        cropData[dstIdx + 3] = srcData[srcIdx + 3];
      }
    }

    const encoded = jpeg.encode({ data: cropData, width: cw, height: ch }, 92);

    await ensureDir(ASSETS_DIR);
    const outputPath = `${ASSETS_DIR}/${fileName}`;
    await RNFS.writeFile(
      outputPath,
      Buffer.from(encoded.data).toString('base64'),
      'base64',
    );
    console.log(`[YOLO] cropFace saved: ${outputPath} (${cw}×${ch})`);
    return `file://${outputPath}`;

  } catch (err: any) {
    console.error('[YOLO] cropFace FAILED:', err?.message ?? String(err));
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5.  MobileFaceNet — recognition
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Embed a cropped face and find the closest match in the gallery.
 *
 * @param croppedFaceUri  file:// URI produced by cropFace().
 * @returns               Best-matching { name, score } or null.
 */
export async function recognizeFace(
  croppedFaceUri: string,
): Promise<RecognitionResult | null> {
  if (!faceModel) throw new Error('[Face] Model not loaded — call loadFaceModel() first');
  if (!gallery)   throw new Error('[Face] Gallery not loaded — call loadGallery() first');

  // 1. Resize to 112×112, normalise, build tensor
  const tensor = await faceUriToFloat32Tensor(croppedFaceUri);

  // 2. Run inference → raw 512-d embedding
  const outputs = await faceModel.run([tensor.buffer]);
  const rawEmb  = new Float32Array(outputs[0]);

  // 3. L2-normalise  (mirrors F.normalize(emb, p=2, dim=1) in Python)
  const queryEmb = l2Normalize(rawEmb);
console.log(
  'Embedding size:',
  rawEmb.length
);
  // 4. Cosine search
  return cosineLookup(queryEmb);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal — YOLO tensor & decode helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Read a JPEG, resize to 640×640, return an RGB Float32Array
 * with values in [0, 1]  (pixel / 255).
 * Layout: HWC  →  [H × W × 3]
 */
async function yoloUriToFloat32Tensor(uri: string): Promise<Float32Array> {
  const resized = await ImageResizer.createResizedImage(
    uri,
    YOLO_INPUT_SIZE,
    YOLO_INPUT_SIZE,
    'JPEG',
    90,
    0,
    undefined,
    false,
    { mode: 'cover', onlyScaleDown: false },
  );

  const buf    = Buffer.from(await RNFS.readFile(resized.uri, 'base64'), 'base64');
  const pixels = jpeg.decode(buf, { useTArray: true }).data as Uint8Array;

  const numPixels = YOLO_INPUT_SIZE * YOLO_INPUT_SIZE;
  const tensor    = new Float32Array(numPixels * 3);

  for (let i = 0; i < numPixels; i++) {
    const base        = i * 4;
    tensor[i * 3 + 0] = pixels[base]     / 255.0;   // R
    tensor[i * 3 + 1] = pixels[base + 1] / 255.0;   // G
    tensor[i * 3 + 2] = pixels[base + 2] / 255.0;   // B
  }

  return tensor;
}

/**
 * Decode YOLOv8 raw output tensor.
 *
 * Expected layout (transposed from YOLO export):
 *   [cx, cy, w, h, conf]  stored as column-major across NUM_ANCHORS.
 *   i.e. index for field `f` at anchor `i` = f * NUM_ANCHORS + i
 *
 * All values are normalised to [0, 1] relative to the 640-px grid.
 */
function decodeYOLOBoxes(rawOutput: Float32Array): BBox[] {
  const boxes: BBox[] = [];

  for (let i = 0; i < NUM_ANCHORS; i++) {
    const cx   = rawOutput[0 * NUM_ANCHORS + i];
    const cy   = rawOutput[1 * NUM_ANCHORS + i];
    const w    = rawOutput[2 * NUM_ANCHORS + i];
    const h    = rawOutput[3 * NUM_ANCHORS + i];
    const conf = rawOutput[4 * NUM_ANCHORS + i];

    // Apply sigmoid only when the raw value is outside [0, 1]
    const score = (conf <= 0 || conf >= 1) ? sigmoid(conf) : conf;
    if (score < CONF_THRESHOLD) continue;

    boxes.push({
      x: cx - w / 2,   // convert centre → top-left
      y: cy - h / 2,
      w,
      h,
      score,
    });
  }

  console.log('[YOLO] Detections above threshold:', boxes.length);
  return boxes;
}

// ─── NMS ──────────────────────────────────────────────────────────────────────

function nms(boxes: BBox[], iouThresh: number): BBox[] {
  const sorted     = [...boxes].sort((a, b) => b.score - a.score);
  const suppressed = new Uint8Array(sorted.length);
  const kept: BBox[] = [];

  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    kept.push(sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (iou(sorted[i], sorted[j]) > iouThresh) suppressed[j] = 1;
    }
  }
  return kept;
}

function iou(a: BBox, b: BBox): number {
  const ax2    = a.x + a.w;
  const ay2    = a.y + a.h;
  const bx2    = b.x + b.w;
  const by2    = b.y + b.h;
  const interW = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const interH = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter  = interW * interH;
  if (inter === 0) return 0;
  const union  = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal — MobileFaceNet tensor & match helpers
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * Read a JPEG, resize to 112×112, return a normalised RGB Float32Array.
 *
 * Normalisation mirrors torchvision:
 *   Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
 *   ⟹  out = (pixel/255 − 0.5) / 0.5  =  pixel/255 × 2 − 1   ∈ [−1, 1]
 *
 * Layout: HWC  →  [112 × 112 × 3]
 * If your TFLite model was exported with CHW layout, uncomment Option B below.
 */
async function faceUriToFloat32Tensor(uri: string): Promise<Float32Array> {
  const resized = await ImageResizer.createResizedImage(
    uri,
    FACE_INPUT_SIZE,
    FACE_INPUT_SIZE,
    'JPEG',
    95,
    0,
    undefined,
    false,
    { mode: 'cover', onlyScaleDown: false },
  );

  const buf    = Buffer.from(await RNFS.readFile(resized.uri, 'base64'), 'base64');
  const pixels = jpeg.decode(buf, { useTArray: true }).data as Uint8Array;
  // pixels: [R, G, B, A,  R, G, B, A, …]  length = 112 × 112 × 4

  const numPixels = FACE_INPUT_SIZE * FACE_INPUT_SIZE;
  const tensor    = new Float32Array(numPixels * 3);

  // ── Option A: HWC (most TFLite models) ─────────────────────────────────────
  for (let i = 0; i < numPixels; i++) {
    const base        = i * 4;
tensor[i * 3 + 0] = (pixels[base]     - 127.5) / 128.0;
tensor[i * 3 + 1] = (pixels[base + 1] - 127.5) / 128.0;
tensor[i * 3 + 2] = (pixels[base + 2] - 127.5) / 128.0;
  }

  // ── Option B: CHW — uncomment if your model needs channel-first ────────────
  // for (let i = 0; i < numPixels; i++) {
  //   const base = i * 4;
  //   tensor[0 * numPixels + i] = (pixels[base]     / 255.0 - 0.5) / 0.5;  // R plane
  //   tensor[1 * numPixels + i] = (pixels[base + 1] / 255.0 - 0.5) / 0.5;  // G plane
  //   tensor[2 * numPixels + i] = (pixels[base + 2] / 255.0 - 0.5) / 0.5;  // B plane
  // }

  return tensor;
}

/** L2-normalise a vector in-place → new Float32Array. */
function l2Normalize(v: Float32Array): Float32Array {
  let norm = 0;
  for (let i = 0; i < v.length; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm);
  if (norm === 0) return v;
  const out = new Float32Array(v.length);
  for (let i = 0; i < v.length; i++) out[i] = v[i] / norm;
  return out;
}

/** Dot-product of two equal-length numeric arrays. */
function dotProduct(
  a: Float32Array | number[],
  b: Float32Array | number[],
): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/**
 * Linear scan through the gallery, return best cosine match.
 * Both query and gallery embeddings are L2-normalised, so
 *   cosine_similarity = dot_product.
 */function cosineLookup(query: Float32Array): RecognitionResult | null {
  if (!gallery) return null;

  let bestName = '';
  let bestScore = -Infinity;

  for (const [name, galleryEmb] of Object.entries(gallery)) {
    const score = dotProduct(query, galleryEmb);

    if (score > bestScore) {
      bestScore = score;
      bestName = name;
    }
  }

  if (!bestName) return null;

  

  console.log(
  `[Face] Best match: "${bestName}" cosine: ${bestScore.toFixed(4)}`
);

return {
  name: bestName,
  score: Number(bestScore.toFixed(4)),
};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Internal — shared utilities
// ═══════════════════════════════════════════════════════════════════════════════

async function ensureDir(path: string): Promise<void> {
  if (!(await RNFS.exists(path).catch(() => false))) {
    await RNFS.mkdir(path);
  }
}

function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

// ═══════════════════════════════════════════════════════════════════════════════
// Convenience — full pipeline in one call
// ═══════════════════════════════════════════════════════════════════════════════

/**
 * High-level helper: runs the entire detect → crop → recognise pipeline.
 *
 * @param rawPhotoUri  Any image URI (camera, gallery, etc.)
 * @param threshold    Cosine similarity threshold below which "Unknown" is returned.
 * @returns            { name, score } or null if no face was detected.
 */
export async function detectAndRecognize(
  rawPhotoUri: string,
  threshold:   number = 0.45,
): Promise<RecognitionResult | null> {
  // Step 1 — normalise to 640×640
  const resizedUri = await prepareYOLOImage(rawPhotoUri);

  // Step 2 — detect face
  const detection = await detectFace(resizedUri);
  if (!detection) {
    console.log('[Pipeline] No face detected');
    return null;
  }

  // Step 3 — crop face
  const croppedUri = await cropFace(resizedUri, detection, 'pipeline_face.jpg');
  if (!croppedUri) {
    console.log('[Pipeline] Crop failed');
    return null;
  }

  // Step 4 — recognise
  const result = await recognizeFace(croppedUri);
  if (!result) return null;

  if (result.score < threshold) {
    console.log(`[Pipeline] Score ${result.score.toFixed(4)} below threshold — Unknown`);
    return { name: 'Unknown', score: result.score };
  }

  return result;
}