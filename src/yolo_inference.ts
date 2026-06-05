import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';
import FaceDetection, { Face } from '@react-native-ml-kit/face-detection';

// ─── Keep in sync with your model's output classes ────────────────────────────
const CLASS_NAMES = [
  'ajay_devgn', 'akshay_kumar', 'amitabh_bachchan', 'chiranjeevi', 'govinda',
  'hrithik_roshan', 'kajol', 'kamal_haasan', 'kangana_ranaut', 'madhuri_dixit',
  'mammootty', 'mithun_chakraborty', 'mohanlal', 'prabhas', 'prakash_raj',
  'radhika_apte', 'ranbir_kapoor', 'rani_mukerji', 'ranveer_singh', 'saif_ali_khan',
  'salman_khan', 'sanjay_dutt', 'tabu', 'vidya_balan', 'waheeda_rehman',
];

const ASSETS_DIR           = `${RNFS.DocumentDirectoryPath}/FaceRecognition`;
const CLASSIFIER_INPUT_SIZE = 224;
const IMAGE_SIZE            = 640;

// ─── Liveness delta thresholds ────────────────────────────────────────────────
// A real expression change should produce at least this much delta.
// Raise if you get false-positives with printed photos.
const BLINK_DELTA_THRESHOLD = 0.15;   // eye-open probability drop (0–1)
const SMILE_DELTA_THRESHOLD = 0.20;   // smile probability rise (0–1)

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface DetectionResult {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** ML Kit facial-expression data extracted from one photo. */
export interface PhotoAnalysis {
  label:        'before' | 'after';   // "before smile/blink" or "after"
  uri:          string;               // resized URI used for detection
  leftEyeOpen:  number;               // 0–1
  rightEyeOpen: number;               // 0–1
  smilingProb:  number;               // 0–1
  detection:    DetectionResult | null;
}

/** Liveness verdict comparing the two photos. */
export interface LivenessResult {
  isLive:        boolean;
  blinkDetected: boolean;   // significant eye-open delta between photos
  smileDetected: boolean;   // significant smile delta between photos
  eyeDelta:      number;    // raw delta value (for debugging)
  smileDelta:    number;    // raw delta value (for debugging)
  before:        PhotoAnalysis;
  after:         PhotoAnalysis;
  /**
   * The cropped face URI that will be fed into the classifier.
   * We always prefer the AFTER photo (clearer expression, better for recognition)
   * but fall back to BEFORE if AFTER had no detection.
   */
  bestFaceUri:   string | null;
  bestDetection: DetectionResult | null;
}

export interface RecognitionResult {
  name:        string;
  score:       number;
  classId?:    number;
  croppedUri?: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Singleton model
// ═══════════════════════════════════════════════════════════════════════════════

let faceModel: TensorflowModel | null = null;

// ═══════════════════════════════════════════════════════════════════════════════
// 1. Load classifier
// ═══════════════════════════════════════════════════════════════════════════════

export async function loadFaceModel(): Promise<void> {
  if (faceModel) {
    console.log('[Face] Model already loaded');
    return;
  }
  try {
    console.log('[Face] Loading classifier TFLite…');
    faceModel = await loadTensorflowModel(
      require('../../assets/classify_float32.tflite'),
      [],
    );
    console.log('[Face] Model loaded successfully');
  } catch (err) {
    console.error('[Face] Model load error:', err);
    throw err;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2. Analyse a single photo with ML Kit
// ═══════════════════════════════════════════════════════════════════════════════

async function analysePhoto(
  photoUri: string,
  label:    'before' | 'after',
): Promise<PhotoAnalysis> {
  const resizedUri = await prepareImage(photoUri, IMAGE_SIZE);

  const base: PhotoAnalysis = {
    label,
    uri:          resizedUri,
    leftEyeOpen:  0,
    rightEyeOpen: 0,
    smilingProb:  0,
    detection:    null,
  };

  let faces: Face[];
  try {
    faces = await FaceDetection.detect(resizedUri, {
      performanceMode:    'accurate',
      landmarkMode:       'none',
      classificationMode: 'all',
    });
  } catch (err: any) {
    console.warn(`[Analysis][${label}] Detection error:`, err?.message);
    return base;
  }

  if (faces.length === 0) {
    console.log(`[Analysis][${label}] No face found`);
    return base;
  }

  // Pick the largest face
  const best = faces.reduce((a, b) =>
    b.frame.width * b.frame.height > a.frame.width * a.frame.height ? b : a,
  );

  const leftEyeOpen  = best.leftEyeOpenProbability  ?? 0;
  const rightEyeOpen = best.rightEyeOpenProbability ?? 0;
  const smilingProb  = best.smilingProbability       ?? 0;

  console.log(
    `[Analysis][${label}] leftEye:${leftEyeOpen.toFixed(2)} ` +
    `rightEye:${rightEyeOpen.toFixed(2)} smile:${smilingProb.toFixed(2)}`,
  );

  const { top, left, width, height } = best.frame;
  const detection: DetectionResult = {
    x: Math.max(0, Math.round(left)),
    y: Math.max(0, Math.round(top)),
    w: Math.round(width),
    h: Math.round(height),
  };

  return { ...base, leftEyeOpen, rightEyeOpen, smilingProb, detection };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3. Liveness check — compare BEFORE photo vs AFTER photo
//
//    User workflow:
//      Photo 1 (beforeUri) → neutral face  (before smile/blink)
//      Photo 2 (afterUri)  → smiling/blinking face
//
//    We measure expression deltas between the two photos.
//    A live person will show a real change; a printed photo won't.
// ═══════════════════════════════════════════════════════════════════════════════

export async function checkLiveness(
  beforeUri: string,    // photo taken BEFORE expression change
  afterUri:  string,    // photo taken AFTER smile / blink
): Promise<LivenessResult> {
  await ensureDir(ASSETS_DIR);

  console.log('[Liveness] Analysing BEFORE photo…');
  const before = await analysePhoto(beforeUri, 'before');

  console.log('[Liveness] Analysing AFTER photo…');
  const after  = await analysePhoto(afterUri, 'after');

  // ── Eye delta: expects a blink (drop in eye-open prob) ────────────────────
  const beforeEyeAvg = (before.leftEyeOpen + before.rightEyeOpen) / 2;
  const afterEyeAvg  = (after.leftEyeOpen  + after.rightEyeOpen)  / 2;
  const eyeDelta     = beforeEyeAvg - afterEyeAvg;   // positive = eyes more closed in AFTER
  const blinkDetected = eyeDelta >= BLINK_DELTA_THRESHOLD;

  // ── Smile delta: expects a smile increase in AFTER ────────────────────────
  const smileDelta    = after.smilingProb - before.smilingProb;   // positive = bigger smile in AFTER
  const smileDetected = smileDelta >= SMILE_DELTA_THRESHOLD;

  console.log(
    `[Liveness] eyeDelta:${eyeDelta.toFixed(3)} (need ≥${BLINK_DELTA_THRESHOLD}) blink:${blinkDetected}`,
  );
  console.log(
    `[Liveness] smileDelta:${smileDelta.toFixed(3)} (need ≥${SMILE_DELTA_THRESHOLD}) smile:${smileDetected}`,
  );

  const isLive = blinkDetected || smileDetected;
  console.log(`[Liveness] isLive:${isLive}`);

  // ── Choose the best face crop for recognition ─────────────────────────────
  // Prefer AFTER photo: expression is clearer and it proves liveness intent.
  // Fallback to BEFORE if AFTER had no detection.
  let bestFaceUri:   string | null          = null;
  let bestDetection: DetectionResult | null = null;

  const sourceForCrop = after.detection ? after : before;

  if (sourceForCrop.detection) {
    bestDetection = sourceForCrop.detection;
    bestFaceUri   = await cropFace(
      sourceForCrop.uri,
      sourceForCrop.detection,
      IMAGE_SIZE,
      'recognition_crop.jpg',
    );
    console.log(
      `[Liveness] Using ${sourceForCrop.label} photo for recognition crop`,
    );
  } else {
    console.warn('[Liveness] Neither photo yielded a face detection — recognition will be skipped');
  }

  return {
    isLive,
    blinkDetected,
    smileDetected,
    eyeDelta,
    smileDelta,
    before,
    after,
    bestFaceUri,
    bestDetection,
  };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 4. Recognition — run classifier on the best cropped face
// ═══════════════════════════════════════════════════════════════════════════════

export async function recognizeFaceFromCrop(
  croppedUri: string,
): Promise<RecognitionResult | null> {
  if (!faceModel) throw new Error('Classifier not loaded — call loadFaceModel() first');

  try {
    const tensor  = await faceUriToFloat32Tensor(croppedUri);
    const outputs = await faceModel.run([tensor.buffer]);
    const probs   = new Float32Array(outputs[0]);

    const allScores = Array.from(probs)
      .map((p, i) => `${CLASS_NAMES[i] ?? 'class_' + i}:${p.toFixed(3)}`)
      .join(' | ');
    console.log(`[Recognition] Scores → ${allScores}`);

    let bestIdx = 0, bestScore = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestScore) { bestScore = probs[i]; bestIdx = i; }
    }

    const name = CLASS_NAMES[bestIdx] ?? `class_${bestIdx}`;
    console.log(`[Recognition] name:${name} classId:${bestIdx} score:${bestScore.toFixed(4)}`);
    return { name, score: bestScore, classId: bestIdx, croppedUri };

  } catch (err) {
    console.warn('[Recognition] Failed:', err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5. Full pipeline — 2 photos → liveness → recognition
// ═══════════════════════════════════════════════════════════════════════════════

export async function detectAndRecognizeFromPhotos(
  beforeUri: string,          // photo BEFORE expression change
  afterUri:  string,          // photo AFTER smile / blink
  threshold: number = 0.45,   // minimum confidence to name someone
): Promise<RecognitionResult & { isLive: boolean; liveness: LivenessResult }> {

  // Step 1 — liveness: compare expression deltas between the two photos
  const liveness = await checkLiveness(beforeUri, afterUri);

  if (!liveness.isLive) {
    console.log('[Pipeline] Liveness FAILED — returning early');
    return { name: 'Liveness check failed', score: 0, isLive: false, liveness };
  }

  if (!liveness.bestFaceUri) {
    console.warn('[Pipeline] No usable face crop — cannot recognise');
    return { name: 'No face detected', score: 0, isLive: true, liveness };
  }

  // Step 2 — recognition
  console.log('[Pipeline] Liveness PASSED — running recognition…');
  const result = await recognizeFaceFromCrop(liveness.bestFaceUri);

  if (!result) {
    return { name: 'Recognition failed', score: 0, isLive: true, liveness };
  }

  if (result.score < threshold) {
    console.log(`[Pipeline] Score ${result.score.toFixed(4)} below threshold ${threshold} → Unknown`);
    return {
      name: 'Unknown',
      score: result.score,
      classId: result.classId,
      croppedUri: liveness.bestFaceUri ?? undefined,
      isLive: true,
      liveness,
    };
  }

  return { ...result, isLive: true, liveness };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6. Lower-level helpers (kept for single-photo fallback use)
// ═══════════════════════════════════════════════════════════════════════════════

export async function detectFace(imageUri: string): Promise<DetectionResult | null> {
  const localPath = imageUri.replace('file://', '');
  const exists    = await RNFS.exists(localPath).catch(() => false);
  if (!exists) { console.warn('[MLKit] File not found:', localPath); return null; }

  let faces: Face[];
  try {
    faces = await FaceDetection.detect(imageUri, {
      performanceMode: 'accurate',
      landmarkMode:    'none',
      classificationMode: 'all',
    });
  } catch (err: any) {
    console.error('[MLKit] Detection error:', err?.message ?? String(err));
    return null;
  }

  if (faces.length === 0) { console.log('[MLKit] No faces detected'); return null; }

  const best = faces.reduce((a, b) =>
    b.frame.width * b.frame.height > a.frame.width * a.frame.height ? b : a,
  );

  const { top, left, width, height } = best.frame;
  return {
    x: Math.max(0, Math.round(left)),
    y: Math.max(0, Math.round(top)),
    w: Math.round(width),
    h: Math.round(height),
  };
}

export async function prepareImage(imageUri: string, size: number = 640): Promise<string> {
  const resized = await ImageResizer.createResizedImage(
    imageUri, size, size, 'JPEG', 90, 0, undefined, false,
    { mode: 'cover', onlyScaleDown: false },
  );
  return resized.uri;
}

export async function cropFace(
  imageUri:  string,
  detection: DetectionResult,
  imageSize: number = 640,
  fileName:  string = 'cropped_face.jpg',
): Promise<string | null> {
  try {
    const resized = await ImageResizer.createResizedImage(
      imageUri, imageSize, imageSize, 'JPEG', 95, 0, undefined, false,
      { mode: 'cover', onlyScaleDown: false },
    );

    const imagePath = resized.uri.replace('file://', '');
    const b64       = await RNFS.readFile(imagePath, 'base64');
    const decoded   = jpeg.decode(Buffer.from(b64, 'base64'), { useTArray: true });

    const { x, y, w, h } = detection;
    if (w <= 0 || h <= 0) return null;

    const x0 = Math.max(0, x),            y0 = Math.max(0, y);
    const x1 = Math.min(imageSize, x + w), y1 = Math.min(imageSize, y + h);
    const cw = x1 - x0,                   ch = y1 - y0;
    if (cw <= 0 || ch <= 0) return null;

    const srcData  = decoded.data as Uint8Array;
    const cropData = new Uint8Array(cw * ch * 4);

    for (let row = 0; row < ch; row++) {
      for (let col = 0; col < cw; col++) {
        const srcIdx = ((y0 + row) * imageSize + (x0 + col)) * 4;
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
    await RNFS.writeFile(outputPath, Buffer.from(encoded.data).toString('base64'), 'base64');
    console.log(`[Crop] Saved: ${outputPath} (${cw}×${ch})`);
    return `file://${outputPath}`;

  } catch (err: any) {
    console.error('[Crop] Failed:', err?.message ?? String(err));
    return null;
  }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

async function faceUriToFloat32Tensor(uri: string): Promise<Float32Array> {
  const resized = await ImageResizer.createResizedImage(
    uri, CLASSIFIER_INPUT_SIZE, CLASSIFIER_INPUT_SIZE, 'JPEG', 95, 0,
    undefined, false, { mode: 'cover', onlyScaleDown: false },
  );

  const imagePath = resized.uri.replace('file://', '');
  const buf       = Buffer.from(await RNFS.readFile(imagePath, 'base64'), 'base64');
  const pixels    = jpeg.decode(buf, { useTArray: true }).data as Uint8Array;

  const numPixels = CLASSIFIER_INPUT_SIZE * CLASSIFIER_INPUT_SIZE;
  const tensor    = new Float32Array(numPixels * 3);

  for (let i = 0; i < numPixels; i++) {
    const base    = i * 4;
    tensor[i*3+0] = pixels[base]     / 255.0;
    tensor[i*3+1] = pixels[base + 1] / 255.0;
    tensor[i*3+2] = pixels[base + 2] / 255.0;
  }

  return tensor;
}

async function ensureDir(path: string): Promise<void> {
  if (!(await RNFS.exists(path).catch(() => false))) await RNFS.mkdir(path);
}