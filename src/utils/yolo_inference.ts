import { loadTensorflowModel, TensorflowModel } from 'react-native-fast-tflite';
import ImageResizer from 'react-native-image-resizer';
import RNFS from 'react-native-fs';
import { Buffer } from 'buffer';
import jpeg from 'jpeg-js';
import FaceDetection, { Face } from '@react-native-ml-kit/face-detection';
import { createThumbnail } from 'react-native-create-thumbnail';

// ─── IMPORTANT: Keep this in sync with your model's output classes ────────────
const CLASS_NAMES = ["Alex", "John", "Emma", "David", "Unknown", "Unknown2"];

const ASSETS_DIR = `${RNFS.DocumentDirectoryPath}/FaceRecognition`;
const CLASSIFIER_INPUT_SIZE = 224;
const IMAGE_SIZE = 640;

// ─── Frame positions ──────────────────────────────────────────────────────────
const FRAME_EARLY_PCT  = 0.0; // 20% into video → used for recognition
const FRAME_LATE_PCT   = 0.85; // 85% into video → compared for liveness

// ─── Liveness thresholds ──────────────────────────────────────────────────────
const BLINK_DELTA_THRESHOLD = 0.0;
const SMILE_DELTA_THRESHOLD = 0.0;

// ═══════════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════════

export interface DetectionResult {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FrameAnalysis {
  frameIndex:   number;
  timestampMs:  number;
  uri:          string;
  leftEyeOpen:  number;
  rightEyeOpen: number;
  smilingProb:  number;
  detection:    DetectionResult | null;
}

export interface LivenessResult {
  isLive:        boolean;
  blinkDetected: boolean;
  smileDetected: boolean;
  frames:        FrameAnalysis[];       // always [earlyFrame, lateFrame]
  bestFrameUri:  string | null;         // cropped face from early (20%) frame
  bestDetection: DetectionResult | null;
}

export interface RecognitionResult {
  name:        string;
  score:       number;
  classId?:    number;
  croppedUri?: string;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Module-level singletons
// ═══════════════════════════════════════════════════════════════════════════════

let faceModel: TensorflowModel | null = null;

// ═══════════════════════════════════════════════════════════════════════════════
// 1.  Model loading
// ═══════════════════════════════════════════════════════════════════════════════

export async function loadFaceModel(): Promise<void> {
  if (faceModel) { console.log('[Face] Model already loaded'); return; }
  try {
    console.log('[Face] Loading classifier TFLite…');
    faceModel = await loadTensorflowModel(
      require('../../assets/classify_float32.tflite'), [],
    );
    console.log('[Face] Model loaded successfully');
  } catch (err) {
    console.error('[Face] Model load error:', err);
    throw err;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 2.  Extract exactly 2 frames: 20% and 85% of video duration
// ═══════════════════════════════════════════════════════════════════════════════

export async function extractFrames(
  videoUri:   string,
  durationMs: number = 3000,
): Promise<{ uri: string; timestampMs: number }[]> {
  await ensureDir(ASSETS_DIR);

  const timestamps = [
    Math.round(durationMs * FRAME_EARLY_PCT),  // e.g. 600ms for 3s video
    Math.round(durationMs * FRAME_LATE_PCT),   // e.g. 2550ms for 3s video
  ];

  console.log('[Frames] Extracting 2 frames at timestamps (ms):', timestamps);

  const frames: { uri: string; timestampMs: number }[] = [];

  for (let i = 0; i < timestamps.length; i++) {
    const ts = timestamps[i];
    const label = i === 0 ? '20%' : '85%';
    try {
      const thumb = await createThumbnail({
        url:       videoUri,
        timeStamp: ts,
        format:    'jpeg',
        dirName:   ASSETS_DIR,
      });
      frames.push({
        uri: thumb.path.startsWith('file://') ? thumb.path : `file://${thumb.path}`,
        timestampMs: ts,
      });
      console.log(`[Frames] Frame ${label} @ ${ts}ms → ${thumb.path}`);
    } catch (err: any) {
      console.warn(`[Frames] Failed to extract frame ${label} @ ${ts}ms:`, err?.message);
    }
  }

  return frames; // [earlyFrame, lateFrame]  (may be 1 if one failed)
}

// ═══════════════════════════════════════════════════════════════════════════════
// 3.  Analyse a single frame with ML Kit
// ═══════════════════════════════════════════════════════════════════════════════

async function analyseFrame(
  frameUri:    string,
  frameIndex:  number,
  timestampMs: number,
): Promise<FrameAnalysis> {
  const resizedUri = await prepareImage(frameUri, IMAGE_SIZE);

  const base: FrameAnalysis = {
    frameIndex,
    timestampMs,
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
    console.warn(`[Liveness] Frame ${frameIndex} detection error:`, err?.message);
    return base;
  }

  if (faces.length === 0) {
    console.log(`[Liveness] Frame ${frameIndex}: no face`);
    return base;
  }

  const best = faces.reduce((a, b) =>
    b.frame.width * b.frame.height > a.frame.width * a.frame.height ? b : a,
  );

  const leftEyeOpen  = best.leftEyeOpenProbability  ?? 0;
  const rightEyeOpen = best.rightEyeOpenProbability ?? 0;
  const smilingProb  = best.smilingProbability       ?? 0;
  const label        = frameIndex === 0 ? '20%' : '85%';

  console.log(
    `[Liveness] Frame ${label} @ ${timestampMs}ms — ` +
    `leftEye:${leftEyeOpen.toFixed(2)} rightEye:${rightEyeOpen.toFixed(2)} smile:${smilingProb.toFixed(2)}`,
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
// 4.  Liveness — compare 20% frame vs 85% frame only
// ═══════════════════════════════════════════════════════════════════════════════

export async function checkLiveness(
  videoUri:   string,
  durationMs: number = 3000,
): Promise<LivenessResult> {
  const rawFrames = await extractFrames(videoUri, durationMs);

  if (rawFrames.length === 0) {
    console.warn('[Liveness] No frames extracted');
    return {
      isLive: false, blinkDetected: false, smileDetected: false,
      frames: [], bestFrameUri: null, bestDetection: null,
    };
  }

  // Analyse both frames
  const frames: FrameAnalysis[] = [];
  for (const { uri, timestampMs } of rawFrames) {
    const analysis = await analyseFrame(uri, frames.length, timestampMs);
    frames.push(analysis);
  }

  const [earlyFrame, lateFrame] = frames; // index 0 = 20%, index 1 = 85%

  // ── Need both frames to compare — if only 1 extracted, fail safe ──────────
  if (!lateFrame) {
    console.warn('[Liveness] Only one frame available — cannot compare for liveness');
    return {
      isLive: false, blinkDetected: false, smileDetected: false,
      frames, bestFrameUri: null, bestDetection: null,
    };
  }

  // ── Blink detection: compare average eye-open between 20% and 85% ─────────
  const earlyEye = (earlyFrame.leftEyeOpen + earlyFrame.rightEyeOpen) / 2;
  const lateEye  = (lateFrame.leftEyeOpen  + lateFrame.rightEyeOpen)  / 2;
  const eyeDelta = Math.abs(lateEye - earlyEye);
  const blinkDetected = eyeDelta >= BLINK_DELTA_THRESHOLD;
  console.log(
    `[Liveness] Eye avg — early(20%):${earlyEye.toFixed(3)} late(85%):${lateEye.toFixed(3)} delta:${eyeDelta.toFixed(3)} blink:${blinkDetected}`,
  );

  // ── Smile detection: compare smile prob between 20% and 85% ───────────────
  const smileDelta = Math.abs(lateFrame.smilingProb - earlyFrame.smilingProb);
  const smileDetected = smileDelta >= SMILE_DELTA_THRESHOLD;
  console.log(
    `[Liveness] Smile — early(20%):${earlyFrame.smilingProb.toFixed(3)} late(85%):${lateFrame.smilingProb.toFixed(3)} delta:${smileDelta.toFixed(3)} smile:${smileDetected}`,
  );

  const isLive = blinkDetected || smileDetected;
  console.log(`[Liveness] blinkDetected:${blinkDetected} smileDetected:${smileDetected} isLive:${isLive}`);

  // ── Use the 20% (early) frame for the recognition preview crop ────────────
  let bestFrameUri: string | null = null;
  let bestDetection: DetectionResult | null = null;

  if (earlyFrame.detection) {
    bestDetection = earlyFrame.detection;
    bestFrameUri  = await cropFace(
      earlyFrame.uri,
      earlyFrame.detection,
      IMAGE_SIZE,
      'recognition_frame.jpg',
    );
    console.log('[Liveness] Best frame for preview: early (20%) frame');
  } else if (lateFrame.detection) {
    // Fallback: use late frame crop if early had no detection (won't be used for recognition)
    bestDetection = lateFrame.detection;
    bestFrameUri  = await cropFace(
      lateFrame.uri,
      lateFrame.detection,
      IMAGE_SIZE,
      'recognition_frame.jpg',
    );
    console.warn('[Liveness] Early frame had no detection — preview falling back to 85% frame');
  }

  return { isLive, blinkDetected, smileDetected, frames, bestFrameUri, bestDetection };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 5.  Recognise using ONLY the 20% (early) frame
// ═══════════════════════════════════════════════════════════════════════════════

export async function recognizeFromAllFrames(
  frames: FrameAnalysis[],
): Promise<RecognitionResult | null> {
  if (!faceModel) throw new Error('Classifier not loaded');

  // Only use index 0 — the 20% frame
  const earlyFrame = frames[0];

  if (!earlyFrame) {
    console.warn('[Recognition] No frames available');
    return null;
  }

  if (!earlyFrame.detection) {
    console.warn('[Recognition] Early (20%) frame has no face detection — cannot recognise');
    return null;
  }

  console.log('[Recognition] Running recognition on early (20%) frame only');

  try {
    const croppedUri = await cropFace(
      earlyFrame.uri,
      earlyFrame.detection,
      IMAGE_SIZE,
      `frame_${earlyFrame.frameIndex}.jpg`,
    );

    if (!croppedUri) {
      console.warn('[Recognition] Crop of early frame failed');
      return null;
    }

    const tensor  = await faceUriToFloat32Tensor(croppedUri);
    const outputs = await faceModel.run([tensor.buffer]);
    const probs   = new Float32Array(outputs[0]);

    // Log all class probabilities
    const allScores = Array.from(probs)
      .map((p, i) => `${CLASS_NAMES[i] ?? 'class_' + i}:${p.toFixed(3)}`)
      .join(' | ');
    console.log(`[Recognition] Early frame scores → ${allScores}`);

    // Find the top class
    let bestIdx = 0, bestScore = probs[0];
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > bestScore) { bestScore = probs[i]; bestIdx = i; }
    }

    const predictedName = CLASS_NAMES[bestIdx] ?? `class_${bestIdx}`;
    console.log(
      `[Recognition] Result: ${predictedName} (classId:${bestIdx}) score:${bestScore.toFixed(4)}`,
    );

    return { name: predictedName, score: bestScore, classId: bestIdx };

  } catch (err) {
    console.warn('[Recognition] Recognition on early frame failed:', err);
    return null;
  }
}

// ═══════════════════════════════════════════════════════════════════════════════
// 6.  Full pipeline — liveness + recognition
// ═══════════════════════════════════════════════════════════════════════════════

export async function detectAndRecognizeFromVideo(
  videoUri:   string,
  durationMs: number = 3000,
  threshold:  number = 0.45,
): Promise<RecognitionResult & { isLive: boolean; liveness: LivenessResult }> {
  // Step 1 — liveness check (compares 20% vs 85% frames)
  const liveness = await checkLiveness(videoUri, durationMs);

  if (!liveness.isLive) {
    console.log('[Pipeline] Liveness check FAILED');
    return { name: 'Liveness check failed', score: 0, isLive: false, liveness };
  }

  // Step 2 — check the 20% frame has a face for recognition
  const earlyFrame = liveness.frames[0];
  if (!earlyFrame?.detection) {
    console.warn('[Pipeline] No face detected in early (20%) frame — cannot recognise');
    return { name: 'No face detected', score: 0, isLive: true, liveness };
  }

  console.log('[Pipeline] Liveness passed — running recognition on early (20%) frame');

  // Step 3 — classify using only the 20% frame
  const result = await recognizeFromAllFrames(liveness.frames);

  if (!result) {
    return { name: 'Recognition failed', score: 0, isLive: true, liveness };
  }

  if (result.score < threshold) {
    console.log(`[Pipeline] Score ${result.score.toFixed(4)} below threshold — Unknown`);
    return { name: 'Unknown', score: result.score, isLive: true, liveness };
  }

  return { ...result, croppedUri: liveness.bestFrameUri ?? undefined, isLive: true, liveness };
}

// ═══════════════════════════════════════════════════════════════════════════════
// 7.  Single image helpers (kept for photo fallback)
// ═══════════════════════════════════════════════════════════════════════════════

export async function detectFace(imageUri: string): Promise<DetectionResult | null> {
  const localPath = imageUri.replace('file://', '');
  const exists = await RNFS.exists(localPath).catch(() => false);
  if (!exists) { console.warn('[MLKit] Image file not found:', localPath); return null; }

  let faces: Face[];
  try {
    faces = await FaceDetection.detect(imageUri, {
      performanceMode: 'accurate', landmarkMode: 'none', classificationMode: 'all',
    });
  } catch (err: any) {
    console.error('[MLKit] Detection error:', err?.message ?? String(err));
    return null;
  }

  if (faces.length === 0) { console.log('[MLKit] No faces detected'); return null; }

  const best = faces.reduce((a, b) =>
    b.frame.width * b.frame.height > a.frame.width * a.frame.height ? b : a,
  );

  console.log('[MLKit] Best face frame:', best.frame);
  console.log(`[MLKit] Smiling: ${((best.smilingProbability ?? 0) * 100).toFixed(1)}%`);

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
    console.error('[Crop] cropFace FAILED:', err?.message ?? String(err));
    return null;
  }
}

export async function recognizeFace(croppedFaceUri: string): Promise<RecognitionResult | null> {
  if (!faceModel) throw new Error('Classifier not loaded — call loadFaceModel() first');

  const tensor  = await faceUriToFloat32Tensor(croppedFaceUri);
  const outputs = await faceModel.run([tensor.buffer]);
  const probs   = new Float32Array(outputs[0]);

  let bestIdx = 0, bestScore = probs[0];
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > bestScore) { bestScore = probs[i]; bestIdx = i; }
  }

  const name = CLASS_NAMES[bestIdx] ?? `class_${bestIdx}`;
  console.log(`[Face] Result — name:${name} classId:${bestIdx} score:${bestScore.toFixed(4)}`);
  return { name, score: bestScore, classId: bestIdx };
}

// ═══════════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════════

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