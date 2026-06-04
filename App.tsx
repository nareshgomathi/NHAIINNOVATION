import { PermissionsAndroid, Platform } from 'react-native';
import FaceDetection from '@react-native-ml-kit/face-detection';
/**
 * App.tsx — FaceID  •  React Native CLI
 * Screens: Home → Scanning → Result (with back navigation)
 */
import { launchImageLibrary } from 'react-native-image-picker';

import { Buffer } from 'buffer';
global.Buffer = Buffer;
import RNFS from 'react-native-fs';

import {
  detectFace,
  cropFace,
  loadFaceModel, prepareImage,
  recognizeFromAllFrames, checkLiveness,
} from './src/utils/yolo_inference';

import React, { useEffect, useState, useCallback } from 'react';
import {
  View, Text, TouchableOpacity, StyleSheet, SafeAreaView,
  StatusBar, Image, ScrollView, ActivityIndicator,
  Dimensions, Alert, LayoutChangeEvent, BackHandler,
  FlatList,
} from 'react-native';
import * as ImagePicker from 'react-native-image-picker';
// Gallery JSON — class names of people the model knows
// Loaded at runtime from the same gallery.json used by loadGallery()


// ─── Constants ────────────────────────────────────────────────────────────────
const { width: SCREEN_W } = Dimensions.get('window');
const NAVBAR_HEIGHT = 70;
const PURPLE = '#7c5cfc';
const PURPLE_LIGHT = '#a78bfa';
const PURPLE_GLOW = 'rgba(124,92,252,0.25)';
const BG = '#0a0a0f';
const CARD = '#13111e';
const CARD2 = '#0f0d1a';
const BORDER = '#1e1a2e';
const BORDER2 = '#2a2240';
const BBOX_COLOR = '#00f5a0';
const GREEN = '#22c55e';
const TEXT_PRIMARY = '#f0eaff';
const TEXT_MUTED = '#6b5a8a';
const TEXT_SUB = '#9080b0';

const MOCK_ID = 'ID-00342';
const MOCK_DOB = '14 Mar 1990';

// ─── Types ────────────────────────────────────────────────────────────────────
type Screen = 'home' | 'scanning' | 'result';

interface BBox { x: number; y: number; w: number; h: number; }

interface RecentScan {
  id: string;
  name: string;
  scannedAt: string;        // formatted string
  scannedDate: string;      // e.g. "02 Jun 2026"
  synced: boolean;
  imageUri: string;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
function getNow(): { time: string; date: string; full: string } {
  const d = new Date();
  const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  const hh = d.getHours();
  const mm = String(d.getMinutes()).padStart(2, '0');
  const ampm = hh >= 12 ? 'PM' : 'AM';
  const h12 = hh % 12 || 12;
  const time = `${h12}:${mm} ${ampm}`;
  const date = `${String(d.getDate()).padStart(2, '0')} ${months[d.getMonth()]} ${d.getFullYear()}`;
  return { time, date, full: `${days[d.getDay()]}, ${date}  •  ${time}` };
}


// ─── BBoxOverlay ──────────────────────────────────────────────────────────────
function BBoxOverlay({ bbox, label, confidence }: { bbox: BBox; label: string; confidence: number }) {
  const [imgSize, setImgSize] = useState({ w: 0, h: 0 });
  const onLayout = (e: LayoutChangeEvent) =>
    setImgSize({ w: e.nativeEvent.layout.width, h: e.nativeEvent.layout.height });

  const scale = imgSize.w / 640;
  const left = bbox.x * scale;
  const top = bbox.y * scale;
  const bW = bbox.w * scale;
  const bH = bbox.h * scale;
  const CORNER = 16;

  return (
    <View style={StyleSheet.absoluteFill} onLayout={onLayout} pointerEvents="none">
      {imgSize.w > 0 && (
        <>
          <View style={{ position: 'absolute', left, top, width: bW, height: bH, borderWidth: 2, borderColor: BBOX_COLOR, borderRadius: 4 }} />
          <View style={{ position: 'absolute', left: left - 1, top: top - 1, width: CORNER, height: CORNER, borderTopWidth: 3, borderLeftWidth: 3, borderColor: BBOX_COLOR }} />
          <View style={{ position: 'absolute', left: left + bW - CORNER + 1, top: top - 1, width: CORNER, height: CORNER, borderTopWidth: 3, borderRightWidth: 3, borderColor: BBOX_COLOR }} />
          <View style={{ position: 'absolute', left: left - 1, top: top + bH - CORNER + 1, width: CORNER, height: CORNER, borderBottomWidth: 3, borderLeftWidth: 3, borderColor: BBOX_COLOR }} />
          <View style={{ position: 'absolute', left: left + bW - CORNER + 1, top: top + bH - CORNER + 1, width: CORNER, height: CORNER, borderBottomWidth: 3, borderRightWidth: 3, borderColor: BBOX_COLOR }} />
          <View style={{ position: 'absolute', left, top: Math.max(0, top - 28), backgroundColor: BBOX_COLOR, borderRadius: 4, paddingHorizontal: 8, paddingVertical: 3, flexDirection: 'row', alignItems: 'center', gap: 6 }}>
            <Text style={{ color: '#000', fontSize: 11, fontWeight: '800', letterSpacing: 0.5 }}>{label}</Text>
            <Text style={{ color: '#003', fontSize: 11, fontWeight: '700' }}>{confidence}%</Text>
          </View>
        </>
      )}
    </View>
  );
}

// ─── Logo Component ───────────────────────────────────────────────────────────
function AppLogo() {
  return (
    <Image
      source={require('./src/logo.png')}
      style={s.logoImg}
      resizeMode="contain"
    />
  );
}

// ─── Header ───────────────────────────────────────────────────────────────────
function Header() {
  return (
    <View style={s.navbar}>
      <View style={s.navLeft}>
        <Image
          source={require('./src/logo.png')}
          style={s.navLogo}
          resizeMode="contain"
        />
      </View>

      <Text style={s.navTitle}>NHAI Netra</Text>
    </View>
  );
}
// ─── Recent Scan Card ─────────────────────────────────────────────────────────
function RecentCard({ item }: { item: RecentScan }) {
  return (
    <View style={s.recentCard}>
      <View style={s.recentAvatarWrapper}>
        <Image source={{ uri: item.imageUri }} style={s.recentAvatar} />
        <View style={[s.recentSyncDot, { backgroundColor: item.synced ? GREEN : '#ef4444' }]} />
      </View>
      <View style={s.recentInfo}>
        <Text style={s.recentName} numberOfLines={1}>{item.name}</Text>
        <View style={s.recentMeta}>
          <Text style={s.recentMetaText}>🕐 {item.scannedAt}</Text>
          <Text style={s.recentMetaDivider}>·</Text>
          <Text style={s.recentMetaText}>📅 {item.scannedDate}</Text>
        </View>
      </View>
      <View style={[s.syncPill, { borderColor: item.synced ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)', backgroundColor: item.synced ? 'rgba(34,197,94,0.08)' : 'rgba(239,68,68,0.08)' }]}>
        <View style={[s.syncDot2, { backgroundColor: item.synced ? GREEN : '#ef4444' }]} />
        <Text style={[s.syncText, { color: item.synced ? GREEN : '#ef4444' }]}>{item.synced ? 'Synced' : 'Local'}</Text>
      </View>
    </View>
  );
}

// ─── Main App ─────────────────────────────────────────────────────────────────
export default function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const [displayImageUri, setDisplayImageUri] = useState<string | null>(null);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [capturedAt, setCapturedAt] = useState('');
  const [bbox, setBbox] = useState<BBox>({ x: 0, y: 0, w: 0, h: 0 });
  const [faceVerified, setFaceVerified] = useState(false);
  const [recognition, setRecognition] = useState({ name: 'Unknown', score: 0 });
  const [recentScans, setRecentScans] = useState<RecentScan[]>([]);
  const [totalScans, setTotalScans] = useState(13);
  const [classNames, setClassNames] = useState<string[]>([]);
  const [videoDurationMs, setVideoDurationMs] = useState(3000);

  const [showClasses, setShowClasses] = useState(false);
  const [croppedFaceUri, setCroppedFaceUri] =
    useState<string | null>(null);


  const first30Classes = [
    'Adriana Lima',
    'Alex Lawther',
    'Alexandra Daddario',
    'Alvaro Morte',
    'Amanda Crew',
    'Andy Samberg',
    'Anne Hathaway',
    'Anthony Mackie',
    'Avril Lavigne',
    'Ben Affleck',
    'Bill Gates',
    'Bobby Morley',
    'Brenton Thwaites',
    'Ronaldo',
    'Brie Larson',
    'Chris Evans',
    'Chris Hemsworth',
    'Chris Pratt',
    'Christian Bale',
    'Smith',
 
  ];
  // ── Hardware back button ──────────────────────────────────────────────────
  useEffect(() => {
    const sub = BackHandler.addEventListener('hardwareBackPress', () => {
      if (screen === 'result' || screen === 'scanning') {
        setScreen('home');
        return true;
      }
      return false;
    });
    return () => sub.remove();
  }, [screen]);

  // ── Load models + class names ─────────────────────────────────────────────
  // BEFORE
  useEffect(() => {
    const loadNames = async () => {
      const destPath = RNFS.DocumentDirectoryPath + '/names.txt';
      const exists = await RNFS.exists(destPath);
      if (!exists) {
        await RNFS.copyFileAssets('names.txt', destPath);
      }
      const content = await RNFS.readFile(destPath, 'utf8');
      const names = content.split('\n').map(n => n.trim()).filter(Boolean);
      setClassNames(names);
    };
    loadNames();
  }, []);

  // AFTER
  useEffect(() => {
    const init = async () => {
      try {
        const destPath = RNFS.DocumentDirectoryPath + '/names.txt';
        const exists = await RNFS.exists(destPath);
        if (!exists) {
          await RNFS.copyFileAssets('names.txt', destPath);
        }
        const content = await RNFS.readFile(destPath, 'utf8');
        const names = content.split('\n').map(n => n.trim()).filter(Boolean);
        setClassNames(names);

        // ← removed loadYOLOModel(); ML Kit needs no pre-loading
        await loadFaceModel();

        console.log('[App] Models and names loaded');
      } catch (e: any) {
        console.error('[App] Init error:', e?.message);
      }
    };
    init();
  }, []);

// REPLACE the entire processDetection function
// WITH
const processVideo = async (uri: string, durationMs: number) => {
  try {
    setScreen('scanning');
    setImageUri(uri);   // show something on scanning screen

    const liveness = await checkLiveness(uri, durationMs);

    if (!liveness.isLive) {
      Alert.alert(
        'Liveness Check Failed',
        `No blink or smile change detected across 5 frames.\nPlease record a short video with natural movement.`,
      );
      setScreen('home');
      return;
    }

    if (!liveness.bestFrameUri) {
      Alert.alert('No Face Detected', 'Could not find a face in any frame.');
      setScreen('home');
      return;
    }
    setFaceVerified(true);

    setCroppedFaceUri(liveness.bestFrameUri);

    // Use middle frame's detection for bbox display
    const middleFrame = liveness.frames[Math.floor(liveness.frames.length / 2)];
    if (middleFrame?.detection) {
      setBbox(middleFrame.detection);
      setDisplayImageUri(middleFrame.uri);
    }

    // Classify the cropped best frame
    const { recognizeFromAllFrames } = require('./src/utils/yolo_inference');
    const result = await recognizeFromAllFrames(liveness.frames);
    const recResult = result
      ? {
          name: (result.classId !== undefined ? classNames[result.classId] : undefined) ?? result.name,
          score: Number((result.score * 100).toFixed(2)),
        }
      : { name: 'Unknown', score: 0 };

    setRecognition(recResult);

    // Save to recents
    const now = getNow();
    const newScan: RecentScan = {
      id: String(Date.now()),
      name: recResult.name,
      scannedAt: now.time,
      scannedDate: now.date,
      synced: Math.random() > 0.4,
      imageUri: liveness.bestFrameUri,
    };
    setRecentScans(prev => [newScan, ...prev].slice(0, 10));
    setTotalScans(prev => prev + 1);

    setTimeout(() => setScreen('result'), 600);
  } catch (e: any) {
    console.error('[App] Video processing error:', e?.message);
    setScreen('home');
  }
};

const handleVideoUpload = async () => {
  const granted = await requestVideoPermission();

  if (!granted) {
    Alert.alert('Permission Denied');
    return;
  }

  launchImageLibrary(
    {
      mediaType: 'video',
      selectionLimit: 1,
    },
    async response => {
      if (response.didCancel) return;

      if (response.errorCode) {
        Alert.alert(
          'Video Selection Error',
          response.errorMessage ?? response.errorCode,
        );
        return;
      }

      const asset = response.assets?.[0];
      const uri = asset?.uri ?? '';

      if (!uri) return;

      const durationMs = Math.round((asset?.duration ?? 3) * 1000);

      await processVideo(uri, durationMs);
    },
  );
};
  // ── Pickers ───────────────────────────────────────────────────────────────
  // const handleUpload = () => {
  //   ImagePicker.launchImageLibrary({ mediaType: 'video', selectionLimit: 1 }, async response => {
  //     if (response.didCancel || response.errorCode) return;
  //     const asset = response.assets?.[0];
  //     const uri = asset?.uri ?? '';
  //     if (!uri) return;
  //     const durationMs = Math.round((asset?.duration ?? 3) * 1000);
  //     setVideoDurationMs(durationMs);
  //     setCapturedAt(getNow().full);
  //     await processVideo(uri, durationMs);
  //   });
  // };


async function requestCameraPermission() {
  if (Platform.OS !== 'android') return true;

  const granted = await PermissionsAndroid.request(
    PermissionsAndroid.PERMISSIONS.CAMERA,
    {
      title: 'Camera Permission',
      message: 'App needs camera access',
      buttonPositive: 'OK',
    },
  );

  return granted === PermissionsAndroid.RESULTS.GRANTED;
}

async function requestVideoPermission() {
  if (Platform.OS !== 'android') return true;

  const permission =
    Platform.Version >= 33
      ? PermissionsAndroid.PERMISSIONS.READ_MEDIA_VIDEO
      : PermissionsAndroid.PERMISSIONS.READ_EXTERNAL_STORAGE;

  const granted = await PermissionsAndroid.request(permission);

  return granted === PermissionsAndroid.RESULTS.GRANTED;
}
const handleCamera = async () => {
  const hasPermission = await requestCameraPermission();

  if (!hasPermission) {
    Alert.alert('Permission Denied');
    return;
  }

  ImagePicker.launchCamera(
  {
    mediaType: 'video',
    videoQuality: 'high',
    durationLimit: 5,
    saveToPhotos: true,
  },
    async response => {
      console.log(response);

      if (response.didCancel) return;

      if (response.errorCode) {
        Alert.alert(
          'Camera Error',
          response.errorMessage ?? response.errorCode,
        );
        return;
      }

      const uri = response.assets?.[0]?.uri ?? '';
const durationMs =
  Math.round((response.assets?.[0]?.duration ?? 3) * 1000);

await processVideo(uri, durationMs);
    },
  );
};

  const handleReset = () => {
  setScreen('home');
  setImageUri(null);
  setCapturedAt('');
  setCroppedFaceUri(null);
  setFaceVerified(false);
  setBbox({ x: 0, y: 0, w: 0, h: 0 });
};

  // ── Derived stats ─────────────────────────────────────────────────────────
  const knownCount = 13;
  const syncedCount = 13;

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={s.safe}>
      <StatusBar barStyle="light-content" backgroundColor={BG} />

      {/* ══ HOME SCREEN ══ */}
      {screen === 'home' && (
        <>
          <Header />
          <ScrollView contentContainerStyle={s.homeContent} showsVerticalScrollIndicator={false}>

            {/* ── Hero greeting ── */}
            <View style={s.heroBanner}>
              <View style={s.heroBannerLeft}>
                <Text style={s.heroBannerLabel}>AI FACE RECOGNITION</Text>
                <Text style={s.heroBannerTitle}>Identify Faces{'\n'}Instantly</Text>
                <Text style={s.heroBannerSub}>On-device TFLite model.{'\n'}Private. Fast. Accurate.</Text>
              </View>
              <View style={s.heroBannerRight}>
                <View style={s.heroRing}>
                  <View style={s.heroRingInner}>
                    <View style={s.heroRingCore}>
                      <Text style={s.heroIcon}>⬡</Text>
                    </View>
                  </View>
                </View>
              </View>
            </View>

            {/* ── Stats Row ── */}
            <View style={s.statsRow}>
              <View style={[s.statCard, { borderColor: 'rgba(124,92,252,0.35)' }]}>
                <Text style={s.statValue}>{totalScans}</Text>
                <Text style={s.statLabel}>Total Scans</Text>
                <View style={[s.statAccent, { backgroundColor: PURPLE }]} />
              </View>
              <View style={[s.statCard, { borderColor: 'rgba(0,245,160,0.25)' }]}>
                <Text style={[s.statValue, { color: BBOX_COLOR }]}>{knownCount}</Text>
                <Text style={s.statLabel}>Identified</Text>
                <View style={[s.statAccent, { backgroundColor: BBOX_COLOR }]} />
              </View>
              <View style={[s.statCard, { borderColor: 'rgba(34,197,94,0.25)' }]}>
                <Text style={[s.statValue, { color: GREEN }]}>{syncedCount}</Text>
                <Text style={s.statLabel}>Synced</Text>
                <View style={[s.statAccent, { backgroundColor: GREEN }]} />
              </View>
            </View>

            {/* ── Trained Classes ── */}
            <View style={s.classesCard}>
              <Text style={s.classesNote}>
                Demo model is trained to recognise:
              </Text>

              <TouchableOpacity
                style={s.viewClassesBtn}
                onPress={() => setShowClasses(!showClasses)}
                activeOpacity={0.8}
              >
                <Text style={s.viewClassesBtnText}>
                  {showClasses
                    ? 'Hide Classes ▲'
                    : 'Click Here To View 20 Persons ▼'}
                </Text>
              </TouchableOpacity>

              <Text style={s.modelNote}>
                Note: Displaying only 20 sample identities for demonstration.
                The AI recognition model can be trained with hundreds or thousands
                of classes depending on available training data and device resources.
              </Text>

              {showClasses && (
                <View style={s.classPillsRow}>
                  {first30Classes.map((name, index) => (
                    <View key={index} style={s.classPill}>
                      <View style={s.classPillDot} />
                      <Text style={s.classPillText}>{name}</Text>
                    </View>
                  ))}
                </View>
              )}
            </View>

            {/* ── Scan Buttons ── */}
            <View style={s.sectionHeader}>
              <View style={s.sectionDot} />
              <Text style={s.sectionTitle}>New Scan</Text>
              <View style={s.sectionLine} />
            </View>

            <TouchableOpacity style={s.btnPrimary} onPress={handleCamera} activeOpacity={0.85}>
              <Text style={s.btnIcon}>📷</Text>
              <View style={s.btnTextBlock}>
                <Text style={s.btnLabel}>Open Camera</Text>
                <Text style={s.btnHint}>Capture face in real-time</Text>
              </View>
              <View style={s.btnArrow}><Text style={s.btnArrowText}>›</Text></View>
            </TouchableOpacity>

            <TouchableOpacity
  style={s.btnSecondary}
  onPress={handleVideoUpload}
>
              <Text style={s.btnIcon}>🖼️</Text>
              <View style={s.btnTextBlock}>
                <Text style={[s.btnLabel, { color: '#c8b8ff' }]}>Upload Video</Text>
                <Text style={[s.btnHint, { color: TEXT_MUTED }]}>Choose from gallery</Text>
              </View>
              <View style={[s.btnArrow, { borderColor: '#3d2f5e' }]}>
                <Text style={[s.btnArrowText, { color: '#c8b8ff' }]}>›</Text>
              </View>
            </TouchableOpacity>

            {/* ── Recent Scans ── */}
            {recentScans.length > 0 && (
              <>
                <View style={s.sectionHeader}>
                  <View style={s.sectionDot} />
                  <Text style={s.sectionTitle}>Recent Scans</Text>
                  <View style={s.sectionLine} />
                  <Text style={s.sectionCount}>{recentScans.length}</Text>
                </View>
                {recentScans.slice(0, 3).map(item => (
                  <RecentCard key={item.id} item={item} />
                ))}
              </>
            )}

            <Text style={s.footerNote}>🔒 Processed locally · never stored remotely</Text>
          </ScrollView>
        </>
      )}

      {/* ══ SCANNING SCREEN ══ */}
      {screen === 'scanning' && (
        <>
          <Header onBack={handleReset} />
          <View style={s.scanningContainer}>
            {imageUri && (
              <View style={s.scanPreviewWrapper}>
                <Image source={{ uri: imageUri }} style={s.scanPreviewImage} />
                <View style={s.scanOverlay}>
                  <View style={s.scanBracketTL} />
                  <View style={s.scanBracketTR} />
                  <View style={s.scanBracketBL} />
                  <View style={s.scanBracketBR} />
                </View>
              </View>
            )}
            <ActivityIndicator size="large" color={PURPLE} style={{ marginTop: 32 }} />
            <Text style={s.scanningTitle}>Running TFLite Detection…</Text>
            <Text style={s.scanningHint}>Analysing facial features</Text>
            <View style={s.dotsRow}>
              {[0, 1, 2, 3, 4].map(i => <View key={i} style={[s.dot, { opacity: 0.3 + i * 0.14 }]} />)}
            </View>
          </View>
        </>
      )}

      {/* ══ RESULT SCREEN ══ */}
      {screen === 'result' && (
        <>
          <Header onBack={handleReset} />
          <ScrollView contentContainerStyle={s.resultContainer} showsVerticalScrollIndicator={false}>

          

            <View style={s.portraitCard}>
              <View style={[s.corner, s.cornerTL]} />
              <View style={[s.corner, s.cornerTR]} />
              <View style={[s.corner, s.cornerBL]} />
              <View style={[s.corner, s.cornerBR]} />

              <View style={s.avatarWrapper}>
                <View style={s.avatarGlow} />
                {imageUri
                  ? <Image
  source={{
    uri: croppedFaceUri || imageUri,
  }}
  style={s.avatarImage}
/>
                  : <View style={s.avatarCircle}><Text style={s.avatarEmoji}>👤</Text></View>
                }
                <View style={s.verifiedBadge}><Text style={s.verifiedText}>✓</Text></View>
              </View>
<Text style={s.resultName}>{recognition.name}</Text>

{faceVerified && recognition.name !== 'Unknown' && (
  <View style={s.authenticityPill}>
    <Text style={s.authenticityIcon}>🟢</Text>
    <Text style={s.authenticityText}>
      Liveness check passed
    </Text>
    <Text style={s.authenticityText}>
      Face Authenticity Verified
    </Text>
  </View>
)}
              
              <View style={s.idRow}>
                <Text style={s.idLabel}>ID</Text>
                <Text style={s.idValue}>{MOCK_ID}</Text>
              </View>

              <View style={s.confidenceRow}>
                <Text style={s.confidenceLabel}>Confidence</Text>
                <View style={s.confidenceBar}>
                  <View style={[s.confidenceFill, { width: `${recognition.score}%` as any }]} />
                </View>
                <Text style={s.confidenceValue}>{recognition.score.toFixed(2)}%</Text>
              </View>

              <View style={s.cardDivider} />

              <View style={s.bboxContainer}>
                <View style={s.bboxTitleRow}>
                  <View style={s.bboxDot} />
                  <Text style={s.bboxTitle}>TFLite Bounding Box</Text>
                </View>
                <View style={s.bboxGrid}>
                  {(['x', 'y', 'w', 'h'] as const).map(key => (
                    <View key={key} style={s.bboxCell}>
                      <Text style={s.bboxKey}>{key.toUpperCase()}</Text>
                      <Text style={s.bboxVal}>{bbox[key]}</Text>
                    </View>
                  ))}
                </View>
              </View>

              <View style={s.cardDivider} />

              <View style={s.metaRow}>
                <Text style={s.metaIcon}>🎂</Text>
                <Text style={s.metaKey}>Date of Birth</Text>
                <Text style={s.metaValue}>{MOCK_DOB}</Text>
              </View>
              <View style={s.metaRow}>
                <Text style={s.metaIcon}>🕐</Text>
                <Text style={s.metaKey}>Scanned At</Text>
                <Text style={s.metaValue}>{capturedAt}</Text>
              </View>

              <View style={s.cardDivider} />

              <View style={s.statusPill}>
                <View style={s.statusDot} />
                <Text style={s.statusText}>Identity Verified</Text>
              </View>
            </View>

            <TouchableOpacity style={s.btnPrimary} onPress={handleReset} activeOpacity={0.85}>
              <Text style={s.btnIcon}>🔄</Text>
              <View style={s.btnTextBlock}>
                <Text style={s.btnLabel}>Scan Again</Text>
                <Text style={s.btnHint}>Start a new recognition</Text>
              </View>
              <View style={s.btnArrow}><Text style={s.btnArrowText}>›</Text></View>
            </TouchableOpacity>

            <TouchableOpacity style={s.btnSecondary} activeOpacity={0.85}>
              <Text style={s.btnIcon}>📋</Text>
              <View style={s.btnTextBlock}>
                <Text style={[s.btnLabel, { color: '#c8b8ff' }]}>Save Report</Text>
                <Text style={[s.btnHint, { color: TEXT_MUTED }]}>Export result as PDF</Text>
              </View>
              <View style={[s.btnArrow, { borderColor: '#3d2f5e' }]}>
                <Text style={[s.btnArrowText, { color: '#c8b8ff' }]}>›</Text>
              </View>
            </TouchableOpacity>

          </ScrollView>
        </>
      )}
    </SafeAreaView>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: BG },
  navbar: {
    height: 70,
    backgroundColor: BG,
    borderBottomWidth: 1,
    borderBottomColor: BORDER,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
  },
  viewClassesBtn: {
    backgroundColor: 'rgba(124,92,252,0.15)',
    borderWidth: 1,
    borderColor: PURPLE,
    borderRadius: 14,
    paddingVertical: 12,
    alignItems: 'center',
    marginBottom: 14,
  },

  viewClassesBtnText: {
    color: PURPLE_LIGHT,
    fontSize: 13,
    fontWeight: '800',
    letterSpacing: 0.8,
  },
  heroBannerTitle: {
    fontSize: 26,
    fontWeight: '900',
    color: '#FFFFFF',
    letterSpacing: 0.8,
    lineHeight: 32,
  },

  navTitle: {
    color: '#60A5FA',
    fontSize: 26,
    fontWeight: '900',
    letterSpacing: 2.5,
    textTransform: 'uppercase',
  },authenticityPill: {
  flexDirection: 'row',
  alignItems: 'center',
  backgroundColor: 'rgba(34,197,94,0.12)',
  borderWidth: 1,
  borderColor: 'rgba(34,197,94,0.35)',
  borderRadius: 20,
  paddingHorizontal: 12,
  paddingVertical: 6,
  marginTop: 8,
  marginBottom: 12,
},

authenticityIcon: {
  fontSize: 10,
  marginRight: 6,
},

authenticityText: {
  color: '#22c55e',
  fontSize: 12,
  fontWeight: '800',
  letterSpacing: 0.4,
},
  modelNote: {
    color: '#8B95A7',
    fontSize: 12,
    lineHeight: 18,
    textAlign: 'center',
    marginBottom: 16,
    paddingHorizontal: 8,
    fontStyle: 'italic',
  },
  sectionTitle: {
    fontSize: 13,
    color: '#C4B5FD',
    fontWeight: '800',
    letterSpacing: 2,
    textTransform: 'uppercase',
  },

  statValue: {
    fontSize: 30,
    fontWeight: '900',
    color: PURPLE_LIGHT,
  },

  resultName: {
    fontSize: 30,
    fontWeight: '900',
    color: '#FFFFFF',
    letterSpacing: 1,
  },

  classPillText: {
    fontSize: 13,
    color: '#E9D5FF',
    fontWeight: '700',
  },

  btnLabel: {
    fontSize: 16,
    fontWeight: '800',
    color: '#FFFFFF',
  },

  recentName: {
    fontSize: 15,
    fontWeight: '800',
    color: '#FFFFFF',
  },
  navLeft: {
    flexDirection: 'row',
    alignItems: 'center',
  },

  navLogo: {
    width: 45,
    height: 45,
    borderRadius: 10,
  },

  navTitle: {
    color: '#2a69c8',
    fontSize: 24,
    fontWeight: '900',
    letterSpacing: 3,
    textTransform: 'uppercase',
    textShadowColor: 'rgba(124,92,252,0.7)',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },
  homeContent: {
    paddingHorizontal: 20,
    paddingTop: NAVBAR_HEIGHT + 20,
    paddingBottom: 40,
  },

  scanningContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingHorizontal: 40,
    paddingTop: NAVBAR_HEIGHT,
  },


  // ── Header ──
  header: {
    flexDirection: 'row', alignItems: 'center',
    paddingHorizontal: 20, paddingVertical: 14,
    borderBottomWidth: 1, borderBottomColor: BORDER,
    justifyContent: 'space-between',
  },
  headerLeft: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  headerDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: PURPLE, marginRight: 10 },
  headerTitle: { color: TEXT_PRIMARY, fontSize: 17, fontWeight: '700', letterSpacing: 3, textTransform: 'uppercase', marginRight: 10 },
  headerBadge: { backgroundColor: PURPLE_GLOW, borderWidth: 1, borderColor: PURPLE, borderRadius: 6, paddingHorizontal: 7, paddingVertical: 2 },
  headerBadgeText: { color: PURPLE_LIGHT, fontSize: 10, fontWeight: '800', letterSpacing: 2 },
  logoImg: { width: 40, height: 40, borderRadius: 10 },
  backBtn: { flexDirection: 'row', alignItems: 'center', flex: 1 },
  backArrow: { color: PURPLE_LIGHT, fontSize: 28, lineHeight: 30, marginRight: 4, marginTop: -2 },
  backText: { color: PURPLE_LIGHT, fontSize: 15, fontWeight: '600' },

  // ── Home ──


  heroBanner: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: CARD, borderRadius: 20, borderWidth: 1, borderColor: BORDER2,
    padding: 20, marginBottom: 20, overflow: 'hidden',
  },
  heroBannerLeft: { flex: 1 },
  heroBannerLabel: { fontSize: 10, color: PURPLE_LIGHT, letterSpacing: 2, fontWeight: '700', textTransform: 'uppercase', marginBottom: 6 },
  heroBannerTitle: { fontSize: 22, fontWeight: '800', color: TEXT_PRIMARY, lineHeight: 28, marginBottom: 8 },
  heroBannerSub: { fontSize: 12, color: TEXT_MUTED, lineHeight: 18 },
  heroBannerRight: { paddingLeft: 16 },
  heroRing: { width: 90, height: 90, borderRadius: 45, borderWidth: 1, borderColor: 'rgba(124,92,252,0.3)', alignItems: 'center', justifyContent: 'center' },
  heroRingInner: { width: 72, height: 72, borderRadius: 36, borderWidth: 1, borderColor: 'rgba(124,92,252,0.5)', alignItems: 'center', justifyContent: 'center' },
  heroRingCore: { width: 54, height: 54, borderRadius: 27, backgroundColor: 'rgba(124,92,252,0.15)', borderWidth: 2, borderColor: PURPLE, alignItems: 'center', justifyContent: 'center' },
  heroIcon: { fontSize: 22, color: PURPLE_LIGHT },

  // ── Stats ──
  statsRow: { flexDirection: 'row', gap: 10, marginBottom: 24 },
  statCard: {
    flex: 1, backgroundColor: CARD2, borderRadius: 14, borderWidth: 1,
    paddingVertical: 14, paddingHorizontal: 12, alignItems: 'center', overflow: 'hidden',
  },
  statValue: { fontSize: 26, fontWeight: '800', color: PURPLE_LIGHT, marginBottom: 4 },
  statLabel: { fontSize: 10, color: TEXT_MUTED, letterSpacing: 0.5, textAlign: 'center' },
  statAccent: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 2, opacity: 0.7 },

  // ── Section header ──
  sectionHeader: { flexDirection: 'row', alignItems: 'center', marginBottom: 12, marginTop: 4 },
  sectionDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: PURPLE, marginRight: 8 },
  sectionTitle: { fontSize: 12, color: TEXT_SUB, fontWeight: '700', letterSpacing: 1.5, textTransform: 'uppercase', marginRight: 10 },
  sectionLine: { flex: 1, height: 1, backgroundColor: BORDER },
  sectionCount: { marginLeft: 8, fontSize: 11, color: PURPLE_LIGHT, fontWeight: '700', backgroundColor: PURPLE_GLOW, paddingHorizontal: 7, paddingVertical: 2, borderRadius: 8, borderWidth: 1, borderColor: PURPLE },

  // ── Classes ──
  classesCard: {
    backgroundColor: CARD2, borderRadius: 16, borderWidth: 1, borderColor: BORDER2,
    padding: 16, marginBottom: 24,
  },
  classesNote: { fontSize: 12, color: TEXT_MUTED, marginBottom: 12 },
  classPillsRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 8 },
  classPill: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: 'rgba(124,92,252,0.1)', borderRadius: 20,
    borderWidth: 1, borderColor: 'rgba(124,92,252,0.3)',
    paddingHorizontal: 12, paddingVertical: 6, gap: 6,
  },
  classPillDot: { width: 5, height: 5, borderRadius: 2.5, backgroundColor: PURPLE_LIGHT },
  classPillText: { fontSize: 12, color: PURPLE_LIGHT, fontWeight: '600' },

  // ── Buttons ──
  btnPrimary: {
    width: '100%', flexDirection: 'row', alignItems: 'center',
    backgroundColor: PURPLE, borderRadius: 16,
    paddingVertical: 16, paddingHorizontal: 20, marginBottom: 12,
    shadowColor: PURPLE, shadowOpacity: 0.45, shadowRadius: 14,
    shadowOffset: { width: 0, height: 6 }, elevation: 8,
  },
  btnSecondary: {
    width: '100%', flexDirection: 'row', alignItems: 'center',
    backgroundColor: CARD, borderRadius: 16,
    paddingVertical: 16, paddingHorizontal: 20, marginBottom: 20,
    borderWidth: 1, borderColor: BORDER,
  },
  btnIcon: { fontSize: 22, marginRight: 14 },
  btnTextBlock: { flex: 1 },
  btnLabel: { fontSize: 15, fontWeight: '700', color: '#fff', marginBottom: 2 },
  btnHint: { fontSize: 12, color: 'rgba(255,255,255,0.55)' },
  btnArrow: { width: 30, height: 30, borderRadius: 8, borderWidth: 1, borderColor: 'rgba(255,255,255,0.25)', alignItems: 'center', justifyContent: 'center' },
  btnArrowText: { color: '#fff', fontSize: 20, lineHeight: 24 },

  // ── Recent scans ──
  recentCard: {
    flexDirection: 'row', alignItems: 'center',
    backgroundColor: CARD2, borderRadius: 14, borderWidth: 1, borderColor: BORDER,
    padding: 12, marginBottom: 10,
  },
  recentAvatarWrapper: { position: 'relative', marginRight: 12 },
  recentAvatar: { width: 46, height: 46, borderRadius: 23, borderWidth: 2, borderColor: BORDER2 },
  recentSyncDot: { position: 'absolute', bottom: 0, right: 0, width: 12, height: 12, borderRadius: 6, borderWidth: 2, borderColor: CARD2 },
  recentInfo: { flex: 1 },
  recentName: { fontSize: 14, fontWeight: '700', color: TEXT_PRIMARY, marginBottom: 4 },
  recentMeta: { flexDirection: 'row', alignItems: 'center', gap: 6 },
  recentMetaText: { fontSize: 11, color: TEXT_MUTED },
  recentMetaDivider: { fontSize: 11, color: BORDER2 },
  syncPill: { flexDirection: 'row', alignItems: 'center', borderWidth: 1, borderRadius: 12, paddingHorizontal: 8, paddingVertical: 4, gap: 4, marginLeft: 8 },
  syncDot2: { width: 6, height: 6, borderRadius: 3 },
  syncText: { fontSize: 10, fontWeight: '700' },

  footerNote: { marginTop: 4, marginBottom: 8, fontSize: 11, color: '#3d3058', textAlign: 'center', letterSpacing: 0.3 },

  // ── Scanning ──
  scanPreviewWrapper: { width: SCREEN_W * 0.55, height: SCREEN_W * 0.55, borderRadius: 20, overflow: 'hidden' },
  scanPreviewImage: { width: '100%', height: '100%', resizeMode: 'cover' },
  scanOverlay: { ...StyleSheet.absoluteFill, borderWidth: 2, borderColor: PURPLE, borderRadius: 20 },
  scanBracketTL: { position: 'absolute', top: -1, left: -1, width: 28, height: 28, borderTopWidth: 4, borderLeftWidth: 4, borderColor: '#fff' },
  scanBracketTR: { position: 'absolute', top: -1, right: -1, width: 28, height: 28, borderTopWidth: 4, borderRightWidth: 4, borderColor: '#fff' },
  scanBracketBL: { position: 'absolute', bottom: -1, left: -1, width: 28, height: 28, borderBottomWidth: 4, borderLeftWidth: 4, borderColor: '#fff' },
  scanBracketBR: { position: 'absolute', bottom: -1, right: -1, width: 28, height: 28, borderBottomWidth: 4, borderRightWidth: 4, borderColor: '#fff' },
  scanningTitle: { marginTop: 20, fontSize: 19, fontWeight: '800', color: TEXT_PRIMARY },
  scanningHint: { marginTop: 6, fontSize: 13, color: TEXT_MUTED },
  dotsRow: { flexDirection: 'row', marginTop: 22, gap: 8 },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: PURPLE },

  // ── Result ──
  resultContainer: {
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: NAVBAR_HEIGHT + 20,
    paddingBottom: 40,
  }, resultImageWrapper: { width: SCREEN_W, height: SCREEN_W, backgroundColor: '#000', overflow: 'hidden', marginHorizontal: -20 },
  resultImage: { width: '100%', height: '100%' },
  resultImageGradient: { position: 'absolute', bottom: 0, left: 0, right: 0, height: 80, backgroundColor: 'rgba(10,10,15,0.75)' },
  resultImageNameTag: { position: 'absolute', bottom: 16, left: 16, flexDirection: 'row', alignItems: 'center', gap: 8, backgroundColor: 'rgba(10,10,15,0.7)', paddingHorizontal: 12, paddingVertical: 6, borderRadius: 20, borderWidth: 1, borderColor: BBOX_COLOR },
  resultImageName: { color: '#fff', fontSize: 14, fontWeight: '800', letterSpacing: 0.3 },

  portraitCard: {
    width: '100%', backgroundColor: CARD, borderRadius: 24, borderWidth: 1, borderColor: BORDER,
    alignItems: 'center', paddingVertical: 28, paddingHorizontal: 24, marginBottom: 20, marginTop: 16,
  },
  corner: { position: 'absolute', width: 20, height: 20, borderColor: PURPLE },
  cornerTL: { top: 12, left: 12, borderTopWidth: 2, borderLeftWidth: 2, borderTopLeftRadius: 6 },
  cornerTR: { top: 12, right: 12, borderTopWidth: 2, borderRightWidth: 2, borderTopRightRadius: 6 },
  cornerBL: { bottom: 12, left: 12, borderBottomWidth: 2, borderLeftWidth: 2, borderBottomLeftRadius: 6 },
  cornerBR: { bottom: 12, right: 12, borderBottomWidth: 2, borderRightWidth: 2, borderBottomRightRadius: 6 },

  avatarWrapper: { alignItems: 'center', justifyContent: 'center', marginBottom: 16 },
  avatarGlow: { position: 'absolute', width: 110, height: 110, borderRadius: 55, backgroundColor: PURPLE_GLOW },
  avatarImage: { width: 90, height: 90, borderRadius: 45, borderWidth: 3, borderColor: PURPLE },
  avatarCircle: { width: 90, height: 90, borderRadius: 45, backgroundColor: '#1c1830', borderWidth: 3, borderColor: PURPLE, alignItems: 'center', justifyContent: 'center' },
  avatarEmoji: { fontSize: 44 },
  verifiedBadge: { position: 'absolute', bottom: 0, right: -4, width: 26, height: 26, borderRadius: 13, backgroundColor: GREEN, alignItems: 'center', justifyContent: 'center', borderWidth: 2, borderColor: CARD },
  verifiedText: { color: '#fff', fontSize: 12, fontWeight: '800' },

  resultName: { fontSize: 26, fontWeight: '800', color: TEXT_PRIMARY, letterSpacing: 0.5, marginBottom: 6 },
  idRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 18, gap: 8 },
  idLabel: { fontSize: 11, fontWeight: '700', color: PURPLE_LIGHT, letterSpacing: 2, textTransform: 'uppercase', backgroundColor: PURPLE_GLOW, paddingHorizontal: 8, paddingVertical: 3, borderRadius: 6, borderWidth: 1, borderColor: PURPLE },
  idValue: { fontSize: 13, color: '#8070a0', letterSpacing: 1 },

  confidenceRow: { width: '100%', flexDirection: 'row', alignItems: 'center', marginBottom: 18 },
  confidenceLabel: { fontSize: 12, color: TEXT_MUTED, width: 80, letterSpacing: 0.5 },
  confidenceBar: { flex: 1, height: 6, backgroundColor: '#1c1830', borderRadius: 3, marginHorizontal: 10, overflow: 'hidden' },
  confidenceFill: { height: 6, borderRadius: 3, backgroundColor: PURPLE },
  confidenceValue: { fontSize: 14, fontWeight: '800', color: PURPLE_LIGHT, width: 52, textAlign: 'right' },

  cardDivider: { width: '100%', height: 1, backgroundColor: BORDER, marginBottom: 16 },

  bboxContainer: { width: '100%', backgroundColor: '#0f0c1a', borderRadius: 14, padding: 14, marginBottom: 16, borderWidth: 1, borderColor: '#2a1f4a' },
  bboxTitleRow: { flexDirection: 'row', alignItems: 'center', marginBottom: 12, gap: 6 },
  bboxDot: { width: 7, height: 7, borderRadius: 3.5, backgroundColor: BBOX_COLOR },
  bboxTitle: { color: BBOX_COLOR, fontSize: 12, fontWeight: '700', letterSpacing: 1 },
  bboxGrid: { flexDirection: 'row', justifyContent: 'space-between' },
  bboxCell: { flex: 1, alignItems: 'center', backgroundColor: '#1c1830', borderRadius: 10, paddingVertical: 10, marginHorizontal: 3 },
  bboxKey: { fontSize: 10, color: TEXT_MUTED, fontWeight: '700', letterSpacing: 1, marginBottom: 4 },
  bboxVal: { fontSize: 14, color: TEXT_PRIMARY, fontWeight: '800' },

  metaRow: { width: '100%', flexDirection: 'row', alignItems: 'center', marginBottom: 14 },
  metaIcon: { fontSize: 15, marginRight: 10 },
  metaKey: { flex: 1, fontSize: 12, color: TEXT_MUTED, letterSpacing: 0.3 },
  metaValue: { fontSize: 13, color: '#c0b0e0', fontWeight: '600' },

  statusPill: { flexDirection: 'row', alignItems: 'center', backgroundColor: 'rgba(34,197,94,0.12)', borderRadius: 20, paddingHorizontal: 16, paddingVertical: 8, borderWidth: 1, borderColor: 'rgba(34,197,94,0.3)', marginTop: 4 },
  statusDot: { width: 8, height: 8, borderRadius: 4, backgroundColor: GREEN, marginRight: 8 },
  statusText: { fontSize: 13, fontWeight: '700', color: GREEN, letterSpacing: 0.5 },
});